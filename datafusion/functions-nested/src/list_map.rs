// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ScalarUDFImpl`] definitions for array_length function.

use arrow::array::AsArray;
use arrow_array::{ArrayRef, FixedSizeListArray, LargeListArray, ListArray, RecordBatch};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{exec_err, Result};
use datafusion_expr::{
    ColumnarValue, Documentation, LambdaArgument, ReturnInfo, ScalarFunctionArgMetadata, ScalarUDFImpl, Signature, Volatility
};
use datafusion_macros::user_doc;
use std::any::Any;
use std::sync::Arc;

make_udf_expr_and_func!(
    ListMap,
    list_map,
    array,
    "maps the values of a list",
    list_map_udf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "maps the values of a list",
    syntax_example = "list_map(array, x -> x*2)",
    sql_example = r#"```sql
> select list_map([1, 2, 3, 4, 5], x -> x*2);
+-------------------------------------------+
| list_map([1, 2, 3, 4, 5], x -> x*2)       |
+-------------------------------------------+
| [2, 4, 6, 8, 10]                          |
+-------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "List expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "lambda", description = "Lambda")
)]
#[derive(Debug)]
pub struct ListMap {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ListMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ListMap {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            aliases: vec![String::from("array_map")],
        }
    }
}

impl ScalarUDFImpl for ListMap {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "list_map"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        unreachable!()
    }

    fn return_type_from_args(
        &self,
        args: datafusion_expr::ReturnTypeArgs,
    ) -> Result<ReturnInfo> {
        let field = Arc::new(Field::new_list_field(
            args.arg_types[1].clone(),
            args.nullables[1],
        ));

        let return_type = match &args.arg_types[0] {
            DataType::List(_) => DataType::List(field),
            DataType::LargeList(_) => DataType::LargeList(field),
            DataType::FixedSizeList(_, size) => DataType::FixedSizeList(field, *size),
            _ => unreachable!(),
        };

        Ok(ReturnInfo::new(return_type, args.nullables[0]))
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let list = &args.args[0];
        
        let (args, body, _captures) = args.lambdas[1].as_ref().unwrap();

        enum ListType {
            List(OffsetBuffer<i32>),
            LargeList(OffsetBuffer<i64>),
            FixedSizeList(i32),
        }

        let (field, list, values, nulls) = match list.data_type() {
            DataType::List(_) => {
                let (field, offsets, values, nulls) =
                    list.to_array(1)?.as_list::<i32>().clone().into_parts();

                (field, ListType::List(offsets), values, nulls)
            }
            DataType::LargeList(_) => {
                let (field, offsets, values, nulls) =
                    list.to_array(1)?.as_list::<i64>().clone().into_parts();

                (field, ListType::LargeList(offsets), values, nulls)
            }
            DataType::FixedSizeList(_, _) => {
                let (field, size, values, nulls) =
                    list.to_array(1)?.as_fixed_size_list().clone().into_parts();

                (field, ListType::FixedSizeList(size), values, nulls)
            }
            _ => unreachable!(),
        };

        let schema = Schema::new(vec![Field::new(
            &args[0],
            field.data_type().clone(),
            field.is_nullable(),
        )]);

        let nullable = body.nullable(&schema)?;

        let lambda_batch = RecordBatch::try_new(Arc::new(schema), vec![values])?;

        let mapped_values = body
            .evaluate(&lambda_batch)?
            .into_array(lambda_batch.num_rows())?;

        let field = Arc::new(Field::new_list_field(
            mapped_values.data_type().clone(),
            nullable,
        ));

        let list = match list {
            ListType::List(offsets) => {
                Arc::new(ListArray::new(field, offsets, mapped_values, nulls)) as ArrayRef
            }
            ListType::LargeList(offsets) => {
                Arc::new(LargeListArray::new(field, offsets, mapped_values, nulls))
            }
            ListType::FixedSizeList(size) => {
                Arc::new(FixedSizeListArray::new(field, size, mapped_values, nulls))
            }
        };

        Ok(ColumnarValue::Array(Arc::new(list)))
    }

    fn lambdas_arguments(
        &self,
        args: &[ScalarFunctionArgMetadata]
    ) -> Result<Vec<Option<Vec<LambdaArgument>>>> {
        let [ScalarFunctionArgMetadata::Value(list), ScalarFunctionArgMetadata::Lambda(_)] = args else {
            return exec_err!("{} expects a value follewed by a lambda, got {:?}", self.name(), args)
        };

        let field = match list {
            DataType::List(field) => field,
            DataType::LargeList(field) => field,
            DataType::FixedSizeList(field, _) => field,
            _ => unreachable!(),
        };

        let value = LambdaArgument::new(field.data_type().clone(), field.is_nullable());

        Ok(vec![None, Some(vec![value])])
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
