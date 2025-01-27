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
use arrow_array::{ListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr::ScalarFunctionArgument;
use datafusion_expr::{
    ColumnarValue, ColumnarValueOrLambda, Documentation, ExprSchemable, ReturnInfo,
    ScalarUDFImpl, Signature, Volatility,
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
            signature: Signature::variadic_any(Volatility::Immutable),
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
        let (field, is_large) = match &args.arg_types[0] {
            DataType::List(field) => (field, false),
            DataType::LargeList(field) => (field, true),
            _ => unreachable!(),
        };

        let (args_names, expr) = args.lambda_arguments[1].unwrap();

        let schema = Schema::new(vec![Field::new(
            &args_names[0],
            field.data_type().clone(),
            field.is_nullable(),
        )]);

        let (data_type, value_nullable) =
            expr.data_type_and_nullable(&DFSchema::try_from(schema).unwrap())?;

        let field = Arc::new(Field::new_list_field(data_type, value_nullable));

        let return_type = if is_large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        };

        Ok(ReturnInfo::new(return_type, args.nullables[0]))
    }

    fn invoke_with_lambda_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs<ColumnarValueOrLambda>,
    ) -> Result<ColumnarValue> {
        let [ColumnarValueOrLambda::Value(list), ColumnarValueOrLambda::Lambda { args, body }] =
            args.args.as_slice()
        else {
            unreachable!()
        };

        let (field, offsets, values, nulls) =
            list.to_array(1)?.as_list::<i32>().clone().into_parts();

        let schema = Schema::new(vec![Field::new(
            &args[0],
            field.data_type().clone(),
            field.is_nullable(),
        )]);

        let lambda_batch = RecordBatch::try_new(Arc::new(schema), vec![values])?;

        let values2 = body
            .evaluate(&lambda_batch)?
            .into_array(lambda_batch.num_rows())?;

        let field = Arc::new(Field::new_list_field(values2.data_type().clone(), values2.null_count() > 0));

        let list = ListArray::new(field, offsets, values2, nulls);

        Ok(ColumnarValue::Array(Arc::new(list)))
    }

    fn lambdas_schemas(
        &self,
        args: &[ScalarFunctionArgument],
        schema: &dyn datafusion_common::ExprSchema,
    ) -> Result<Vec<Option<Schema>>> {
        let [ScalarFunctionArgument::Expr(list), ScalarFunctionArgument::Lambda { arg_names, expr: _ }] =
            args
        else {
            unreachable!()
        };

        let (data_type, _null) = list.data_type_and_nullable(schema)?;
        
        let field = match data_type {
            DataType::List(field) => field,
            DataType::LargeList(field) => field,
            _ => unreachable!()
        };

        let schema = Schema::new(vec![Field::new(
            &arg_names[0],
            field.data_type().clone(),
            field.is_nullable(),
        )]);

        Ok(vec![None, Some(schema)])
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
