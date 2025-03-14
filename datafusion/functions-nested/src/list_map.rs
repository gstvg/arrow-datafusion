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

//! [`ScalarUDFImpl`] definitions for list_map function.

use arrow::{
    array::{
        Array, ArrayRef, ArrowPrimitiveType, AsArray, FixedSizeListArray, LargeListArray,
        ListArray, PrimitiveArray, RecordBatch,
    },
    buffer::OffsetBuffer,
    compute::take_record_batch,
    datatypes::{
        ArrowNativeType, DataType, Field, Int32Type, Int64Type, Schema, UInt32Type,
    },
};
use datafusion_common::{exec_err, Result};
use datafusion_expr::{
    ColumnarValue, Documentation, LambdaArgument, ReturnInfo, ScalarFunctionArgMetadata,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, ValueOrLambda, Volatility,
};
use datafusion_functions::utils::take_function_args;
use datafusion_macros::user_doc;
use std::iter::repeat_n;
use std::{any::Any, sync::Arc};

make_udf_expr_and_func!(
    ListMap,
    list_map,
    array lambda,
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
        if args.lambdas != [false, true] {
            return exec_err!("");
        }

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

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let args = args.into_lambda_args();

        let [ValueOrLambda::Value(list), ValueOrLambda::Lambda(lambda)] =
            take_function_args("list_map", args)?
        else {
            unreachable!()
        };

        println!("captures={}\n", lambda.captures.schema_ref());

        enum ListType {
            List(OffsetBuffer<i32>),
            LargeList(OffsetBuffer<i64>),
            FixedSizeList(i32),
        }

        let (field, list, values, nulls, element_indices, captures) = match list
            .data_type()
        {
            DataType::List(_) => {
                let (field, offsets, values, nulls) =
                    list.to_array(1)?.as_list::<i32>().clone().into_parts();

                let element_indices = make_list_element_indices::<Int32Type>(&offsets);

                let captured = if lambda.captures.num_columns() > 0 {
                    let array_indices = make_list_array_indices::<Int32Type>(&offsets);

                    println!("{:?}", array_indices.values().as_ref());
                    take_record_batch(&lambda.captures, &array_indices)?
                } else {
                    lambda.captures.clone()
                };

                (
                    field,
                    ListType::List(offsets),
                    values,
                    nulls,
                    Arc::new(element_indices) as ArrayRef,
                    captured,
                )
            }
            DataType::LargeList(_) => {
                let (field, offsets, values, nulls) =
                    list.to_array(1)?.as_list::<i64>().clone().into_parts();

                let element_indices = make_list_element_indices::<Int64Type>(&offsets);

                let captured = if lambda.captures.num_columns() > 0 {
                    let array_indices = make_list_array_indices::<Int64Type>(&offsets);

                    take_record_batch(&lambda.captures, &array_indices)?
                } else {
                    lambda.captures.clone()
                };

                (
                    field,
                    ListType::LargeList(offsets),
                    values,
                    nulls,
                    Arc::new(element_indices) as ArrayRef,
                    captured,
                )
            }
            DataType::FixedSizeList(_, _) => {
                let list = list.to_array(1)?;
                let (field, size, values, nulls) =
                    list.as_fixed_size_list().clone().into_parts();

                let element_indices = make_fsl_element_indices(size, list.len());

                let captured = if lambda.captures.num_columns() > 0 {
                    let array_indices = make_fsl_array_indices(size, list.len());

                    take_record_batch(&lambda.captures, &array_indices)?
                } else {
                    lambda.captures.clone()
                };

                (
                    field,
                    ListType::FixedSizeList(size),
                    values,
                    nulls,
                    Arc::new(element_indices) as ArrayRef,
                    captured,
                )
            }
            _ => unreachable!(),
        };

        let args = [
            lambda.args_names.first().map(|values_name| {
                Arc::new(
                    Field::new(
                        values_name,
                        field.data_type().clone(),
                        field.is_nullable(),
                    )
                    .with_metadata(field.metadata().clone()), //really?
                )
            }),
            lambda.args_names.get(1).map(|index_name| {
                Arc::new(Field::new(
                    index_name,
                    element_indices.data_type().clone(),
                    false,
                ))
            }),
        ];

        let lambda_batch = RecordBatch::try_new(
            Arc::new(Schema::new(
                args.into_iter().flatten()
                    .chain(captures
                        .schema()
                        .fields()
                        .iter()
                        .cloned())
                    .collect::<Vec<_>>(),
            )),
            [
                &[values, element_indices][..lambda.args_names.len()],
                captures.columns(),
            ]
            .concat(),
        )?;

        let mapped_values = lambda
            .body
            .evaluate(&lambda_batch)?
            .into_array(lambda_batch.num_rows())?;

        let field = Arc::new(Field::new_list_field(
            mapped_values.data_type().clone(),
            lambda.body.nullable(lambda_batch.schema_ref())?,
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
        args: &[ScalarFunctionArgMetadata],
    ) -> Result<Vec<Option<Vec<LambdaArgument>>>> {
        let [ScalarFunctionArgMetadata::Value(list), ScalarFunctionArgMetadata::Lambda(_)] =
            args
        else {
            return exec_err!(
                "{} expects a value follewed by a lambda, got {:?}",
                self.name(),
                args
            );
        };

        let (field, index_type) = match list {
            DataType::List(field) => (field, DataType::Int32),
            DataType::LargeList(field) => (field, DataType::Int64),
            DataType::FixedSizeList(field, _) => (field, DataType::UInt32),
            _ => return exec_err!("expected list, got {list}"),
        };

        let value = LambdaArgument::new(field.data_type().clone(), field.is_nullable())
            .with_metadata(field.metadata().clone());
        let index = LambdaArgument::new(index_type, false);

        Ok(vec![None, Some(vec![value, index])])
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn make_list_array_indices<T: ArrowPrimitiveType>(
    offsets: &[T::Native],
) -> PrimitiveArray<T> {
    let mut indices =
        Vec::with_capacity(offsets.last().unwrap().as_usize() - offsets[0].as_usize());

    for (i, (&start, &end)) in std::iter::zip(offsets, &offsets[1..]).enumerate() {
        indices.extend(repeat_n(T::Native::usize_as(i), end.as_usize() - start.as_usize()));
    }

    PrimitiveArray::new(indices.into(), None)
}

fn make_list_element_indices<T: ArrowPrimitiveType>(
    offsets: &[T::Native],
) -> PrimitiveArray<T> {
    let mut indices = vec![
        T::default_value();
        offsets.last().unwrap().as_usize() - offsets[0].as_usize()
    ];

    for (&start, &end) in std::iter::zip(offsets, &offsets[1..]) {
        for i in 0..end.as_usize() - start.as_usize() {
            indices[start.as_usize() + i] = T::Native::usize_as(i);
        }
    }

    PrimitiveArray::new(indices.into(), None)
}

fn make_fsl_array_indices(
    list_size: i32,
    array_len: usize,
) -> PrimitiveArray<UInt32Type> {
    let mut indices = vec![0; list_size as usize * array_len];

    for i in 0..array_len {
        for j in 0..list_size as usize {
            indices[i + j] = i as u32;
        }
    }

    PrimitiveArray::new(indices.into(), None)
}

fn make_fsl_element_indices(
    list_size: i32,
    array_len: usize,
) -> PrimitiveArray<UInt32Type> {
    let mut indices = vec![0; list_size as usize * array_len];

    for i in 0..array_len {
        for j in 0..list_size as usize {
            indices[i + j] = j as u32;
        }
    }

    PrimitiveArray::new(indices.into(), None)
}

/*
Expr::Lambda
Lambda PhysicalExpr
Expr::Lambda -> PhysicalExpr
Expr::*_with_lambdas
ExprSchemaNode = ExprContext<Arc<Schema>>
extend ScalarUDF[Impl]
list_map
capture support
*/
