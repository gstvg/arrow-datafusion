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

use arrow::datatypes::FieldRef;
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
use datafusion_common::{
    exec_err, utils::take_function_args, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::{
    ColumnarValue, Documentation, LambdaParameter, ScalarFunctionArgMetadata,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, ValueOrLambda, ValueOrLambdaField,
    Volatility,
};
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

    fn return_field_from_args(
        &self,
        args: datafusion_expr::ReturnFieldArgs,
    ) -> Result<Field> {
        let args = args.lambda_args();

        let [ValueOrLambdaField::Value(list), ValueOrLambdaField::Lambda(lambda)] =
            take_function_args(self.name(), args)?
        else {
            return exec_err!("");
        };

        // lambda is the resulting field of executing the lambda body
        // with the parameters returned in lambdas_parameters
        let field = Arc::new(lambda.clone().with_name(Field::LIST_FIELD_DEFAULT_NAME));

        let return_type = match list.data_type() {
            DataType::List(_) => DataType::List(field),
            DataType::LargeList(_) => DataType::LargeList(field),
            DataType::FixedSizeList(_, size) => DataType::FixedSizeList(field, *size),
            _ => unreachable!(),
        };

        Ok(Field::new("", return_type, list.is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;

        // into_lambda_args allows the convenient match below
        let args = args.into_lambda_args();

        let [ValueOrLambda::Value(list_value), ValueOrLambda::Lambda(lambda)] =
            take_function_args(self.name(), args)?
        else {
            unreachable!()
        };

        // println!("captures={}\n", lambda.captures.schema_ref());

        let list_array = list_value.to_array(num_rows)?;
        let list_type = ListType::try_from(list_array.as_ref())?;

        // if any column got captured, we need to adjust it to the values arrays,
        // duplicating values of list with mulitple values and removing values of empty lists
        // array_indices is not cheap so is important to avoid it when no column is captured
        let adjusted_captures = lambda
            .captures
            .map(|captures| take_record_batch(&captures, &list_type.array_indices()))
            .transpose()?;

        // pass closures to merge_captures_with_lazy_args so that it calls only the needed ones based on the number of arguments
        // avoiding unnecessary computations
        let values_param = || Ok(Arc::clone(list_type.values()));
        let indices_param = || Ok(list_type.elements_indices());

        // the order of the merged schema is an unspecified implementation detail that may change in the future,
        // using this function is the correct way to merge as it return the correct ordering and will change in sync
        // the implementation without the need for fixes. It also computes only the parameters requested
        let lambda_batch = merge_captures_with_lazy_args(
            adjusted_captures.as_ref(),
            &lambda.fields, // ScalarUDF already merged the fields returned in lambdas_parameters with the parameters names definied in the lambda, so we don't need to
            &[&values_param, &indices_param],
        )?;

        // call the transforming expression with the record batch composed of the list values merged with captured columns
        let mapped_values = lambda
            .body
            .evaluate(&lambda_batch)?
            .into_array(lambda_batch.num_rows())?;

        //TODO: should metadata be passed? If so, with the same keys or prefixed/suffixed?
        let field = Arc::new(Field::new_list_field(
            mapped_values.data_type().clone(),
            lambda.body.nullable(lambda_batch.schema_ref())?,
        ));

        let mapped_list = match list_type {
            ListType::List(list) => Arc::new(ListArray::new(
                field,
                list.offsets().clone(),
                mapped_values,
                list.nulls().cloned(),
            )) as ArrayRef,
            ListType::LargeList(large_list) => Arc::new(LargeListArray::new(
                field,
                large_list.offsets().clone(),
                mapped_values,
                large_list.nulls().cloned(),
            )),
            ListType::FixedSizeList(fixed_size_list) => {
                Arc::new(FixedSizeListArray::new(
                    field,
                    fixed_size_list.value_length(),
                    mapped_values,
                    fixed_size_list.nulls().cloned(),
                ))
            }
        };

        Ok(ColumnarValue::Array(mapped_list))
    }

    fn lambdas_parameters(
        &self,
        args: &[ScalarFunctionArgMetadata],
    ) -> Result<Vec<Option<Vec<LambdaParameter>>>> {
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

        // we don't need to omit the index in the case the lambda don't specify, e.g. list_map([], v -> v*2),
        // nor check whether the lambda contains more than two parameters, e.g. list_map([], (v, i, j) -> v+i+j),
        // as datafusion will do that for us
        let value = LambdaParameter::new(field.data_type().clone(), field.is_nullable())
            .with_metadata(field.metadata().clone());
        let index = LambdaParameter::new(index_type, false);

        Ok(vec![None, Some(vec![value, index])])
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn make_list_array_indices<T: ArrowPrimitiveType>(
    offsets: &OffsetBuffer<T::Native>,
) -> PrimitiveArray<T> {
    let mut indices =
        Vec::with_capacity(offsets.last().unwrap().as_usize() - offsets[0].as_usize());

    for (i, (&start, &end)) in std::iter::zip(&offsets[..], &offsets[1..]).enumerate() {
        indices.extend(repeat_n(
            T::Native::usize_as(i),
            end.as_usize() - start.as_usize(),
        ));
    }

    PrimitiveArray::new(indices.into(), None)
}

fn make_list_element_indices<T: ArrowPrimitiveType>(
    offsets: &OffsetBuffer<T::Native>,
) -> PrimitiveArray<T> {
    let mut indices = vec![
        T::default_value();
        offsets.last().unwrap().as_usize() - offsets[0].as_usize()
    ];

    for (&start, &end) in std::iter::zip(&offsets[..], &offsets[1..]) {
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

/// Merge the lambda body captured columns with it's arguments
/// Datafusion relies on an unspecified field ordering implemented in this function
/// As such, this is the only correct way to merge the captured values with the arguments
/// The number of args should not be lower than the number of params
///
/// See also merge_captures_with_lazy_args and merge_captures_with_boxed_lazy_args that lazily
/// computes only the necessary arguments to match the number of params
pub fn merge_captures_with_args(
    captures: Option<&RecordBatch>,
    params: &[FieldRef],
    args: &[ArrayRef],
) -> Result<RecordBatch> {
    if args.len() < params.len() {
        return exec_err!(
            "merge_captures_with_args called with {} params but with {} args",
            params.len(),
            args.len()
        );
    }

    // the order of the merged batch must be kept in sync with ScalarFunction::lambdas_schemas variants
    let (fields, columns) = match captures {
        Some(captures) => {
            let fields = params
                .iter()
                .cloned()
                .chain(captures.schema().fields().iter().cloned())
                .collect::<Vec<_>>();

            let columns = [args, captures.columns()].concat();

            (fields, columns)
        }
        None => (params.to_vec(), args.to_vec()),
    };

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Lazy version of merge_captures_with_args that receives closures to compute the arguments,
/// and calls only the necessary to match the number of params
pub fn merge_captures_with_lazy_args(
    captures: Option<&RecordBatch>,
    params: &[FieldRef],
    args: &[&dyn Fn() -> Result<ArrayRef>],
) -> Result<RecordBatch> {
    merge_captures_with_args(
        captures,
        params,
        &args
            .iter()
            .take(params.len())
            .map(|arg| arg())
            .collect::<Result<Vec<_>>>()?,
    )
}

/// Variation of merge_captures_with_lazy_args that take boxed closures
pub fn merge_captures_with_boxed_lazy_args(
    captures: Option<&RecordBatch>,
    params: &[FieldRef],
    args: &[Box<dyn Fn() -> Result<ArrayRef>>],
) -> Result<RecordBatch> {
    merge_captures_with_args(
        captures,
        params,
        &args
            .iter()
            .take(params.len())
            .map(|arg| arg())
            .collect::<Result<Vec<_>>>()?,
    )
}

trait LazyArgument {
    fn compute(self) -> Result<ArrayRef>;
}

impl LazyArgument for &ArrayRef {
    fn compute(self) -> Result<ArrayRef> {
        Ok(Arc::clone(self))
    }
}

impl LazyArgument for Box<dyn FnOnce() -> Result<ArrayRef>> {
    fn compute(self) -> Result<ArrayRef> {
        self()
    }
}

impl LazyArgument for &dyn Fn() -> Result<ArrayRef> {
    fn compute(self) -> Result<ArrayRef> {
        self()
    }
}

enum ListType<'a> {
    List(&'a ListArray),
    LargeList(&'a LargeListArray),
    FixedSizeList(&'a FixedSizeListArray),
}

impl<'a> TryFrom<&'a dyn Array> for ListType<'a> {
    type Error = DataFusionError;

    fn try_from(array: &'a dyn Array) -> std::result::Result<Self, Self::Error> {
        match array.data_type() {
            DataType::List(_) => Ok(ListType::List(array.as_list())),
            DataType::LargeList(_) => Ok(ListType::LargeList(array.as_list())),
            DataType::FixedSizeList(_, _) => {
                Ok(ListType::FixedSizeList(array.as_fixed_size_list()))
            }
            data_type => exec_err!("expected list, got {data_type}"),
        }
    }
}

impl<'a> TryFrom<&'a ScalarValue> for ListType<'a> {
    type Error = DataFusionError;

    fn try_from(value: &'a ScalarValue) -> std::result::Result<Self, Self::Error> {
        match value {
            ScalarValue::List(list) => Ok(ListType::List(list)),
            ScalarValue::LargeList(list) => Ok(ListType::LargeList(list)),
            ScalarValue::FixedSizeList(list) => Ok(ListType::FixedSizeList(list)),
            _ => exec_err!("expected list, got {}", value.data_type()),
        }
    }
}

impl<'a> TryFrom<&'a ColumnarValue> for ListType<'a> {
    type Error = DataFusionError;

    fn try_from(value: &'a ColumnarValue) -> std::result::Result<Self, Self::Error> {
        match value {
            ColumnarValue::Array(array) => array.as_ref().try_into(),
            ColumnarValue::Scalar(scalar_value) => scalar_value.try_into(),
        }
    }
}

impl ListType<'_> {
    fn values(&self) -> &ArrayRef {
        match self {
            ListType::List(list) => list.values(),
            ListType::LargeList(large_list) => large_list.values(),
            ListType::FixedSizeList(fixed_size_list) => fixed_size_list.values(),
        }
    }

    fn array_indices(&self) -> ArrayRef {
        match self {
            ListType::List(list) => {
                Arc::new(make_list_array_indices::<Int32Type>(list.offsets()))
            }
            ListType::LargeList(large_list) => {
                Arc::new(make_list_array_indices::<Int64Type>(large_list.offsets()))
            }
            ListType::FixedSizeList(fixed_size_list) => Arc::new(make_fsl_array_indices(
                fixed_size_list.value_length(),
                fixed_size_list.len(),
            )),
        }
    }

    fn elements_indices(&self) -> ArrayRef {
        match self {
            ListType::List(list) => {
                Arc::new(make_list_element_indices::<Int32Type>(list.offsets()))
            }
            ListType::LargeList(large_list) => {
                Arc::new(make_list_element_indices::<Int64Type>(large_list.offsets()))
            }
            ListType::FixedSizeList(fixed_size_list) => {
                Arc::new(make_fsl_element_indices(
                    fixed_size_list.value_length(),
                    fixed_size_list.len(),
                ))
            }
        }
    }
}

/*
Expr::Lambda
Lambda PhysicalExpr
Expr::Lambda -> PhysicalExpr
Expr::*_with_lambdas_params
PhysicalExpr::*_with_lambdas_params
extend ScalarUDF[Impl]
Expr::*_with_schema
PhysicalExpr::*_with_schema
sql parse+unparse
list_map
remove unsupported comments from lambda docs
capture support
remove unsupported comments from lambda capture docs
*/
