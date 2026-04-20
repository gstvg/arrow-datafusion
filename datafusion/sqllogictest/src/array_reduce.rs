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

//! [`HigherOrderUDF`] definitions for array_reduce function.

use std::{fmt::Debug, sync::Arc};

use arrow::{
    array::{
        Array, ArrowPrimitiveType, AsArray, BooleanArray, FixedSizeListArray,
        PrimitiveArray, Scalar, new_null_array,
    },
    compute::{kernels, take},
    datatypes::{ArrowNativeType, DataType, FieldRef, UInt32Type, UInt64Type},
};
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::logical_expr::{
    ColumnarValue, HigherOrderFunctionArgs, HigherOrderReturnFieldArgs,
    HigherOrderSignature, HigherOrderUDF, LambdaParametersProgress, ValueOrLambda,
    Volatility,
};

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayReduce {
    signature: HigherOrderSignature,
    aliases: Vec<String>,
}

impl Default for ArrayReduce {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayReduce {
    pub fn new() -> Self {
        Self {
            signature: HigherOrderSignature::user_defined(Volatility::Immutable)
                .with_cast_values_for_lambdas(),
            aliases: vec![String::from("list_reduce")],
        }
    }
}

impl HigherOrderUDF for ArrayReduce {
    fn name(&self) -> &str {
        "array_reduce"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &HigherOrderSignature {
        &self.signature
    }

    fn coerce_value_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [list, initial_value] = arg_types else {
            return plan_err!(
                "{} function requires 2 value arguments, got {}",
                self.name(),
                arg_types.len()
            );
        };

        let coerced_list = match list {
            DataType::FixedSizeList(_, _) => list.clone(),
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::ListView(field)
            | DataType::LargeListView(field) => {
                DataType::FixedSizeList(Arc::clone(field), 2)
            }
            _ => {
                return plan_err!(
                    "{} expected a list as first argument, got {}",
                    self.name(),
                    list
                );
            }
        };

        Ok(vec![coerced_list, initial_value.clone()])
    }

    fn lambda_parameters(
        &self,
        step: usize,
        fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
    ) -> Result<LambdaParametersProgress> {
        // optional finish not supported for simplicity
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(initial_value),
            ValueOrLambda::Lambda(merge),
            ValueOrLambda::Lambda(_finish),
        ] = fields
        else {
            return plan_err!(
                "reduce expects a list value, then an initial value, then a merge lambda and finally a finish lambda"
            );
        };

        let list_field = match list.data_type() {
            DataType::FixedSizeList(field, _) => field,
            _ => return plan_err!("reduce expects a list as it's first argument"),
        };

        Ok(match (step, merge) {
            (0, None) => {
                // at the first step, we use the initial_value as merge accumulator,
                // and return None for finish since we don't know the output of merge
                LambdaParametersProgress::Partial(vec![
                    // merge
                    Some(vec![Arc::clone(initial_value), Arc::clone(list_field)]),
                    // finish
                    None,
                ])
            }
            (1, Some(accumulator)) | (0, Some(accumulator)) => {
                // now we can use the merge output as it's accumulator and
                // as the finish parameter
                LambdaParametersProgress::Complete(vec![
                    // merge
                    vec![Arc::clone(accumulator), Arc::clone(list_field)],
                    // finish
                    vec![Arc::clone(accumulator)],
                ])
            }
            (1, None) => {
                return plan_err!("merge should be resolved at reduce step 1 (0-based)");
            }
            _ => todo!(),
        })
    }

    fn cast_values_for_lambdas(
        &self,
        fields: &[ValueOrLambda<FieldRef, FieldRef>],
    ) -> Result<Vec<FieldRef>> {
        // optional finish not supported for simplicity
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(_initial_value),
            ValueOrLambda::Lambda(merge),
            ValueOrLambda::Lambda(_finish),
        ] = fields
        else {
            return plan_err!(
                "reduce expects a list value, then an initial value, then a merge lambda and finally a finish lambda"
            );
        };

        // cast the initial value to the output of the merge lambda
        Ok(vec![Arc::clone(list), Arc::clone(merge)])
    }

    fn return_field_from_args(
        &self,
        args: HigherOrderReturnFieldArgs,
    ) -> Result<FieldRef> {
        // optional finish not supported for simplicity
        let [
            ValueOrLambda::Value(_list),
            ValueOrLambda::Value(_initial_value),
            ValueOrLambda::Lambda(_merge),
            ValueOrLambda::Lambda(finish),
        ] = args.arg_fields
        else {
            return plan_err!(
                "reduce expects a list value, then an initial value, then a merge lambda and finally a finish lambda"
            );
        };

        Ok(Arc::clone(finish))
    }

    fn invoke_with_args(&self, args: HigherOrderFunctionArgs) -> Result<ColumnarValue> {
        // optional finish not supported for simplicity
        let [
            ValueOrLambda::Value(list),
            ValueOrLambda::Value(initial_value),
            ValueOrLambda::Lambda(merge),
            ValueOrLambda::Lambda(finish),
        ] = &args.args[..]
        else {
            return exec_err!(
                "reduce expects a list value, then an initial value, then a merge lambda and finally a finish lambda"
            );
        };

        let list_array = list.to_array(args.number_rows)?;
        let fsl = list_array.as_fixed_size_list();

        let mut fsl_values = Arc::clone(fsl.values());
        let mut acc = initial_value.to_array(args.number_rows)?;

        let indices: &dyn Array = if u32::try_from(fsl_values.len()).is_ok() {
            &indices::<UInt32Type>(fsl)
        } else {
            &indices::<UInt64Type>(fsl)
        };

        for _ in 0..fsl.value_length() {
            let value = take(&fsl_values, indices, None)?;

            acc = merge
                .evaluate(&[&|| Ok(Arc::clone(&acc)), &|| Ok(Arc::clone(&value))])?
                .into_array(args.number_rows)?;

            // slice values so that indices point to the next element instead of recomputing the indices
            fsl_values = fsl_values.slice(1, fsl_values.len() - 1);
        }

        let finished = finish.evaluate(&[&|| Ok(Arc::clone(&acc))])?;

        match fsl.nulls() {
            Some(nulls) => {
                let finished = finished.into_array(nulls.len() - nulls.null_count())?;

                Ok(ColumnarValue::Array(kernels::merge::merge(
                    &BooleanArray::new(nulls.inner().clone(), None),
                    &finished,
                    &Scalar::new(new_null_array(finished.data_type(), 1)),
                )?))
            }
            None => Ok(finished),
        }
    }
}

fn indices<T: ArrowPrimitiveType>(fsl: &FixedSizeListArray) -> PrimitiveArray<T> {
    let values = match fsl.nulls() {
        Some(nulls) => {
            let mut vec = Vec::with_capacity(nulls.len() - nulls.null_count());

            for (start, end) in nulls.valid_slices() {
                vec.extend(
                    (start..end)
                        .map(|i| T::Native::usize_as(i * fsl.value_length() as usize)),
                );
            }

            vec.into()
        }
        None => (0..fsl.values().len())
            .step_by(fsl.value_length() as usize)
            .map(T::Native::usize_as)
            .collect(),
    };

    PrimitiveArray::new(values, None)
}
