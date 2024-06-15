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

use std::borrow::Cow;
use std::cmp::Ordering;

use arrow::array::{
    make_array, new_empty_array, new_null_array, Array, Int8Array, MutableArrayData, PrimitiveArray
};
use arrow::compute::take;
use arrow::datatypes::{DataType, Int32Type, Int8Type, UnionMode};

use arrow::buffer::NullBuffer;
use datafusion_common::cast::as_union_array;
use datafusion_common::{exec_datafusion_err, exec_err, ExprSchema, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Expr};
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion_physical_expr::scatter;
use itertools::Itertools;

#[derive(Debug)]
pub struct UnionExtractFun {
    signature: Signature,
}

impl Default for UnionExtractFun {
    fn default() -> Self {
        Self::new()
    }
}

impl UnionExtractFun {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for UnionExtractFun {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "union_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        todo!()
    }

    fn return_type_from_exprs(
        &self,
        args: &[Expr],
        _: &dyn ExprSchema,
        arg_types: &[DataType],
    ) -> Result<DataType> {
        let fields = if let DataType::Union(fields, _) = &arg_types[0] {
            fields
        } else {
            return exec_err!(
                "union_extract first argument must be a union, got {} instead",
                arg_types[0]
            );
        };

        let field_name = if let Expr::Literal(ScalarValue::Utf8(Some(field_name))) = &args[1] {
            field_name
        } else {
            return exec_err!(
                "union_extract second argument must be a string literal, got {} instead",
                arg_types[1]
            );
        };

        let field = fields
            .iter()
            .find_map(|(_, field)| {
                if field.name() == field_name {
                    Some(field)
                } else {
                    None
                }
            })
            .ok_or_else(|| exec_datafusion_err!("field_name {field_name} not found on union"))?;

        Ok(field.data_type().clone())
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let union = &args[0];
        let field_name = &args[1];

        match (union, field_name) {
            (
                ColumnarValue::Array(union_array),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(field_name))),
            ) => {
                let union_array = as_union_array(&union_array)?;

                let (fields, mode) = match union_array.data_type() {
                    DataType::Union(fields, mode) => (fields, mode),
                    _ => unreachable!(),
                };

                let type_id = fields
                    .iter()
                    .find(|(_, field)| field.name() == field_name)
                    .map(|(type_id, _)| type_id)
                    .ok_or_else(|| {
                        exec_datafusion_err!("field_name {field_name} not found on union")
                    })?;

                match mode {
                    UnionMode::Sparse => {
                        let sparse = union_array.child(type_id);

                        if fields.len() == 1
                            || union_array.is_empty()
                            || sparse.null_count() == sparse.len()
                            || union_array
                                .type_ids()
                                .iter()
                                .all(|value_type_id| *value_type_id == type_id)
                        {
                            Ok(ColumnarValue::Array(sparse.clone()))
                        } else {
                            let type_ids = <PrimitiveArray<Int8Type>>::new(
                                union_array.type_ids().clone(),
                                None,
                            );

                            let selected = arrow::compute::kernels::cmp::eq(
                                &type_ids,
                                &Int8Array::new_scalar(type_id),
                            )?;

                            let nulls = match sparse.nulls() {
                                Some(nulls) => NullBuffer::union(
                                    Some(nulls),
                                    Some(&selected.into_parts().0.into()),
                                )
                                .unwrap(),
                                None => selected.into_parts().0.into(),
                            };

                            let data = sparse
                                .to_data()
                                .into_builder()
                                .nulls(Some(nulls))
                                .build()?;

                            let array = make_array(data);

                            Ok(ColumnarValue::Array(array))
                        }
                    }
                    UnionMode::Dense => {
                        let dense = union_array.child(type_id);

                        /*
                        fields.len() == 1
                        dense.is_empty()
                        multiple types
                            other types all empty
                        all type_ids match
                        sequential offsets
                            dense.len() == union_array.len()
                            dense.len() > union_array.len()
                        */

                        if union_array.is_empty() {
                            match dense.is_empty() {
                                true => Ok(ColumnarValue::Array(dense.clone())),
                                false => Ok(ColumnarValue::Array(new_empty_array(
                                    dense.data_type(),
                                ))),
                            }
                        } else if dense.is_empty() {
                            Ok(ColumnarValue::Array(new_null_array(
                                dense.data_type(),
                                union_array.len(),
                            )))
                        } else if fields.len() == 1 {
                            let offsets = union_array.offsets().unwrap();

                            let sequential = dense.len() >= union_array.len()
                                && offsets
                                    .windows(2)
                                    .all(|window| window[0] + 1 == window[1]);

                            if sequential {
                                if dense.len() == union_array.len() {
                                    Ok(ColumnarValue::Array(dense.clone()))
                                } else {
                                    Ok(ColumnarValue::Array(
                                        dense.slice(
                                            offsets[0] as usize,
                                            union_array.len(),
                                        ),
                                    ))
                                }
                            } else {
                                let indices = <PrimitiveArray<Int32Type>>::try_new(
                                    offsets.clone(),
                                    None,
                                )?;

                                Ok(ColumnarValue::Array(take(dense, &indices, None)?))
                            }
                        } else {
                            let type_ids = union_array.type_ids();
                            let offsets = union_array.offsets().unwrap();

                            let others_are_empty = fields
                                .iter()
                                .filter(|(field_type_id, _)| *field_type_id != type_id)
                                .all(|(field_type_id, _)| {
                                    union_array.child(field_type_id).is_empty()
                                });

                            if others_are_empty
                                || type_ids
                                    .iter()
                                    .all(|value_type_id| *value_type_id == type_id)
                            {
                                let sequential = dense.len() >= union_array.len()
                                    && offsets
                                        .windows(2)
                                        .all(|window| window[0] + 1 == window[1]);

                                if sequential {
                                    match union_array.len().cmp(&dense.len()) {
                                        Ordering::Less => {
                                            let offsets = union_array.offsets().unwrap();

                                            let start = offsets[0] as usize;

                                            Ok(ColumnarValue::Array(
                                                dense.slice(start, union_array.len()),
                                            ))
                                        }
                                        Ordering::Equal => {
                                            // the union array contains only values of the type we are looking for and the child array len equals to the parent union
                                            Ok(ColumnarValue::Array(dense.clone()))
                                        }
                                        Ordering::Greater => unreachable!(),
                                    }
                                } else {
                                    let offsets = <PrimitiveArray<Int32Type>>::new(
                                        union_array.offsets().unwrap().clone(),
                                        None,
                                    );

                                    Ok(ColumnarValue::Array(take(
                                        &dense, &offsets, None,
                                    )?))
                                }
                            } else {
                                // the union array contains values other than the one we are looking for, we need to scatter the ones we want

                                // dense union array can have child arrays with values which aren't pointed by any offset
                                // meaning that child array will have a length bigger than the number of values of it's type in the parent union array
                                // if so, we need to filter the values which are actualy referenced by an offset, so scatter works correclty

                                let dense_data = dense.to_data();

                                let mut mutable = MutableArrayData::new(vec![&dense_data], true, union_array.len());

                                let mut type_match = false;
                                let mut start = 0;
                                let mut last_offset = 0;

                                for (index, value_type_id) in type_ids.iter().enumerate() {

                                    let offset = offsets[index] as usize;

                                    match (type_match, *value_type_id == type_id) {
                                        (true, true) => {
                                            if last_offset + 1 != offset {
                                                mutable.extend(0, start, offset);

                                                start = offset;
                                            }
                                        },
                                        (true, false) => {
                                            mutable.extend(0, start, offset);

                                            start = offset;
                                            type_match = false;
                                        },
                                        (false, true) => {
                                            mutable.extend_nulls(offset - last_offset);

                                            type_match = true;
                                        },
                                        (false, false) => {},
                                    }

                                    last_offset = offset;
                                }

                                let data = mutable.freeze();
                                return Ok(ColumnarValue::Array(make_array(data)));

                                let type_ids_array = <PrimitiveArray<Int8Type>>::new(
                                    type_ids.clone(),
                                    None,
                                );

                                let selected = arrow::compute::kernels::cmp::eq(
                                    &type_ids_array,
                                    &Int8Array::new_scalar(type_id),
                                )?;

                                let sequential = selected
                                    .values()
                                    .set_indices()
                                    .tuple_windows()
                                    .all(|(a, b)| offsets[a] + 1 == offsets[b]);

                                let truthy = if sequential {
                                    Cow::Borrowed(dense)
                                } else {
                                    let offsets = <PrimitiveArray<Int32Type>>::new(
                                        union_array.offsets().unwrap().clone(),
                                        None,
                                    );

                                    let type_offsets =
                                        arrow::compute::filter(&offsets, &selected)?;

                                    Cow::Owned(take(&dense, &type_offsets, None)?)
                                };

                                Ok(ColumnarValue::Array(scatter(
                                    &selected,
                                    truthy.as_ref(),
                                )?))
                            }
                        }
                    }
                }
            }
            (
                ColumnarValue::Scalar(ScalarValue::Union(value, fields, _)),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(field_name))),
            ) => {
                let result = match value {
                    Some((type_id, value)) => {
                        let field_type_id =
                            fields
                                .iter()
                                .find_map(|(i, field)| {
                                    if field.name() == field_name {
                                        Some(i)
                                    } else {
                                        None
                                    }
                                })
                                .ok_or_else(|| {
                                    exec_datafusion_err!("field_name {field_name} not found on union")
                                })?;

                        if field_type_id == *type_id {
                            *value.clone()
                        } else {
                            ScalarValue::Null
                        }
                    }
                    None => ScalarValue::Null,
                };

                Ok(ColumnarValue::Scalar(result))
            }
            _ => todo!(),
        }
    }
}

/*

union array verify

                                    // let mut last_offsets = [0; 128];

                                    // for (&type_id, &offset) in type_ids.values().iter().zip(offsets.values()) {

                                    //     if last_offsets[type_id as usize] >= offset {

                                    //     }

                                    //     last_offsets[type_id as usize] = offset;
                                    // }


old

                                    // let truthy = dense.to_data();

                                    // let mut mutable = MutableArrayData::new(vec![&truthy], true, mask.len());

                                    // // the SlicesIterator slices only the true values. So the gaps left by this iterator we need to
                                    // // fill with falsy values

                                    // SlicesIterator::new(&mask).for_each(|(start, end)| {

                                    //     if start > mutable.len() {
                                    //         mutable.extend_nulls(start - mutable.len());
                                    //     }

                                    //     let first_offset = offsets.values()[start];

                                    //     let mut slice = first_offset..first_offset + 1;

                                    //     for &offset in &offsets.values()[start + 1..end] {

                                    //         slice.end += 1;

                                    //         if slice.end != offset {

                                    //             mutable.extend(0, slice.start as usize, slice.end as usize);

                                    //             let gap = offset - slice.end;

                                    //             mutable.extend_nulls(gap as usize);

                                    //             slice = offset..offset + 1;
                                    //         }
                                    //     }

                                    //     mutable.extend(0, slice.start as usize, slice.end as usize);
                                    // });

                                    // // the remaining part is falsy
                                    // if mutable.len() < mask.len() {
                                    //     mutable.extend_nulls(mask.len() - mutable.len());
                                    // }

                                    // let data = mutable.freeze();

                                    // Ok(ColumnarValue::Array(make_array(data)))

*/
