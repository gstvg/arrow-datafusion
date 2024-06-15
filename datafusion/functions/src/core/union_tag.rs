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

use std::sync::Arc;

use arrow::array::{Array, DictionaryArray, PrimitiveArray, StringArray};

use arrow::datatypes::{DataType, Int8Type};

use datafusion_common::cast::as_union_array;
use datafusion_common::{exec_err, internal_datafusion_err, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};

#[derive(Debug)]
pub struct UnionTagFun {
    signature: Signature,
}

impl Default for UnionTagFun {
    fn default() -> Self {
        Self::new()
    }
}

impl UnionTagFun {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for UnionTagFun {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "union_tag"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(DataType::Utf8),
        ))
    }

    fn invoke(&self, columns: &[ColumnarValue]) -> Result<ColumnarValue> {
        let union_ = &columns[0];

        match union_ {
            ColumnarValue::Array(array) => {
                let union_array = as_union_array(&array)?;

                let fields = match union_array.data_type() {
                    DataType::Union(fields, _) => fields,
                    _ => unreachable!(),
                };

                let keys_buffer = union_array.type_ids();

                let keys = PrimitiveArray::try_new(keys_buffer.clone(), None)?;

                let values_len = fields
                    .iter()
                    .map(|(type_id, _)| type_id)
                    .max()
                    .map(|v| v + 1)
                    .unwrap_or_default();

                let mut values = vec![""; values_len as usize];

                for (type_id, field) in fields.iter() {
                    values[type_id as usize] = field.name()
                }

                let values = StringArray::from(values);

                let dictionary_array =
                    DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values))?;

                Ok(ColumnarValue::Array(Arc::new(dictionary_array)))
            }
            ColumnarValue::Scalar(ScalarValue::Union(value, fields, _)) => match value {
                Some((value_type_id, _)) => fields
                    .iter()
                    .find(|(type_id, _)| value_type_id == type_id)
                    .map(|(_, field)| {
                        ColumnarValue::Scalar(ScalarValue::Dictionary(
                            Box::new(DataType::Dictionary(
                                Box::new(DataType::Int8),
                                Box::new(DataType::Utf8),
                            )),
                            Box::new(ScalarValue::Utf8(Some(field.name().clone()))),
                        ))
                    })
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "union scalar with unknow type_id {value_type_id}"
                        )
                    }),
                None => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
            },
            v => exec_err!("union_tag only support unions, got {:?}", v.data_type()),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Int8DictionaryArray};
    use datafusion_expr::{ColumnarValue, ScalarUDFImpl};

    use super::UnionTagFun;

    use std::sync::Arc;

    use arrow::array::UnionBuilder;

    use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, UnionMode};

    use datafusion_common::{internal_err, Result, ScalarValue};

    #[test]
    fn union_array() -> Result<()> {
        let mut builder = UnionBuilder::new_sparse();

        builder.append::<Int32Type>("a", 1)?;
        builder.append::<Float64Type>("b", 3.0).unwrap();
        builder.append::<Int32Type>("a", 4)?;

        let union = builder.build()?;

        let result = UnionTagFun::new().invoke(&[ColumnarValue::Array(Arc::new(union))])?;
        let dict = Int8DictionaryArray::from_iter(["a", "b", "a"]);

        if let ColumnarValue::Array(result_array) = result {
            assert_eq!(result_array.to_data(), dict.into_data());

            Ok(())
        } else {
            internal_err!("expeceted ColumnarValue::Array got Scalar instead")
        }
    }

    #[test]
    fn union_scalar() -> Result<()> {
        let fields = [(0, Arc::new(Field::new("a", DataType::UInt32, false)))]
            .into_iter()
            .collect();

        let scalar = ScalarValue::Union(
            Some((0, Box::new(ScalarValue::UInt32(Some(0))))),
            fields,
            UnionMode::Dense,
        );

        let result = UnionTagFun::new().invoke(&[ColumnarValue::Scalar(scalar)])?;

        if let ColumnarValue::Scalar(scalar_result) = result {
            assert_eq!(ScalarValue::Utf8(Some("a".into())), scalar_result);

            Ok(())
        } else {
            internal_err!("expeceted ColumnarValue::Scalar got Array instead")
        }
    }
}
