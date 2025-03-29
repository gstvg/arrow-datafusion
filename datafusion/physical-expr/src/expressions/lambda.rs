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

//! Physical column reference: [`Column`]

use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use crate::physical_expr::PhysicalExpr;
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::{internal_err, Result};
use datafusion_expr::ColumnarValue;

/// Represents the column at a given index in a RecordBatch
///
/// This is a physical expression that represents a column at a given index in an
/// arrow [`Schema`] / [`RecordBatch`].
///
/// Unlike the [logical `Expr::Column`], this expression is always resolved by schema index,
/// even though it does have a name. This is because the physical plan is always
/// resolved to a specific schema and there is no concept of "relation"
///
/// # Example:
///  If the schema is `a`, `b`, `c` the `Column` for `b` would be represented by
///  index 1, since `b` is the second column in the schema.
///
/// ```
/// # use datafusion_physical_expr::expressions::Column;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// // Schema with columns a, b, c
/// let schema = Schema::new(vec![
///    Field::new("a", DataType::Int32, false),
///    Field::new("b", DataType::Int32, false),
///    Field::new("c", DataType::Int32, false),
/// ]);
///
/// // reference to column b is index 1
/// let column_b = Column::new_with_schema("b", &schema).unwrap();
/// assert_eq!(column_b.index(), 1);
///
/// // reference to column c is index 2
/// let column_c = Column::new_with_schema("c", &schema).unwrap();
/// assert_eq!(column_c.index(), 2);
/// ```
/// [logical `Expr::Column`]: https://docs.rs/datafusion/latest/datafusion/logical_expr/enum.Expr.html#variant.Column
#[derive(Debug, Eq, Clone)]
pub struct Lambda {
    inner: Arc<dyn PhysicalExpr>,
    args: Vec<String>,
}

impl PartialEq for Lambda {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl Hash for Lambda {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl Lambda {
    /// Create a new lambda expression with the given body and arguments names
    pub fn new(inner: Arc<dyn PhysicalExpr>, args: Vec<String>) -> Self {
        Self { inner, args }
    }

    pub fn inner(&self) -> &Arc<dyn PhysicalExpr> {
        &self.inner
    }

    pub fn args(&self) -> &[String] {
        &self.args
    }
}

impl std::fmt::Display for Lambda {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "lambda(({}) -> {})", self.args.join(", "), self.inner)
    }
}

impl PhysicalExpr for Lambda {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Null)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        internal_err!("Lambda::evaluate() should not be called")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.inner]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if let Some(inner) = children.pop() {
            Ok(Arc::new(Self {
                inner,
                args: self.args.clone(),
            }))
        } else {
            Ok(self)
        }
    }
}
