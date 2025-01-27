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

//! Physical lambda reference: [`Lambda`]

use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use crate::physical_expr::PhysicalExpr;
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::{internal_err, Result};
use datafusion_expr::{ColumnarValue, Expr};

/// Encapsulates the lambda expression
#[derive(Debug, Eq, Clone)]
pub struct Lambda {
    pub(crate) inner: Arc<dyn PhysicalExpr>,
    pub(crate) args: Vec<String>,
    pub(crate) expr: Expr
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
    /// Create a new lambda expression
    pub fn new(inner: Arc<dyn PhysicalExpr>, args: Vec<String>, expr: Expr) -> Self {
        Self {
            inner, args, expr
        }
    }
}

impl std::fmt::Display for Lambda {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "lambda({})", self.inner)
    }
}

impl PhysicalExpr for Lambda {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the data type of this expression, given the schema of the input
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Null)
    }

    /// Decide whether this expression is nullable, given the schema of the input
    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    /// Evaluate the expression
    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        internal_err!("Lambda::evaluate() should not be called")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }
}
