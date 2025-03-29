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

//! Defines physical expressions that can evaluated at runtime during query execution

#[macro_use]
mod binary;
mod case;
mod cast;
mod column;
mod in_list;
mod is_not_null;
mod is_null;
mod lambda;
mod like;
mod literal;
mod negative;
mod no_op;
mod not;
mod try_cast;
mod unknown_column;

use datafusion_common::Result;
use std::sync::Arc;

/// Module with some convenient methods used in expression building
pub use crate::aggregate::stats::StatsType;
pub use crate::PhysicalSortExpr;
use crate::{scalar_function::lambdas_schemas_from_args, ScalarFunctionExpr};

use arrow_schema::Schema;
pub use binary::{binary, similar_to, BinaryExpr};
pub use case::{case, CaseExpr};
pub use cast::{cast, CastExpr};
pub use column::{col, with_new_schema, Column};
pub use datafusion_expr::utils::format_state_name;
use datafusion_physical_expr_common::{
    physical_expr::PhysicalExpr, tree_node::ExprContext,
};
pub use in_list::{in_list, InListExpr};
pub use is_not_null::{is_not_null, IsNotNullExpr};
pub use is_null::{is_null, IsNullExpr};
pub use lambda::Lambda;
pub use like::{like, LikeExpr};
pub use literal::{lit, Literal};
pub use negative::{negative, NegativeExpr};
pub use no_op::NoOp;
pub use not::{not, NotExpr};
pub use try_cast::{try_cast, TryCastExpr};
pub use unknown_column::UnKnownColumn;

pub type ExprSchemaNode = ExprContext<Arc<Schema>>;

pub fn new_expr_with_schema(
    expr: Arc<dyn PhysicalExpr>,
    schema: Arc<Schema>,
) -> Result<ExprSchemaNode> {
    if let Some(scalar_function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        let lambdas_schemas = lambdas_schemas_from_args(
            scalar_function.fun(),
            scalar_function.args(),
            &schema,
        )?;

        let children = expr
            .children()
            .into_iter()
            .cloned()
            .zip(lambdas_schemas)
            .map(|(v, lambda_schema)| {
                if v.as_any().downcast_ref::<Lambda>().is_some() {
                    new_expr_with_schema(v, Arc::new(lambda_schema.unwrap().into()))
                } else {
                    new_expr_with_schema(v, Arc::clone(&schema))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ExprContext {
            expr,
            data: schema,
            children,
        })
    } else {
        let children = expr
            .children()
            .into_iter()
            .cloned()
            .map(|v| new_expr_with_schema(v, Arc::clone(&schema)))
            .collect::<Result<Vec<_>>>()?;

        Ok(ExprContext {
            expr,
            data: schema,
            children,
        })
    }
}
