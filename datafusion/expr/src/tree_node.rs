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

//! Tree node implementation for Logical Expressions

use std::collections::HashSet;

use crate::expr::{
    AggregateFunction, Alias, Between, BinaryExpr, Case, Cast, GroupingSet, InList,
    InSubquery, Lambda, Like, Placeholder, ScalarFunction, TryCast, Unnest,
    WindowFunction, AggregateFunctionParams,WindowFunctionParams,
};
use crate::{Expr, ExprFunctionExt};
use datafusion_common::tree_node::{
    Transformed, TreeNode, TreeNodeContainer, TreeNodeRecursion, TreeNodeRefContainer,
    TreeNodeVisitor,
};
use datafusion_common::{DFSchema, Result};

/// Implementation of the [`TreeNode`] trait
///
/// This allows logical expressions (`Expr`) to be traversed and transformed
/// Facilitates tasks such as optimization and rewriting during query
/// planning.
impl TreeNode for Expr {
    /// Applies a function `f` to each child expression of `self`.
    ///
    /// The function `f` determines whether to continue traversing the tree or to stop.
    /// This method collects all child expressions and applies `f` to each.
    fn apply_children<'n, F: FnMut(&'n Self) -> Result<TreeNodeRecursion>>(
        &'n self,
        f: F,
    ) -> Result<TreeNodeRecursion> {
        match self {
            Expr::Alias(Alias { expr, .. })
            | Expr::Unnest(Unnest { expr })
            | Expr::Not(expr)
            | Expr::IsNotNull(expr)
            | Expr::IsTrue(expr)
            | Expr::IsFalse(expr)
            | Expr::IsUnknown(expr)
            | Expr::IsNotTrue(expr)
            | Expr::IsNotFalse(expr)
            | Expr::IsNotUnknown(expr)
            | Expr::IsNull(expr)
            | Expr::Negative(expr)
            | Expr::Cast(Cast { expr, .. })
            | Expr::TryCast(TryCast { expr, .. })
            | Expr::InSubquery(InSubquery { expr, .. }) => expr.apply_elements(f),
            Expr::GroupingSet(GroupingSet::Rollup(exprs))
            | Expr::GroupingSet(GroupingSet::Cube(exprs)) => exprs.apply_elements(f),
            Expr::ScalarFunction(ScalarFunction { args, .. }) => {
                args.apply_elements(f)
            }
            Expr::GroupingSet(GroupingSet::GroupingSets(lists_of_exprs)) => {
                lists_of_exprs.apply_elements(f)
            }
            // TODO: remove the next line after `Expr::Wildcard` is removed
            #[expect(deprecated)]
            Expr::Column(_)
            // Treat OuterReferenceColumn as a leaf expression
            | Expr::OuterReferenceColumn(_, _)
            | Expr::ScalarVariable(_, _)
            | Expr::Literal(_)
            | Expr::Exists { .. }
            | Expr::ScalarSubquery(_)
            | Expr::Wildcard { .. }
            | Expr::Placeholder(_) => Ok(TreeNodeRecursion::Continue),
            Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
                (left, right).apply_ref_elements(f)
            }
            Expr::Like(Like { expr, pattern, .. })
            | Expr::SimilarTo(Like { expr, pattern, .. }) => {
                (expr, pattern).apply_ref_elements(f)
            }
            Expr::Between(Between {
                              expr, low, high, ..
                          }) => (expr, low, high).apply_ref_elements(f),
            Expr::Case(Case { expr, when_then_expr, else_expr }) =>
                (expr, when_then_expr, else_expr).apply_ref_elements(f),
            Expr::AggregateFunction(AggregateFunction { params: AggregateFunctionParams { args, filter, order_by, ..}, .. }) =>
                (args, filter, order_by).apply_ref_elements(f),
            Expr::WindowFunction(WindowFunction {
                params : WindowFunctionParams {
                    args,
                    partition_by,
                    order_by,
                    ..}, ..}) => {
                (args, partition_by, order_by).apply_ref_elements(f)
            }
            Expr::InList(InList { expr, list, .. }) => {
                (expr, list).apply_ref_elements(f)
            }
            Expr::Lambda (Lambda{ params: _, body}) => body.apply_elements(f)
        }
    }

    /// Maps each child of `self` using the provided closure `f`.
    ///
    /// The closure `f` takes ownership of an expression and returns a `Transformed` result,
    /// indicating whether the expression was transformed or left unchanged.
    fn map_children<F: FnMut(Self) -> Result<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        Ok(match self {
            // TODO: remove the next line after `Expr::Wildcard` is removed
            #[expect(deprecated)]
            Expr::Column(_)
            | Expr::Wildcard { .. }
            | Expr::Placeholder(Placeholder { .. })
            | Expr::OuterReferenceColumn(_, _)
            | Expr::Exists { .. }
            | Expr::ScalarSubquery(_)
            | Expr::ScalarVariable(_, _)
            | Expr::Literal(_) => Transformed::no(self),
            Expr::Unnest(Unnest { expr, .. }) => expr
                .map_elements(f)?
                .update_data(|expr| Expr::Unnest(Unnest { expr })),
            Expr::Alias(Alias {
                expr,
                relation,
                name,
                metadata,
            }) => f(*expr)?.update_data(|e| {
                e.alias_qualified_with_metadata(relation, name, metadata)
            }),
            Expr::InSubquery(InSubquery {
                expr,
                subquery,
                negated,
            }) => expr.map_elements(f)?.update_data(|be| {
                Expr::InSubquery(InSubquery::new(be, subquery, negated))
            }),
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => (left, right)
                .map_elements(f)?
                .update_data(|(new_left, new_right)| {
                    Expr::BinaryExpr(BinaryExpr::new(new_left, op, new_right))
                }),
            Expr::Like(Like {
                negated,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            }) => {
                (expr, pattern)
                    .map_elements(f)?
                    .update_data(|(new_expr, new_pattern)| {
                        Expr::Like(Like::new(
                            negated,
                            new_expr,
                            new_pattern,
                            escape_char,
                            case_insensitive,
                        ))
                    })
            }
            Expr::SimilarTo(Like {
                negated,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            }) => {
                (expr, pattern)
                    .map_elements(f)?
                    .update_data(|(new_expr, new_pattern)| {
                        Expr::SimilarTo(Like::new(
                            negated,
                            new_expr,
                            new_pattern,
                            escape_char,
                            case_insensitive,
                        ))
                    })
            }
            Expr::Not(expr) => expr.map_elements(f)?.update_data(Expr::Not),
            Expr::IsNotNull(expr) => expr.map_elements(f)?.update_data(Expr::IsNotNull),
            Expr::IsNull(expr) => expr.map_elements(f)?.update_data(Expr::IsNull),
            Expr::IsTrue(expr) => expr.map_elements(f)?.update_data(Expr::IsTrue),
            Expr::IsFalse(expr) => expr.map_elements(f)?.update_data(Expr::IsFalse),
            Expr::IsUnknown(expr) => expr.map_elements(f)?.update_data(Expr::IsUnknown),
            Expr::IsNotTrue(expr) => expr.map_elements(f)?.update_data(Expr::IsNotTrue),
            Expr::IsNotFalse(expr) => expr.map_elements(f)?.update_data(Expr::IsNotFalse),
            Expr::IsNotUnknown(expr) => {
                expr.map_elements(f)?.update_data(Expr::IsNotUnknown)
            }
            Expr::Negative(expr) => expr.map_elements(f)?.update_data(Expr::Negative),
            Expr::Between(Between {
                expr,
                negated,
                low,
                high,
            }) => (expr, low, high).map_elements(f)?.update_data(
                |(new_expr, new_low, new_high)| {
                    Expr::Between(Between::new(new_expr, negated, new_low, new_high))
                },
            ),
            Expr::Case(Case {
                expr,
                when_then_expr,
                else_expr,
            }) => (expr, when_then_expr, else_expr)
                .map_elements(f)?
                .update_data(|(new_expr, new_when_then_expr, new_else_expr)| {
                    Expr::Case(Case::new(new_expr, new_when_then_expr, new_else_expr))
                }),
            Expr::Cast(Cast { expr, data_type }) => expr
                .map_elements(f)?
                .update_data(|be| Expr::Cast(Cast::new(be, data_type))),
            Expr::TryCast(TryCast { expr, data_type }) => expr
                .map_elements(f)?
                .update_data(|be| Expr::TryCast(TryCast::new(be, data_type))),
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                args.map_elements(f)?.map_data(|new_args| {
                    Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
                        func, new_args,
                    )))
                })?
            }
            Expr::WindowFunction(WindowFunction {
                fun,
                params:
                    WindowFunctionParams {
                        args,
                        partition_by,
                        order_by,
                        window_frame,
                        null_treatment,
                    },
            }) => (args, partition_by, order_by).map_elements(f)?.update_data(
                |(new_args, new_partition_by, new_order_by)| {
                    Expr::WindowFunction(WindowFunction::new(fun, new_args))
                        .partition_by(new_partition_by)
                        .order_by(new_order_by)
                        .window_frame(window_frame)
                        .null_treatment(null_treatment)
                        .build()
                        .unwrap()
                },
            ),
            Expr::AggregateFunction(AggregateFunction {
                func,
                params:
                    AggregateFunctionParams {
                        args,
                        distinct,
                        filter,
                        order_by,
                        null_treatment,
                    },
            }) => (args, filter, order_by).map_elements(f)?.map_data(
                |(new_args, new_filter, new_order_by)| {
                    Ok(Expr::AggregateFunction(AggregateFunction::new_udf(
                        func,
                        new_args,
                        distinct,
                        new_filter,
                        new_order_by,
                        null_treatment,
                    )))
                },
            )?,
            Expr::GroupingSet(grouping_set) => match grouping_set {
                GroupingSet::Rollup(exprs) => exprs
                    .map_elements(f)?
                    .update_data(|ve| Expr::GroupingSet(GroupingSet::Rollup(ve))),
                GroupingSet::Cube(exprs) => exprs
                    .map_elements(f)?
                    .update_data(|ve| Expr::GroupingSet(GroupingSet::Cube(ve))),
                GroupingSet::GroupingSets(lists_of_exprs) => lists_of_exprs
                    .map_elements(f)?
                    .update_data(|new_lists_of_exprs| {
                        Expr::GroupingSet(GroupingSet::GroupingSets(new_lists_of_exprs))
                    }),
            },
            Expr::InList(InList {
                expr,
                list,
                negated,
            }) => (expr, list)
                .map_elements(f)?
                .update_data(|(new_expr, new_list)| {
                    Expr::InList(InList::new(new_expr, new_list, negated))
                }),
            Expr::Lambda(Lambda { params, body }) => body
                .map_elements(f)?
                .update_data(|body| Expr::Lambda(Lambda { params, body })),
        })
    }
}

impl Expr {
    /// Visits a expr similarly to [`Self::visit`], including lambdas that
    /// may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    pub fn visit_with_schema<
        'n,
        // V: for<'a> TreeNodeVisitor<'a, Node = (&'n Self, &'a Schema)>,
        V: for<'a> TreeNodeVisitor<'a, Node = (&'a Self, &'a DFSchema)>,
    >(
        &'n self,
        schema: &DFSchema,
        visitor: &mut V,
    ) -> Result<TreeNodeRecursion> {
        match self {
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                let mut lambdas_schemas =
                    func.lambdas_schemas_from_args(args, schema)?.into_iter();

                visitor
                    .f_down(&(self, schema))?
                    .visit_children(|| {
                        self.apply_children(|c| {
                            c.visit_with_schema(
                                &lambdas_schemas.next().unwrap(),
                                visitor,
                            )
                        })
                    })?
                    .visit_parent(|| visitor.f_up(&(self, schema)))
            }
            _ => visitor
                .f_down(&(self, schema))?
                .visit_children(|| {
                    self.apply_children(|c| c.visit_with_schema(schema, visitor))
                })?
                .visit_parent(|| visitor.f_up(&(self, schema))),
        }
    }

    /// Similarly to [`Self::rewrite`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    pub fn rewrite_with_schema<R: ExprWithLambdasRewriter>(
        self,
        schema: &DFSchema,
        rewriter: &mut R,
    ) -> Result<Transformed<Self>> {
        rewriter
            .f_down(self, schema)?
            .transform_children(|n| match n {
                Expr::ScalarFunction(ScalarFunction { ref func, ref args }) => {
                    let mut lambdas_schemas =
                        func.lambdas_schemas_from_args(args, schema)?.into_iter();

                    n.map_children(|n| {
                        n.rewrite_with_schema(&lambdas_schemas.next().unwrap(), rewriter)
                    })
                }
                _ => n.map_children(|n| n.rewrite_with_schema(schema, rewriter)),
            })?
            .transform_parent(|n| rewriter.f_up(n, schema))
    }

    /// Similarly to [`Self::apply`], calls `f` on this expr and all its inputs,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn apply_with_schema<
        'n,
        F: FnMut(&'n Self, &DFSchema) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn apply_with_schema_impl<
            'n,
            F: FnMut(&'n Expr, &DFSchema) -> Result<TreeNodeRecursion>,
        >(
            node: &'n Expr,
            schema: &DFSchema,
            f: &mut F,
        ) -> Result<TreeNodeRecursion> {
            f(node, schema)?.visit_children(|| {
                node.apply_children_with_schema(schema, |c, schema| {
                    apply_with_schema_impl(c, schema, f)
                })
            })
        }

        apply_with_schema_impl(self, schema, &mut f)
    }

    /// Similarly to [`Self::transform`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn transform_with_schema<
        F: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>,
    >(
        self,
        schema: &DFSchema,
        f: F,
    ) -> Result<Transformed<Self>> {
        self.transform_up_with_schema(schema, f)
    }

    /// Similarly to [`Self::transform_down`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn transform_down_with_schema<
        F: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>,
    >(
        self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn transform_down_with_schema_impl<
            F: FnMut(Expr, &DFSchema) -> Result<Transformed<Expr>>,
        >(
            node: Expr,
            schema: &DFSchema,
            f: &mut F,
        ) -> Result<Transformed<Expr>> {
            f(node, schema)?.transform_children(|n| {
                n.map_children_with_schema(schema, |n, schema| {
                    transform_down_with_schema_impl(n, schema, f)
                })
            })
        }

        transform_down_with_schema_impl(self, schema, &mut f)
    }

    /// Similarly to [`Self::transform_up`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn transform_up_with_schema<
        F: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>,
    >(
        self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn transform_up_with_schema_impl<
            F: FnMut(Expr, &DFSchema) -> Result<Transformed<Expr>>,
        >(
            node: Expr,
            schema: &DFSchema,
            f: &mut F,
        ) -> Result<Transformed<Expr>> {
            node.map_children_with_schema(schema, |n, schema| {
                transform_up_with_schema_impl(n, schema, f)
            })?
            .transform_parent(|n| f(n, schema))
        }

        transform_up_with_schema_impl(self, schema, &mut f)
    }

    /// Similarly to [`Self::transform_down`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn transform_down_up_with_schema<
        FD: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>,
        FU: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>,
    >(
        self,
        schema: &DFSchema,
        mut f_down: FD,
        mut f_up: FU,
    ) -> Result<Transformed<Self>> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn transform_down_up_with_schema_impl<
            FD: FnMut(Expr, &DFSchema) -> Result<Transformed<Expr>>,
            FU: FnMut(Expr, &DFSchema) -> Result<Transformed<Expr>>,
        >(
            node: Expr,
            schema: &DFSchema,
            f_down: &mut FD,
            f_up: &mut FU,
        ) -> Result<Transformed<Expr>> {
            f_down(node, schema)?
                .transform_children(|n| {
                    n.map_children_with_schema(schema, |n, schema| {
                        transform_down_up_with_schema_impl(n, schema, f_down, f_up)
                    })
                })?
                .transform_parent(|n| f_up(n, schema))
        }

        transform_down_up_with_schema_impl(self, schema, &mut f_down, &mut f_up)
    }

    /// Similarly to [`Self::apply`], calls `f` on this expr and its inputs
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn apply_children_with_schema<
        'n,
        F: FnMut(&'n Self, &DFSchema) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        match self {
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                let mut lambdas_schemas =
                    func.arguments_schema_from_logical_args(args, schema)?.into_iter();

                self.apply_children(|expr| f(expr, &lambdas_schemas.next().unwrap()))
            }
            _ => self.apply_children(|e| f(e, schema)),
        }
    }

    /// Similarly to [`Self::map_children`], apply `f` to rewrite the node's children (but not the node itself),
    /// including the scoped schema of the children, which differ in case it's a lambda, like in `list_map([1, 2], v -> v*2)`.
    ///
    /// Returns the current node.
    pub fn map_children_with_schema<F: FnMut(Self, &DFSchema) -> Result<Transformed<Self>>>(
        self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        match self {
            Expr::ScalarFunction(ScalarFunction { ref func, ref args }) => {
                let mut lambdas_schemas =
                    func.arguments_schema_from_logical_args(args, schema)?.into_iter();

                self.map_children(|expr| f(expr, &lambdas_schemas.next().unwrap()))
            }
            _ => self.map_children(|expr| f(expr, schema)),
        }
    }
    
    /// Similarly to [`Self::map_children`], rewrites all lambdas that may
    /// appear in expressions such as `list_map([1, 2], v -> v*2)`.
    ///
    /// Returns the current node.
    pub fn map_children_with_lambdas_params<F: FnMut(Self, &HashSet<String>) -> Result<Transformed<Self>>>(
        self,
        args: &HashSet<String>,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        match &self {
            Expr::Lambda(Lambda {
                params,
                body: _,
            }) => {
                let mut args = args.clone();

                args.extend(params.iter().cloned());

                self.map_children(|expr| f(expr, &args))
            }
            _ => self.map_children(|expr| f(expr, args)),
        }
    }

    pub fn exists_with_schema<F: FnMut(&Self, &DFSchema) -> Result<bool>>(
        &self,
        schema: &DFSchema,
        mut f: F,
    ) -> Result<bool> {
        let mut found = false;

        self.apply_children_with_schema(schema, |n, schema| {
            if f(n, schema)? {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })?;

        Ok(found)
    }

    /// Similarly to [`Self::transform_up`], rewrites this expr and its inputs using `f`,
    /// including lambdas that may appear in expressions such as `list_map([1, 2], v -> v*2)`.
    pub fn transform_up_with_lambdas_params<
        F: FnMut(Self, &HashSet<String>) -> Result<Transformed<Self>>,
    >(
        self,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn transform_up_with_lambdas_params_impl<
            F: FnMut(Expr, &HashSet<String>) -> Result<Transformed<Expr>>,
        >(
            node: Expr,
            args: &HashSet<String>,
            f: &mut F,
        ) -> Result<Transformed<Expr>> {
            match &node {
                Expr::Lambda(Lambda {
                    params,
                    body: _,
                }) => {
                    let mut args = args.clone();

                    args.extend(params.iter().cloned());

                    node.map_children(|n| {
                        transform_up_with_lambdas_params_impl(n, &args, f)
                    })?
                    .transform_parent(|n| f(n, &args))
                }
                _ => node
                    .map_children(|n| transform_up_with_lambdas_params_impl(n, args, f))?
                    .transform_parent(|n| f(n, args)),
            }
        }

        transform_up_with_lambdas_params_impl(self, &HashSet::new(), &mut f)
    }

    pub fn apply_with_lambdas_params<
        'n,
        F: FnMut(&'n Self, &HashSet<&str>) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn apply_with_lambdas_params_impl<
            'n,
            F: FnMut(&'n Expr, &HashSet<&str>) -> Result<TreeNodeRecursion>,
        >(
            node: &'n Expr,
            args: &HashSet<&str>,
            f: &mut F,
        ) -> Result<TreeNodeRecursion> {
            match node {
                Expr::Lambda(Lambda {
                    params,
                    body: _,
                }) => {
                    let mut args = args.clone();

                    args.extend(params.iter().map(|v| v.as_str()));

                    f(node, &args)?.visit_children(|| {
                        node.apply_children(|c| apply_with_lambdas_params_impl(c, &args, f))
                    })
                }
                _ => f(node, args)?.visit_children(|| {
                    node.apply_children(|c| apply_with_lambdas_params_impl(c, args, f))
                }),
            }
        }

        apply_with_lambdas_params_impl(self, &HashSet::new(), &mut f)
    }
}

pub trait ExprWithLambdasRewriter: Sized {
    /// Invoked while traversing down the tree before any children are rewritten.
    /// Default implementation returns the node as is and continues recursion.
    fn f_down(&mut self, node: Expr, _schema: &DFSchema) -> Result<Transformed<Expr>> {
        Ok(Transformed::no(node))
    }

    /// Invoked while traversing up the tree after all children have been rewritten.
    /// Default implementation returns the node as is and continues recursion.
    fn f_up(&mut self, node: Expr, _schema: &DFSchema) -> Result<Transformed<Expr>> {
        Ok(Transformed::no(node))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use arrow::datatypes::{DataType, Field, Fields, Schema, UnionFields};
    use datafusion_common::{
        tree_node::{Transformed, TreeNodeRecursion, TreeNodeVisitor},
        DFSchema, Result,
    };

    use crate::{
        col, expr::Lambda, Expr, LambdaParameter, ScalarFunctionArgMetadata, ScalarUDF,
        ScalarUDFImpl,
    };

    use super::ExprWithLambdasRewriter;

    fn schema() -> DFSchema {
        DFSchema::try_from(Schema::new(vec![Field::new(
            "v",
            DataType::new_list(DataType::new_list(DataType::Int32, false), false),
            false,
        )]))
        .unwrap()
    }

    // list_map(v, |v| -> list_map(v, |v| -> -v))
    fn list_map() -> Expr {
        ScalarUDF::new_from_impl(ListMapFunc).call(vec![
            col("v"),
            Expr::Lambda(Lambda {
                params: vec!["v".into()],
                body: Box::new(ScalarUDF::new_from_impl(ListMapFunc).call(vec![
                    col("v"),
                    Expr::Lambda(Lambda {
                        params: vec!["v".into()],
                        body: Box::new(-col("v")),
                    }),
                ])),
            }),
        ])
    }

    fn short_data_type(d: &DataType) -> String {
        match d {
            DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Timestamp(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Binary
            | DataType::FixedSizeBinary(_)
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View => d.to_string(),
            DataType::List(field) => {
                format!("List<{}>", short_data_type(field.data_type()))
            }
            DataType::ListView(field) => {
                format!("ListView<{}>", short_data_type(field.data_type()))
            }
            DataType::FixedSizeList(field, size) => {
                format!(
                    "FixedSizeList<{}, {size}>",
                    short_data_type(field.data_type())
                )
            }
            DataType::LargeList(field) => {
                format!("LargeList<{}>", short_data_type(field.data_type()))
            }
            DataType::LargeListView(field) => {
                format!("LargeListView<{}>", short_data_type(field.data_type()))
            }
            DataType::Struct(fields) => {
                format!("Struct<{}>", short_fields(fields))
            }
            DataType::Union(union_fields, union_mode) => {
                format!(
                    "Union<{}, {union_mode:?}>",
                    short_union_fields(union_fields)
                )
            }
            DataType::Dictionary(data_type, data_type1) => format!(
                "Dictionary<{}, {}>",
                short_data_type(data_type),
                short_data_type(data_type1)
            ),
            DataType::Map(field, sorted) => {
                format!("Map<{}, {sorted}>", field.data_type())
            }
            DataType::RunEndEncoded(field, field1) => format!(
                "RunEndEncoded<{}, {}>",
                short_data_type(field.data_type()),
                short_data_type(field1.data_type())
            ),
        }
    }

    fn short_fields(f: &Fields) -> String {
        f.iter()
            .map(|f| format!("{}: {}", f.name(), short_data_type(f.data_type())))
            .collect()
    }

    fn short_union_fields(f: &UnionFields) -> String {
        f.iter()
            .map(|(type_id, f)| {
                format!(
                    "{}({type_id}): {}",
                    f.name(),
                    short_data_type(f.data_type())
                )
            })
            .collect()
    }

    #[derive(Debug)]
    struct ListMapFunc;

    impl ScalarUDFImpl for ListMapFunc {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn name(&self) -> &str {
            "list_map"
        }

        fn signature(&self) -> &datafusion_expr_common::signature::Signature {
            unimplemented!()
        }

        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }

        fn lambdas_parameters(
            &self,
            args: &[ScalarFunctionArgMetadata],
        ) -> Result<Vec<Option<Vec<LambdaParameter>>>> {
            let ScalarFunctionArgMetadata::Value(DataType::List(field)) = &args[0] else {
                unimplemented!()
            };

            Ok(vec![
                None,
                Some(vec![LambdaParameter::new(
                    field.data_type().clone(),
                    field.is_nullable(),
                )]),
            ])
        }
        
        fn invoke_with_args(&self, _args: crate::ScalarFunctionArgs) -> Result<datafusion_expr_common::columnar_value::ColumnarValue> {
            todo!()
        }
    }

    #[test]
    fn test_apply_with_schema2() {
        let list_map = list_map();
        let mut strings = vec![];
        let mut count = 0;

        list_map
            .apply_with_schema(&schema(), |expr, _schema| {
                strings.push(format!("apply {expr}"));

                count += 1;

                if count > 10 {
                    Ok(TreeNodeRecursion::Stop)
                } else {
                    Ok(TreeNodeRecursion::Continue)
                }
            })
            .unwrap();

        println!("{}", strings.join("\n"))
    }

    #[test]
    fn test_apply_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let mut strings = vec![];

        list_map
            .apply_with_schema(&schema, |expr, schema| {
                strings.push(format!("apply {} {}", expr, short_fields(schema.fields())));

                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert_eq!(
            strings,
            [
                "apply list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>",
                "apply v v=List<List<Int32>>",
                "apply (v) -> list_map(v, (v) -> (- v)) v=List<Int32>"
            ]
        )
    }

    #[test]
    fn test_transform_up_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let mut strings = vec![];

        list_map
            .transform_up_with_schema(&schema, |expr, schema| {
                strings.push(format!(
                    "transform_up {} {}",
                    expr,
                    short_fields(schema.fields())
                ));

                Ok(Transformed::no(expr))
            })
            .unwrap();

        assert_eq!(
            strings,
            [
                "transform_up v v=List<List<Int32>>", 
                "transform_up v v=List<Int32>",
                "transform_up v v=Int32", 
                "transform_up (- v) v=Int32", 
                "transform_up (v) -> (- v) v=Int32", 
                "transform_up list_map(v, (v) -> (- v)) v=List<Int32>", 
                "transform_up (v) -> list_map(v, (v) -> (- v)) v=List<Int32>",
                "transform_up list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>"
            ]
        )
    }

    #[test]
    fn test_transform_down_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let mut strings = vec![];

        list_map
            .transform_down_with_schema(&schema, |expr, schema| {
                strings.push(format!(
                    "transform_down {} {}",
                    expr,
                    short_fields(schema.fields())
                ));

                Ok(Transformed::no(expr))
            })
            .unwrap();

        assert_eq!(
            strings,
            ["transform_down list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>", "transform_down v v=List<List<Int32>>", "transform_down (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "transform_down list_map(v, (v) -> (- v)) v=List<Int32>", "transform_down v v=List<Int32>", "transform_down (v) -> (- v) v=Int32", "transform_down (- v) v=Int32", "transform_down v v=Int32"]
        )
    }

    #[test]
    fn test_transform_down_up_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let strings = RefCell::new(vec![]);

        list_map
            .transform_down_up_with_schema(
                &schema,
                |expr, schema| {
                    strings.borrow_mut().push(format!(
                        "transform_down {} {}",
                        expr,
                        short_fields(schema.fields())
                    ));

                    Ok(Transformed::no(expr))
                },
                |expr, schema| {
                    strings.borrow_mut().push(format!(
                        "transform_up {} {}",
                        expr,
                        short_fields(schema.fields())
                    ));

                    Ok(Transformed::no(expr))
                },
            )
            .unwrap();

        assert_eq!(
            strings.into_inner(),
            ["transform_down list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>", "transform_down v v=List<List<Int32>>", "transform_up v v=List<List<Int32>>", "transform_down (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "transform_down list_map(v, (v) -> (- v)) v=List<Int32>", "transform_down v v=List<Int32>", "transform_up v v=List<Int32>", "transform_down (v) -> (- v) v=Int32", "transform_down (- v) v=Int32", "transform_down v v=Int32", "transform_up v v=Int32", "transform_up (- v) v=Int32", "transform_up (v) -> (- v) v=Int32", "transform_up list_map(v, (v) -> (- v)) v=List<Int32>", "transform_up (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "transform_up list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>"]
        )
    }

    #[test]
    fn test_exists_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let mut strings = vec![];

        list_map
            .exists_with_schema(&schema, |expr, schema| {
                strings.push(format!(
                    "exists {} {}",
                    expr,
                    short_fields(schema.fields())
                ));

                Ok(false)
            })
            .unwrap();

        assert_eq!(
            strings,
            ["exists list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>", "exists v v=List<List<Int32>>", "exists (v) -> list_map(v, (v) -> (- v)) v=List<Int32>"]
        )
    }

    #[test]
    fn test_visit_with_schema() {
        let schema = schema();
        let list_map = list_map();
        let mut visitor = OkVisitor::default();

        list_map.visit_with_schema(&schema, &mut visitor).unwrap();

        assert_eq!(
            visitor.strings,
            ["f_down list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>", "f_down v v=List<List<Int32>>", "f_up v v=List<List<Int32>>", "f_down (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "f_down list_map(v, (v) -> (- v)) v=List<Int32>", "f_down v v=List<Int32>", "f_up v v=List<Int32>", "f_down (v) -> (- v) v=Int32", "f_down (- v) v=Int32", "f_down v v=Int32", "f_up v v=Int32", "f_up (- v) v=Int32", "f_up (v) -> (- v) v=Int32", "f_up list_map(v, (v) -> (- v)) v=List<Int32>", "f_up (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "f_up list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>"]

        )
    }

    #[test]
    fn test_rewrite_with_schema() {
        let schema = schema();
        let list_map = list_map();

        let mut rewriter = OkVisitor::default();

        list_map
            .rewrite_with_schema(&schema, &mut rewriter)
            .unwrap();

        assert_eq!(
            rewriter.strings,
            ["f_down list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>", "f_down v v=List<List<Int32>>", "f_up v v=List<List<Int32>>", "f_down (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "f_down list_map(v, (v) -> (- v)) v=List<Int32>", "f_down v v=List<Int32>", "f_up v v=List<Int32>", "f_down (v) -> (- v) v=Int32", "f_down (- v) v=Int32", "f_down v v=Int32", "f_up v v=Int32", "f_up (- v) v=Int32", "f_up (v) -> (- v) v=Int32", "f_up list_map(v, (v) -> (- v)) v=List<Int32>", "f_up (v) -> list_map(v, (v) -> (- v)) v=List<Int32>", "f_up list_map(v, (v) -> list_map(v, (v) -> (- v))) v=List<List<Int32>>"]

        )
    }

    #[derive(Default)]
    struct OkVisitor {
        strings: Vec<String>,
    }

    impl<'n> TreeNodeVisitor<'n> for OkVisitor {
        type Node = (&'n Expr, &'n DFSchema);

        fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
            self.strings.push(format!(
                "f_down {} {}",
                node.0,
                short_fields(node.1.fields())
            ));

            Ok(TreeNodeRecursion::Continue)
        }

        fn f_up(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
            self.strings.push(format!(
                "f_up {} {}",
                node.0,
                short_fields(node.1.fields())
            ));

            Ok(TreeNodeRecursion::Continue)
        }
    }

    impl ExprWithLambdasRewriter for OkVisitor {
        fn f_down(&mut self, node: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
            self.strings.push(format!(
                "f_down {} {}",
                node,
                short_fields(schema.fields())
            ));

            Ok(Transformed::no(node))
        }

        fn f_up(&mut self, node: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
            self.strings
                .push(format!("f_up {} {}", node, short_fields(schema.fields())));

            Ok(Transformed::no(node))
        }
    }
}
