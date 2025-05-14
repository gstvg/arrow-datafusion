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

//! Declaration of built-in (scalar) functions.
//! This module contains built-in functions' enumeration and metadata.
//!
//! Generally, a function has:
//! * a signature
//! * a return type, that is a function of the incoming argument's types
//! * the computation, that must accept each valid signature
//!
//! * Signature: see `Signature`
//! * Return type: a function `(arg_types) -> return_type`. E.g. for sqrt, ([f32]) -> f32, ([f64]) -> f64.
//!
//! This module also has a set of coercion rules to improve user experience: if an argument i32 is passed
//! to a function that supports f64, it is coerced to f64.

use std::any::Any;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use crate::expressions::{Column, LambdaExpr, Literal};
use crate::PhysicalExpr;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, Schema};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{internal_err, DFSchema, HashSet, Result, ScalarValue};
use arrow::datatypes::Field;
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::sort_properties::ExprProperties;
use datafusion_expr::type_coercion::functions::data_types_with_scalar_udf;
use datafusion_expr::{
    expr_vec_fmt, ColumnarValue, ScalarFunctionArgMetadata,
    ScalarFunctionArgs, ScalarFunctionLambdaArg, ScalarUDF,
    ReturnFieldArgs,
};

/// Physical expression of a scalar function
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct ScalarFunctionExpr {
    fun: Arc<ScalarUDF>,
    name: String,
    args: Vec<Arc<dyn PhysicalExpr>>,
    return_field: Field,
}

impl Debug for ScalarFunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ScalarFunctionExpr")
            .field("fun", &"<FUNC>")
            .field("name", &self.name)
            .field("args", &self.args)
            .field("return_field", &self.return_field)
            .finish()
    }
}

impl ScalarFunctionExpr {
    /// Create a new Scalar function
    pub fn new(
        name: &str,
        fun: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        return_field: Field,
    ) -> Self {
        Self {
            fun,
            name: name.to_owned(),
            args,
            return_field,
        }
    }

    /// Create a new Scalar function
    pub fn try_new(
        fun: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        schema: &Schema,
    ) -> Result<Self> {
        let lambdas_schemas = lambdas_schemas_from_args(&fun, &args, schema)?;

        let arg_fields = std::iter::zip(&args, lambdas_schemas)
            .map(|(e, schema)| {
                if let Some(lambda) = e.as_any().downcast_ref::<LambdaExpr>() {
                    lambda.body().return_field(&schema)
                } else {
                    e.return_field(&schema)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // verify that input data types is consistent with function's `TypeSignature`
        let arg_types = arg_fields
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        
        data_types_with_scalar_udf(&arg_types, &fun)?;

        let arguments = args
            .iter()
            .map(|e| {
                e.as_any()
                    .downcast_ref::<Literal>()
                    .map(|literal| literal.value())
            })
            .collect::<Vec<_>>();

        let lambdas = args
            .iter()
            .map(|e| e.as_any().is::<LambdaExpr>())
            .collect::<Vec<_>>();

        let ret_args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &arguments,
            lambdas: &lambdas,
        };

        let return_field = fun.return_field_from_args(ret_args)?;
        let name = fun.name().to_string();
        
        Ok(Self {
            fun,
            name,
            args,
            return_field,
        })
    }

    /// Get the scalar function implementation
    pub fn fun(&self) -> &ScalarUDF {
        &self.fun
    }

    /// The name for this expression
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Input arguments
    pub fn args(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.args
    }

    /// Data type produced by this expression
    pub fn return_type(&self) -> &DataType {
        self.return_field.data_type()
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.return_field = self.return_field.with_nullable(nullable);
        self
    }

    pub fn nullable(&self) -> bool {
        self.return_field.is_nullable()
    }
}

impl fmt::Display for ScalarFunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.name, expr_vec_fmt!(self.args))
    }
}

impl PhysicalExpr for ScalarFunctionExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.return_field.data_type().clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(self.return_field.is_nullable())
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let args = self
            .args
            .iter()
            .map(|e| match e.as_any().downcast_ref::<LambdaExpr>() {
                Some(_) => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
                None => Ok(e.evaluate(batch)?),
            })
            .collect::<Result<Vec<_>>>()?;

        let arg_fields_owned = self
            .args
            .iter()
            .map(|e| e.return_field(batch.schema_ref()))
            .collect::<Result<Vec<_>>>()?;
        let arg_fields = arg_fields_owned.iter().collect::<Vec<_>>();

        let input_empty = args.is_empty();
        let input_all_scalar = args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));

        let params = self.fun().inner().lambdas_parameters(&std::iter::zip(&self.args, &args)
        .map(
            |(expr, value)| match expr.as_any().downcast_ref::<LambdaExpr>() {
                Some(lambda) => ScalarFunctionArgMetadata::Lambda(lambda.params()),
                None => ScalarFunctionArgMetadata::Value(value.data_type()),
            },
        )
        .collect::<Vec<_>>())?;

        let lambdas = std::iter::zip(&self.args, params)
            .map(|(arg, lambda_params)| {
                arg.as_any()
                    .downcast_ref::<LambdaExpr>()
                    .map(|lambda| {
                        let mut indices = HashSet::new();

                        arg.apply_with_lambdas_params(|expr, lambdas_params| {
                            if let Some(column) = expr.as_any().downcast_ref::<Column>() {
                                if !lambdas_params.contains(column.name()) {
                                    indices.insert(
                                        batch.schema_ref().index_of(column.name())?,
                                    );
                                }
                            }

                            Ok(TreeNodeRecursion::Continue)
                        })?;

                        let mut indices = indices.into_iter().collect::<Vec<_>>();

                        indices.sort_unstable();

                        let fields =
                            std::iter::zip(lambda.params(), lambda_params.unwrap())
                                .map(|(name, param)| Arc::new(param.into_field(name)))
                                .collect();

                        let captures = if !indices.is_empty() {
                            Some(batch.project(&indices)?)
                        } else {
                            None
                        };

                        Ok(ScalarFunctionLambdaArg {
                            params: lambda.params(),
                            fields,
                            body: lambda.body().as_ref(),
                            captures,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;

        // evaluate the function
        let output = self.fun.invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: batch.num_rows(),
            return_field: &self.return_field,
            lambdas,
        })?;

        if let ColumnarValue::Array(array) = &output {
            if array.len() != batch.num_rows() {
                // If the arguments are a non-empty slice of scalar values, we can assume that
                // returning a one-element array is equivalent to returning a scalar.
                let preserve_scalar =
                    array.len() == 1 && !input_empty && input_all_scalar;
                return if preserve_scalar {
                    ScalarValue::try_from_array(array, 0).map(ColumnarValue::Scalar)
                } else {
                    internal_err!("UDF {} returned a different number of rows than expected. Expected: {}, Got: {}",
                            self.name, batch.num_rows(), array.len())
                };
            }
        }
        Ok(output)
    }

    fn return_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(self.return_field.clone())
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.args.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(ScalarFunctionExpr::new(
            &self.name,
            Arc::clone(&self.fun),
            children,
            self.return_field.clone(),
        )))
    }

    fn evaluate_bounds(&self, children: &[&Interval]) -> Result<Interval> {
        self.fun.evaluate_bounds(children)
    }

    fn propagate_constraints(
        &self,
        interval: &Interval,
        children: &[&Interval],
    ) -> Result<Option<Vec<Interval>>> {
        self.fun.propagate_constraints(interval, children)
    }

    fn get_properties(&self, children: &[ExprProperties]) -> Result<ExprProperties> {
        let sort_properties = self.fun.output_ordering(children)?;
        let preserves_lex_ordering = self.fun.preserves_lex_ordering(children)?;
        let children_range = children
            .iter()
            .map(|props| &props.range)
            .collect::<Vec<_>>();
        let range = self.fun().evaluate_bounds(&children_range)?;

        Ok(ExprProperties {
            sort_properties,
            range,
            preserves_lex_ordering,
        })
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, expr) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            expr.fmt_sql(f)?;
        }
        write!(f, ")")
    }
}

pub fn lambdas_schemas_from_args(
    fun: &ScalarUDF,
    args: &[Arc<dyn PhysicalExpr>],
    schema: &Schema,
) -> Result<Vec<Schema>> {
    let args_metadata = args
        .iter()
        .map(|e| match e.as_any().downcast_ref::<LambdaExpr>() {
            Some(lambda) => Ok(ScalarFunctionArgMetadata::Lambda(lambda.params())),
            None => Ok(ScalarFunctionArgMetadata::Value(e.data_type(schema)?)),
        })
        .collect::<Result<Vec<_>>>()?;

    let captures = args
        .iter()
        .map(|arg| {
            if arg.as_any().is::<LambdaExpr>() {
                let mut columns = HashSet::new();

                arg.apply_with_lambdas_params(|n, lambdas_params| {
                    if let Some(column) = n.as_any().downcast_ref::<Column>() {
                        if !lambdas_params.contains(column.name()) {
                            columns.insert(schema.index_of(column.name())?);
                        }
                        // columns.insert(column.index());
                    }

                    Ok(TreeNodeRecursion::Continue)
                })?;

                Ok(columns)
            } else {
                Ok(HashSet::new())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(fun
        .lambdas_schemas(
            &args_metadata,
            &captures,
            &DFSchema::try_from(schema.clone()).unwrap(),
        )?
        .into_iter()
        .map(|dfschema| dfschema.into_owned().into())
        .collect())
}

pub trait PhysicalExprExt {
    fn apply_with_lambdas_params<
        F: FnMut(&Self, &HashSet<&str>) -> Result<TreeNodeRecursion>,
    >(
        &self,
        f: F,
    ) -> Result<TreeNodeRecursion>;

    fn apply_with_schema<'n, F: FnMut(&'n Self, &Schema) -> Result<TreeNodeRecursion>>(
        &'n self,
        schema: &Schema,
        f: F,
    ) -> Result<TreeNodeRecursion>;

    fn apply_children_with_schema<
        'n,
        F: FnMut(&'n Self, &Schema) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        schema: &Schema,
        f: F,
    ) -> Result<TreeNodeRecursion>;
}

impl PhysicalExprExt for Arc<dyn PhysicalExpr> {
    fn apply_with_lambdas_params<
        F: FnMut(&Self, &HashSet<&str>) -> Result<TreeNodeRecursion>,
    >(
        &self,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn apply_with_lambdas_params_impl<
            F: FnMut(&Arc<dyn PhysicalExpr>, &HashSet<&str>) -> Result<TreeNodeRecursion>,
        >(
            node: &Arc<dyn PhysicalExpr>,
            args: &HashSet<&str>,
            f: &mut F,
        ) -> Result<TreeNodeRecursion> {
            match node.as_any().downcast_ref::<LambdaExpr>() {
                Some(lambda) => {
                    let mut args = args.clone();

                    args.extend(lambda.params().iter().map(|v| v.as_str()));

                    f(node, &args)?.visit_children(|| {
                        node.apply_children(|c| {
                            apply_with_lambdas_params_impl(c, &args, f)
                        })
                    })
                }
                _ => f(node, args)?.visit_children(|| {
                    node.apply_children(|c| apply_with_lambdas_params_impl(c, args, f))
                }),
            }
        }

        apply_with_lambdas_params_impl(self, &HashSet::new(), &mut f)
    }

    fn apply_with_schema<'n, F: FnMut(&'n Self, &Schema) -> Result<TreeNodeRecursion>>(
        &'n self,
        schema: &Schema,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn apply_with_lambdas_impl<
            'n,
            F: FnMut(&'n Arc<dyn PhysicalExpr>, &Schema) -> Result<TreeNodeRecursion>,
        >(
            node: &'n Arc<dyn PhysicalExpr>,
            schema: &Schema,
            f: &mut F,
        ) -> Result<TreeNodeRecursion> {
            f(node, schema)?.visit_children(|| {
                node.apply_children_with_schema(schema, |c, schema| {
                    apply_with_lambdas_impl(c, schema, f)
                })
            })
        }

        apply_with_lambdas_impl(self, schema, &mut f)
    }

    fn apply_children_with_schema<
        'n,
        F: FnMut(&'n Self, &Schema) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        schema: &Schema,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        if let Some(scalar_function) = self.as_any().downcast_ref::<ScalarFunctionExpr>()
        {
            let mut lambdas_schemas = lambdas_schemas_from_args(
                scalar_function.fun(),
                scalar_function.args(),
                schema,
            )?
            .into_iter();

            self.apply_children(|expr| f(expr, &lambdas_schemas.next().unwrap()))
        } else {
            self.apply_children(|e| f(e, schema))
        }
    }
}
