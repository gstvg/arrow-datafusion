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

use crate::expressions::{new_expr_with_schema, Column, Lambda, Literal};
use crate::PhysicalExpr;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, Schema};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{internal_err, DFSchema, HashSet, Result, ScalarValue};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::sort_properties::ExprProperties;
use datafusion_expr::type_coercion::functions::data_types_with_scalar_udf;
use datafusion_expr::{
    expr_vec_fmt, ColumnarValue, Expr, ReturnTypeArgs, ScalarFunctionArgMetadata,
    ScalarFunctionArgs, ScalarFunctionLambdaArg, ScalarUDF, IS_LAMBDA_ARG,
};

/// Physical expression of a scalar function
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct ScalarFunctionExpr {
    fun: Arc<ScalarUDF>,
    name: String,
    args: Vec<Arc<dyn PhysicalExpr>>,
    return_type: DataType,
    nullable: bool,
}

impl Debug for ScalarFunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ScalarFunctionExpr")
            .field("fun", &"<FUNC>")
            .field("name", &self.name)
            .field("args", &self.args)
            .field("return_type", &self.return_type)
            .finish()
    }
}

impl ScalarFunctionExpr {
    /// Create a new Scalar function
    pub fn new(
        name: &str,
        fun: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        return_type: DataType,
    ) -> Self {
        Self {
            fun,
            name: name.to_owned(),
            args,
            return_type,
            nullable: true,
        }
    }

    /// Create a new Scalar function
    pub fn try_new(
        fun: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        schema: &Schema,
    ) -> Result<Self> {
        let lambdas_schemas = lambdas_schemas_from_args(&fun, &args, schema)?;

        let (arg_types, nullables): (Vec<_>, Vec<_>) =
            std::iter::zip(&args, lambdas_schemas)
                .map(|(e, lambda_schema)| {
                    if let Some(lambda) = e.as_any().downcast_ref::<Lambda>() {
                        let schema = lambda_schema.unwrap();

                        Ok((
                            lambda.inner().data_type(schema.as_arrow())?,
                            lambda.inner().nullable(schema.as_arrow())?,
                        ))
                    } else {
                        Ok((e.data_type(schema)?, e.nullable(schema)?))
                    }
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();

        // verify that input data types is consistent with function's `TypeSignature`
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
            .map(|e| e.as_any().is::<Lambda>())
            .collect::<Vec<_>>();

        let ret_args = ReturnTypeArgs {
            arg_types: &arg_types,
            scalar_arguments: &arguments,
            nullables: &nullables,
            lambdas: &lambdas,
        };
        let (return_type, nullable) = fun.return_type_from_args(ret_args)?.into_parts();

        let name = fun.name().to_string();

        Ok(Self {
            fun,
            name,
            args,
            return_type,
            nullable,
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
        &self.return_type
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    pub fn nullable(&self) -> bool {
        self.nullable
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
        Ok(self.return_type.clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(self.nullable)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let args = self
            .args
            .iter()
            .map(|e| match e.as_any().downcast_ref::<Lambda>() {
                Some(_) => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
                None => Ok(e.evaluate(batch)?),
            })
            .collect::<Result<Vec<_>>>()?;

        let input_empty = args.is_empty();
        let input_all_scalar = args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));

        let self_with_schema =
            new_expr_with_schema(Arc::new(self.clone()), batch.schema())?;

        let lambdas = self_with_schema
            .children
            .iter()
            .map(|a| {
                a.expr
                    .as_any()
                    .downcast_ref::<Lambda>()
                    .map(|lambda| {
                        // let mut captured_indices = vec![false; batch.num_columns()];

                        // a.apply(|ctx| {
                        //     if let Some(column) =
                        //         ctx.expr.as_any().downcast_ref::<Column>()
                        //     {
                        //         let field = ctx.data.field_with_name(column.name())?;

                        //         if !field.metadata().contains_key(IS_LAMBDA_ARG) {
                        //             let index = batch.schema_ref().index_of(field.name())?;

                        //             captured_indices[index] = true;
                        //         }
                        //     }

                        //     Ok(TreeNodeRecursion::Continue)
                        // })?;

                        // let null_array = Arc::new(NullArray::new(batch.num_rows())) as ArrayRef;

                        // let (fields, arrays): (Vec<_>, _) = std::iter::zip(batch.schema_ref().fields(), captured_indices)
                        //     .enumerate()
                        //     .map(|(i, (field, captured))| if captured {
                        //         (Arc::clone(field), Arc::clone(batch.column(i)))
                        //     } else {
                        //         (Arc::new(field.as_ref().clone().with_data_type(DataType::Null)), Arc::clone(&null_array))
                        //     })
                        //     .unzip();

                        // let captures = RecordBatch::try_new_with_options(
                        //     Arc::new(Schema::new(Fields::from(fields))),
                        //     arrays,
                        //     &RecordBatchOptions::new().with_match_field_names(true).with_row_count(Some(batch.num_rows()))
                        // )?;

                        let indices = a.data.fields()
                            .iter()
                            .filter(|field| !field.metadata().contains_key(IS_LAMBDA_ARG))
                            .map(|field| batch.schema_ref().index_of(field.name()))
                            .collect::<Result<Vec<_>, _>>()?;

                        Ok(ScalarFunctionLambdaArg {
                            args_names: lambda.args(),
                            body: lambda.inner().as_ref(),
                            captures: batch.project(&indices)?,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;

        // evaluate the function
        let output = self.fun.invoke_with_args(ScalarFunctionArgs {
            args,
            number_rows: batch.num_rows(),
            return_type: &self.return_type,
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

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.args.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(
            ScalarFunctionExpr::new(
                &self.name,
                Arc::clone(&self.fun),
                children,
                self.return_type().clone(),
            )
            .with_nullable(self.nullable),
        ))
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
}

pub fn lambdas_schemas_from_args(fun: &ScalarUDF, args: &[Arc<dyn PhysicalExpr>], schema: &Schema) -> Result<Vec<Option<DFSchema>>> {         
    let args_metadata = args
        .iter()
        .map(|e| match e.as_any().downcast_ref::<Lambda>() {
            Some(lambda) => Ok(ScalarFunctionArgMetadata::Lambda(lambda.args())),
            None => Ok(ScalarFunctionArgMetadata::Value(e.data_type(schema)?)),
        })
        .collect::<Result<Vec<_>>>()?;

    let captures = args.iter()
        .map(|arg| match arg.as_any().downcast_ref::<Lambda>() {
            Some(lambda) => {
                let mut columns = HashSet::new();

                apply_with_lambdas2(lambda.inner(), |n| {
                    if let Some(column) = n.as_any().downcast_ref::<Column>() {
                        if let Ok(index) = schema.index_of(column.name()) {
                            columns.insert(index);
                        }
                        // columns.insert(column.index());
                    }

                    Ok(TreeNodeRecursion::Continue)
                })?;

                Ok(columns)
            }
            None => Ok(HashSet::new()),
        })
        .collect::<Result<Vec<_>>>()?;

    //TOOD: augment every lambda schema with the outer schema
    fun.lambdas_schemas(
        &args_metadata,
        &captures,
        &DFSchema::try_from(schema.clone()).unwrap(),
    )
}

pub fn apply_with_lambdas2<
    'n,
    F: FnMut(&'n Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
>(
    this: &'n Arc<dyn PhysicalExpr>,
    mut f: F,
) -> Result<TreeNodeRecursion> {
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn apply_impl<'n, F: FnMut(&'n Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>>(
        node: &'n Arc<dyn PhysicalExpr>,
        f: &mut F,
    ) -> Result<TreeNodeRecursion> {
        if let Some(lambda) = node.as_any().downcast_ref::<Lambda>() {
            f(node)?.visit_children(|| apply_impl(lambda.inner(), f))
        } else {
            f(node)?.visit_children(|| node.apply_children(|c| apply_impl(c, f)))
        }
    }

    apply_impl(this, &mut f)
}

/// Create a physical expression for the UDF.
#[deprecated(since = "45.0.0", note = "use ScalarFunctionExpr::new() instead")]
pub fn create_physical_expr(
    fun: &ScalarUDF,
    input_phy_exprs: &[Arc<dyn PhysicalExpr>],
    input_schema: &Schema,
    args: &[Expr],
    input_dfschema: &DFSchema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let input_expr_types = input_phy_exprs
        .iter()
        .map(|e| e.data_type(input_schema))
        .collect::<Result<Vec<_>>>()?;

    // verify that input data types is consistent with function's `TypeSignature`
    data_types_with_scalar_udf(&input_expr_types, fun)?;

    // Since we have arg_types, we don't need args and schema.
    let return_type =
        fun.return_type_from_exprs(args, input_dfschema, &input_expr_types)?;

    Ok(Arc::new(
        ScalarFunctionExpr::new(
            fun.name(),
            Arc::new(fun.clone()),
            input_phy_exprs.to_vec(),
            return_type,
        )
        .with_nullable(fun.is_nullable(args, input_dfschema)),
    ))
}
