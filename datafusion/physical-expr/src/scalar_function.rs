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
use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use crate::expressions::{Column, LambdaExpr, Literal};
use crate::PhysicalExpr;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::Field;
use arrow::datatypes::{DataType, Schema};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{internal_err, HashSet, Result, ScalarValue};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::sort_properties::ExprProperties;
use datafusion_expr::type_coercion::functions::data_types_with_scalar_udf;
use datafusion_expr::{
    expr_vec_fmt, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarFunctionLambdaArg, ScalarUDF, ValueOrLambdaParameter,
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

        let args_metadata = std::iter::zip(&self.args, &arg_fields_owned)
            .map(
                |(expr, field)| match expr.as_any().downcast_ref::<LambdaExpr>() {
                    Some(lambda) => ValueOrLambdaParameter::Lambda(lambda.params()),
                    None => ValueOrLambdaParameter::Value(field.clone()),
                },
            )
            .collect::<Vec<_>>();

        let params = self.fun().inner().lambdas_parameters(&args_metadata)?;

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

                        let params =
                            std::iter::zip(lambda.params(), lambda_params.unwrap())
                                .map(|(name, param)| Arc::new(param.with_name(name)))
                                .collect();

                        let captures = if !indices.is_empty() {
                            Some(batch.project(&indices)?)
                        } else {
                            None
                        };

                        Ok(ScalarFunctionLambdaArg {
                            params,
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

pub fn lambdas_schemas_from_args<'a>(
    fun: &ScalarUDF,
    args: &[Arc<dyn PhysicalExpr>],
    schema: &'a Schema,
) -> Result<Vec<Cow<'a, Schema>>> {
    let args_metadata = args
        .iter()
        .map(|e| match e.as_any().downcast_ref::<LambdaExpr>() {
            Some(lambda) => Ok(ValueOrLambdaParameter::Lambda(lambda.params())),
            None => Ok(ValueOrLambdaParameter::Value(e.return_field(schema)?)),
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

    fun.arguments_arrow_schema(&args_metadata, &captures, schema)
}

pub trait PhysicalExprExt {
    fn apply_with_lambdas_params<
        'n,
        F: FnMut(&'n Self, &HashSet<&'n str>) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
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
        'n,
        F: FnMut(&'n Self, &HashSet<&'n str>) -> Result<TreeNodeRecursion>,
    >(
        &'n self,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
        fn apply_with_lambdas_params_impl<
            'n,
            F: FnMut(
                &'n Arc<dyn PhysicalExpr>,
                &HashSet<&'n str>,
            ) -> Result<TreeNodeRecursion>,
        >(
            node: &'n Arc<dyn PhysicalExpr>,
            args: &HashSet<&'n str>,
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

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::{tree_node::TreeNodeRecursion, DFSchema, HashSet, Result};
    use datafusion_expr::{
        col, expr::Lambda, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
        ValueOrLambdaParameter, Volatility,
    };
    use datafusion_expr_common::{columnar_value::ColumnarValue, signature::Signature};
    use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

    use super::{lambdas_schemas_from_args, PhysicalExprExt};
    use crate::{create_physical_expr, ScalarFunctionExpr};

    fn list_list_int() -> Schema {
        Schema::new(vec![Field::new(
            "v",
            DataType::new_list(DataType::new_list(DataType::Int32, false), false),
            false,
        )])
    }

    fn list_int() -> Schema {
        Schema::new(vec![Field::new(
            "v",
            DataType::new_list(DataType::Int32, false),
            false,
        )])
    }

    fn int() -> Schema {
        Schema::new(vec![Field::new("v", DataType::Int32, false)])
    }

    fn list_transform_udf() -> ScalarUDF {
        ScalarUDF::new_from_impl(ListMapFunc::new())
    }

    fn args() -> Vec<Expr> {
        vec![
            col("v"),
            Expr::Lambda(Lambda::new(
                vec!["v".into()],
                list_transform_udf().call(vec![
                    col("v"),
                    Expr::Lambda(Lambda::new(vec!["v".into()], -col("v"))),
                ]),
            )),
        ]
    }

    // list_transform(v, |v| -> list_transform(v, |v| -> -v))
    fn list_transform() -> Arc<dyn PhysicalExpr> {
        let e = list_transform_udf().call(args());

        create_physical_expr(
            &e,
            &DFSchema::try_from(list_list_int()).unwrap(),
            &Default::default(),
        )
        .unwrap()
    }

    #[derive(Debug)]
    struct ListMapFunc {
        signature: Signature,
    }

    impl ListMapFunc {
        pub fn new() -> Self {
            Self {
                signature: Signature::any(2, Volatility::Immutable),
            }
        }
    }

    impl ScalarUDFImpl for ListMapFunc {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn name(&self) -> &str {
            "list_transform"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }

        fn lambdas_parameters(
            &self,
            args: &[ValueOrLambdaParameter],
        ) -> Result<Vec<Option<Vec<Field>>>> {
            let ValueOrLambdaParameter::Value(value_field) = &args[0] else {
                unimplemented!()
            };
            let DataType::List(field) = value_field.data_type() else {
                unimplemented!()
            };

            Ok(vec![
                None,
                Some(vec![Field::new(
                    "",
                    field.data_type().clone(),
                    field.is_nullable(),
                )]),
            ])
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unimplemented!()
        }
    }

    #[test]
    fn test_lambdas_schemas_from_args() {
        let schema = list_list_int();
        let expr = list_transform();

        let args = expr
            .as_any()
            .downcast_ref::<ScalarFunctionExpr>()
            .unwrap()
            .args();

        let schemas =
            lambdas_schemas_from_args(&list_transform_udf(), args, &schema).unwrap();

        assert_eq!(schemas, &[Cow::Borrowed(&schema), Cow::Owned(list_int())]);
    }

    #[test]
    fn test_apply_with_schema() {
        let mut steps = vec![];

        list_transform()
            .apply_with_schema(&list_list_int(), |node, schema| {
                steps.push((node.to_string(), schema.clone()));

                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        let expected = [
            (
                "list_transform(v@0, (v) -> list_transform(v@0, (v) -> (- v@0)))",
                list_list_int(),
            ),
            ("(v) -> list_transform(v@0, (v) -> (- v@0))", list_int()),
            ("list_transform(v@0, (v) -> (- v@0))", list_int()),
            ("(v) -> (- v@0)", int()),
            ("(- v@0)", int()),
            ("v@0", int()),
            ("v@0", int()),
            ("v@0", int()),
        ]
        .map(|(a, b)| (String::from(a), b));

        assert_eq!(steps, expected);
    }

    #[test]
    fn test_apply_with_lambdas_params() {
        let list_transform = list_transform();
        let mut steps = vec![];

        list_transform
            .apply_with_lambdas_params(|node, params| {
                steps.push((node.to_string(), params.clone()));

                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        let expected = [
            (
                "list_transform(v@0, (v) -> list_transform(v@0, (v) -> (- v@0)))",
                HashSet::from(["v"]),
            ),
            (
                "(v) -> list_transform(v@0, (v) -> (- v@0))",
                HashSet::from(["v"]),
            ),
            ("list_transform(v@0, (v) -> (- v@0))", HashSet::from(["v"])),
            ("(v) -> (- v@0)", HashSet::from(["v"])),
            ("(- v@0)", HashSet::from(["v"])),
            ("v@0", HashSet::from(["v"])),
            ("v@0", HashSet::from(["v"])),
            ("v@0", HashSet::from(["v"])),
        ]
        .map(|(a, b)| (String::from(a), b));

        assert_eq!(steps, expected);
    }
}
