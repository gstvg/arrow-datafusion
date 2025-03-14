

struct ExprWithSchema {
    pub expr: Expr,
    pub schema: SchemaRef,
}

impl TreeNode for ExprWithSchema {
    fn apply_children<'n, F: FnMut(&'n Self) -> Result<TreeNodeRecursion>>(
        &'n self,
        mut f: F,
    ) -> Result<TreeNodeRecursion> {
        
        if let Expr::ScalarFunction(ScalarFunction{ func, args }) = self.expr {
            let lambdas_schemas = func.lambdas_schemas_from_args(&args, &self.schema)?;

            for (arg, lambda_schema) in std::iter::zip(args, lambda_schemas) {
                // this is an owned ExprWithSchema, but the TreeNode trait require an &ExprWithSchema
                let tnr = arg.apply_children(|e| ExprWithSchema {
                    // this is a &Expr, but on map_children this is an owned Expr, 
                    expr: v,
                    schema: lambda_schema.unwrap_or_else(|| self.schema.clone())
                })?;

                match tnr {
                    TreeNodeRecursion::Continue | TreeNodeRecursion::Jump => {}
                    TreeNodeRecursion::Stop => return Ok(TreeNodeRecursion::Stop),
                }
            }

        } else {
            self.expr.apply_children(|v| {
                ExprWithSchema {
                    expr: v,
                    schema: self.schema.clone(),
                }
            })
        }
    }

    fn map_children<F: FnMut(Self) -> Result<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        if let Expr::ScalarFunction(ScalarFunction{ func, args }) = &self.expr {
            let lambdas_schemas = func.lambdas_schemas_from_args(&args, &self.schema)?;

            for (arg, lambda_schema) in std::iter::zip(args, lambda_schemas) {

            }
        } else {
            self.expr
                .map_children(|expr| {
                    f(ExprWithSchema {
                        expr,
                        data: self.schema.clone(),
                    })
                    .map(|t| t.update_data(|v| v.expr))
                })
                .map(|v| {
                    v.update_data(|exp| ExprWithSchema {
                        expr,
                        data: self.schema.clone(),
                    })
                })
        }
    }
}

// struct LogicalExprContext<'a, T> {
//     pub expr: &'a Expr,
//     pub data: T,
//     pub children: Vec<LogicalExprContext<'a, T>>,
// }

// impl<'a, T: Clone> LogicalExprContext<'a, T> {
//     pub fn new(expr: &'a Expr, data: T) -> Self {
//         let children = match expr {
//             Expr::Alias(alias) => vec![Self::new(&alias.expr, data.clone())],
//             Expr::Column(column) => vec![],
//             Expr::ScalarVariable(data_type, items) => vec![],
//             Expr::Literal(scalar_value) => vec![],
//             Expr::BinaryExpr(binary_expr) => vec![
//                 Self::new(&binary_expr.left, data.clone()),
//                 Self::new(&binary_expr.right, data.clone()),
//             ],
//             Expr::Like(like) | Expr::SimilarTo(like) => vec![
//                 Self::new(&like.expr, data.clone()),
//                 Self::new(&like.pattern, data.clone()),
//             ],
//             Expr::Not(expr)
//             | Expr::IsNotNull(expr)
//             | Expr::IsNull(expr)
//             | Expr::IsTrue(expr)
//             | Expr::IsFalse(expr)
//             | Expr::IsUnknown(expr)
//             | Expr::IsNotTrue(expr)
//             | Expr::IsNotFalse(expr)
//             | Expr::IsNotUnknown(expr)
//             | Expr::Negative(expr) => vec![Self::new(&expr, data.clone())],
//             Expr::Between(between) => vec![
//                 Self::new(&between.expr, data.clone()),
//                 Self::new(&between.low, data.clone()),
//                 Self::new(&between.high, data.clone()),
//             ],
//             Expr::Case(case) => todo!(),
//             Expr::Cast(cast) => todo!(),
//             Expr::TryCast(try_cast) => todo!(),
//             Expr::ScalarFunction(scalar_function) => todo!(),
//             Expr::AggregateFunction(aggregate_function) => todo!(),
//             Expr::WindowFunction(window_function) => todo!(),
//             Expr::InList(in_list) => todo!(),
//             Expr::Exists(exists) => todo!(),
//             Expr::InSubquery(in_subquery) => todo!(),
//             Expr::ScalarSubquery(subquery) => todo!(),
//             Expr::Wildcard { qualifier, options } => todo!(),
//             Expr::GroupingSet(grouping_set) => todo!(),
//             Expr::Placeholder(placeholder) => todo!(),
//             Expr::OuterReferenceColumn(data_type, column) => todo!(),
//             Expr::Unnest(unnest) => todo!(),
//             Expr::Lambda { arg_names, expr } => todo!(),
//         };

//         Self {
//             expr,
//             data,
//             children,
//         }
//     }
// }

// impl<'a, T> ConcreteTreeNode for LogicalExprContext<'a, T> {
//     fn children(&self) -> &[Self] {
//         &self.children
//     }

//     fn take_children(mut self) -> (Self, Vec<Self>) {
//         let children = std::mem::take(&mut self.children);
//         (self, children)
//     }

//     fn with_new_children(mut self, children: Vec<Self>) -> Result<Self> {
//         self.children = children;
//         Err(todo!())
//     }
// }

/*

Regardless of the decided criteria, one suggestion to increase the discoverability of external functions is to add an select input with the external projects as options on the datafusion SQL functions docs to also list functions of the selected projects, if any, alongside core functions, with a clear marking stating that is a external function and a link for the containing project/repo.

Those projects could have a file in it's repo, like `functions.json`, that datafusion would pull and add to the docs. The file could be auto generated by serializing [Documentation](https://docs.rs/datafusion/latest/datafusion/logical_expr/struct.Documentation.html), by adding `#[derive(Serialize, Deserizalize)]` to it.
Whether the file should include functions of the main branch or the lastest stable release of the project is a open question

 Any error while pulling or processing the file could be ignored, and either the functions are temporally removed from the docs, or the last valid file is used. Propagating any error and blocking the CI or anyone working on the core would be bad.

Other options like running the external project to generate the docs on the fly or including the external project as git submodule are problematic, at best.

*/

fn concat_unions(unions: &[UnionArray]) -> Result<UnionArray> {
    let DataType::Union(fields, _) = unions[0].data_type() else {
        unreachable!()
    };

    let len = unions.iter().map(|u| u.len()).sum();

    let mut offsets = Vec::with_capacity(len);
    let mut type_ids = Vec::with_capacity(len);

    offsets.extend(unions[0].offsets().unwrap());
    type_ids.extend(unions[0].type_ids());

    let mut field_len_scan = [0; 256];

    for union in &unions[1..] {
        let type_ids = union.type_ids();
        let offsets = union.offsets().unwrap();

        let mut delta = [0; 256];

        let likely_sliced = fields
            .iter()
            .map(|(type_id, _field)| union.child(type_id).len())
            .sum()
            > union.len()
            && type_ids.inner().ptr_offset() > 0;

        if likely_sliced {
            let mut first_pos = [0; 256];
            let mut last_pos = [0; 256];

            for (i, type_id) in type_ids.iter().enumerate().rev() {
                first_pos[*type_id as u8 as usize] = i;
            }

            for (i, type_id) in type_ids.iter().enumerate() {
                last_pos[*type_id as u8 as usize] = i;
            }

            for (type_id, _field) in fields.iter() {
                let child = union.child(type_id);

                let sliced = child
                    .slice(first_pos[type_id], last_pos[type_id] - first_post[type_id]);

                delta[type_id as usize] = child.len() - sliced.len();
                field_len_scan[type_id as usize] += sliced.len() as i32;
            }
        } else {
            for (type_id, _field) in fields.iter() {
                field_len_scan[type_id as usize] += union.child(type_id).len() as i32;
            }
        }

        offsets.extend(
            std::iter::zip(type_ids, offsets).map(|(type_id, offset)| {
                field_len_scan[*type_id as u8 as usize] + *offset
            }),
        );

        if likely_sliced {
            for (field_len_scan, adjust) in std::iter::zip(&mut field_len_scan, &delta) {
                *field_len_scan -= adjust;
            }
        }
    }

    let children = fields
        .iter()
        .map(|(type_id, _field)| {
            concat(&unions.iter().map(|u| u.child(type_id).collect::<Vec<_>>()))
        })
        .collect::<Result<_>>()?;

    unsafe {
        Ok(UnionArray::new_unchecked(
            fields.clone(),
            type_ids.into(),
            Some(offsets.into()),
            children,
        ))
    }
}

fn u(type_ids: &[i8], offsets: &[i32]) -> [i32; 256] {

    let mut sum = [0; 256];

    for (&type_id, &offset) in std::iter::zip(type_ids, offsets) {
        let idx = type_id as u8 as usize;
        sum[idx] = if sum[idx] + 1 == offset { offset } else { -1 };
    }

    sum
}

impl Expr {
    pub fn visit_with_schema<'a, V>(
        &'a self,
        schema: Schema,
        visitor: &'a mut V,
    ) -> Result<TreeNodeRecursion>
    where
        &'a mut V: for<'b> TreeNodeVisitor<'b, Node = (&'a Expr, &'b Schema)>,
    {
        let mut wrapper = VisitorWrapper::new(schema, visitor);

        self.visit(&mut wrapper)
    }

    pub fn rewrite_with_schema<R>(
        self,
        schema: Schema,
        rewriter: &mut R,
    ) -> Result<Transformed<Self>>
    where
        R: TreeNodeRewriter<Node = (Expr, Arc<Schema>)>,
    {
        let mut wrapper = RewriterWrapper::new(schema, rewriter);

        self.rewrite(&mut wrapper)
    }

    pub fn apply_with_schema<'n, F>(
        &'n self,
        schema: Schema,
        mut f: F,
    ) -> Result<TreeNodeRecursion>
    where
        F: FnMut(&'n Expr, &Schema) -> Result<TreeNodeRecursion>,
    {
        let mut visitor = LambdaAwareVisitor::new(
            schema,
            |e: &'n Expr, s: &Schema| f(e, s),
            |_: &'n Expr, _: &Schema| Ok(TreeNodeRecursion::Continue),
        );

        self.visit(&mut visitor)
    }

    pub fn transform_with_schema<F>(
        self,
        schema: Schema,
        mut f: F,
    ) -> Result<Transformed<Self>>
    where
        F: FnMut(Self, &Schema) -> Result<Transformed<Self>>,
    {
        self.transform(|expr| f(expr, &schema))
    }

    pub fn transform_down_with_schema<F>(
        self,
        schema: Schema,
        mut f: F,
    ) -> Result<Transformed<Self>>
    where
        F: FnMut(Self, &Schema) -> Result<Transformed<Self>>,
    {
        self.transform_down(|expr| f(expr, &schema))
    }

    pub fn transform_up_with_schema<F>(
        self,
        schema: Schema,
        mut f: F,
    ) -> Result<Transformed<Self>>
    where
        F: FnMut(Self, &Schema) -> Result<Transformed<Self>>,
    {
        self.transform_up(|expr| f(expr, &schema))
    }

    pub fn transform_down_up_with_schema<FD, FU>(
        self,
        schema: Schema,
        mut f_down: FD,
        mut f_up: FU,
    ) -> Result<Transformed<Self>>
    where
        FD: FnMut(Self, &Schema) -> Result<Transformed<Self>>,
        FU: FnMut(Self, &Schema) -> Result<Transformed<Self>>,
    {
        let mut rewriter = LambdaAwareRewriter::new(
            schema,
            |expr, schema: &Schema| f_up(expr, schema),
            |expr, schema: &Schema| f_down(expr, schema),
        );

        self.rewrite(&mut rewriter)
    }

    pub fn exists_with_schema<F>(&self, schema: Schema, mut f: F) -> Result<bool>
    where
        F: FnMut(&Expr, &Schema) -> Result<bool>,
    {
        let mut exists = false;

        let mut lambda_visitor = LambdaAwareVisitor::new(
            schema,
            |e: &Expr, s: &Schema| {
                if f(e, s)? {
                    exists = true;
                    Ok(TreeNodeRecursion::Stop)
                } else {
                    Ok(TreeNodeRecursion::Continue)
                }
            },
            |_: &Expr, _: &Schema| Ok(TreeNodeRecursion::Continue),
        );

        self.visit(&mut lambda_visitor)?;

        Ok(exists)
    }
}

type LambdaSchemaIterator =
    std::iter::Flatten<<Vec<Option<Schema>> as IntoIterator>::IntoIter>;

struct SchemaStack {
    depth: usize,
    schemas: Vec<Schema>,
    lambdas_schemas: Vec<(usize, LambdaSchemaIterator)>,
}

impl SchemaStack {
    fn new(schema: Schema) -> Self {
        Self {
            depth: 0,
            schemas: vec![schema],
            lambdas_schemas: vec![],
        }
    }

    fn reset(&mut self) {
        self.depth = 0;
        self.schemas.truncate(1);
        self.lambdas_schemas.truncate(0);
    }

    fn down_scalar_function(&mut self, lambdas_schemas: Vec<Option<Schema>>) {
        self.lambdas_schemas
            .push((self.depth, lambdas_schemas.into_iter().flatten()));
    }

    fn down_lambda(&mut self) -> Result<()> {
        if self.depth == 1 {
            return exec_err!("visiting a top level lambda is not supported");
        }

        let (scalar_function_depth, lambdas_schemas) = self
            .lambdas_schemas
            .last_mut()
            .ok_or_else(|| internal_datafusion_err!("there should be at least one lambda schema iterator added by the parent ScalarFunction expr"))?;

        if *scalar_function_depth != self.depth - 1 {
            return exec_err!(
                "lambdas should only exist as direct child of a scalar functions"
            );
        }

        let schema = lambdas_schemas.next().ok_or_else(|| {
            internal_datafusion_err!("there should be a schema for every lambda")
        })?;

        self.schemas.push(schema);

        Ok(())
    }

    fn down_with(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        self.depth += 1;

        f(self)
    }

    fn down(&mut self, node: &Expr) -> Result<()> {
        self.depth += 1;

        match node {
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                let schema = self.schema()?;

                let lambdas_schemas = func.lambdas_schemas_from_args(
                    args,
                    &DFSchema::try_from(schema.clone()).unwrap(),
                )?;

                self.down_scalar_function(lambdas_schemas);
            }
            Expr::Lambda { .. } => {
                self.down_lambda()?;
            }
            _ => {}
        }

        Ok(())
    }

    fn up_lambda(&mut self) -> Result<()> {
        self.schemas
            .pop()
            .ok_or_else(|| internal_datafusion_err!(""))?;

        Ok(())
    }

    fn up_scalar_function(&mut self) -> Result<()> {
        let _ = self
            .lambdas_schemas
            .pop()
            .ok_or_else(|| internal_datafusion_err!(""))?;

        Ok(())
    }

    fn up(&mut self, node: &Expr) -> Result<()> {
        self.depth -= 1;

        match node {
            Expr::Lambda { .. } => self.up_lambda(),
            Expr::ScalarFunction(_) => self.up_scalar_function(),
            _ => Ok(()),
        }
    }

    fn schema(&self) -> Result<&Schema> {
        self.schemas.last().ok_or_else(|| {
            internal_datafusion_err!(
                "at least one schema should exist, and never be removed"
            )
        })
    }
}

struct VisitorWrapper<V> {
    stack: SchemaStack,
    visitor: V,
}

impl<V> VisitorWrapper<V> {
    fn new(schema: Schema, visitor: V) -> Self {
        Self {
            stack: SchemaStack::new(schema),
            visitor,
        }
    }
}

impl<'n, V> TreeNodeVisitor<'n> for VisitorWrapper<V>
where
    V: for<'s> TreeNodeVisitor<'s, Node = (&'n Expr, &'s Schema)>,
{
    type Node = Expr;

    fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        let schema = self.stack.schema()?;

        let tnr = self.visitor.f_down(&(node, schema));

        self.stack.down(node)?;

        tnr
    }

    fn f_up(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        let schema = self.stack.schema()?;

        let tnr = self.visitor.f_up(&(node, schema));

        self.stack.up(node)?;

        tnr
    }
}

struct LambdaAwareVisitor<FU, FD>
// where
//     FU: FnMut(&'n Expr, &Schema) -> Result<TreeNodeRecursion>,
//     FD: FnMut(&'n Expr, &Schema) -> Result<TreeNodeRecursion>,
{
    stack: SchemaStack,
    fu: FU,
    fd: FD,
    // phantom: std::marker::PhantomData<&'n ()>,
}

impl<FU, FD> LambdaAwareVisitor<FU, FD>
// where
//     FU: FnMut(&'n Expr, &Schema) -> Result<TreeNodeRecursion>,
//     FD: FnMut(&'n Expr, &Schema) -> Result<TreeNodeRecursion>,
{
    fn new(schema: Schema, fu: FU, fd: FD) -> Self {
        Self {
            stack: SchemaStack::new(schema),
            fu,
            fd,
            // phantom: PhantomData,
        }
    }
}

impl<'n, FU, FD> TreeNodeVisitor<'n> for LambdaAwareVisitor<FU, FD>
where
    FU: for<'s> FnMut(&'n Expr, &'s Schema) -> Result<TreeNodeRecursion>,
    FD: for<'s> FnMut(&'n Expr, &'s Schema) -> Result<TreeNodeRecursion>,
{
    type Node = Expr;

    fn f_down(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        let schema = self.stack.schema()?;

        let tnr = (self.fu)(node, schema);

        self.stack.down(node)?;

        tnr
    }

    fn f_up(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        let schema = self.stack.schema()?;

        let tnr = (self.fd)(node, schema);

        self.stack.up(node)?;

        tnr
    }
}

struct LambdaAwareRewriter<FU, FD> {
    stack: SchemaStack,
    fu: FU,
    fd: FD,
}

impl<FU, FD> LambdaAwareRewriter<FU, FD> {
    fn new(schema: Schema, fu: FU, fd: FD) -> Self {
        Self {
            stack: SchemaStack::new(schema),
            fu,
            fd,
        }
    }
}

impl<FU, FD> TreeNodeRewriter for LambdaAwareRewriter<FU, FD>
where
    FU: for<'s> FnMut(Expr, &'s Schema) -> Result<Transformed<Expr>>,
    FD: for<'s> FnMut(Expr, &'s Schema) -> Result<Transformed<Expr>>,
{
    type Node = Expr;

    fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let schema = self.stack.schema()?;

        let transformed = (self.fu)(node, schema)?;

        self.stack.down(&transformed.data)?;

        Ok(transformed)
    }

    fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let schema = self.stack.schema()?;

        let transformed = (self.fd)(node, schema)?;

        self.stack.up(&transformed.data)?;

        Ok(transformed)
    }
}

struct RewriterWrapper<'a, R> {
    stack: SchemaStack,
    rewriter: &'a mut R,
}

impl<'a, R> RewriterWrapper<'a, R> {
    fn new(schema: Schema, rewriter: &'a mut R) -> Self {
        Self {
            stack: SchemaStack::new(schema),
            rewriter,
        }
    }
}

impl<R> TreeNodeRewriter for RewriterWrapper<'_, R>
where
    R: TreeNodeRewriter<Node = (Expr, Arc<Schema>)>,
{
    type Node = Expr;

    fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let schema = Arc::new(self.stack.schema()?.clone());

        let transformed = self
            .rewriter
            .f_down((node, schema))?
            .update_data(|(expr, _schema)| expr);

        self.stack.down(&transformed.data)?;

        Ok(transformed)
    }

    fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let schema = Arc::new(self.stack.schema()?.clone());

        let transformed = self
            .rewriter
            .f_up((node, schema))?
            .update_data(|(expr, _schema)| expr);

        self.stack.up(&transformed.data)?;

        Ok(transformed)
    }
}

#[repr(transparent)]
struct PhysicalExprWithLambda(Arc<dyn PhysicalExpr>);

impl<'a> From<&'a Arc<dyn PhysicalExpr>> for &'a PhysicalExprWithLambda {
    fn from(value: &'a Arc<dyn PhysicalExpr>) -> Self {
        unsafe { &*(value as *const _ as *const PhysicalExprWithLambda) }
    }
}

impl TreeNode for PhysicalExprWithLambda {
    fn apply_children<
        'n,
        F: FnMut(&'n Self) -> Result<datafusion_common::tree_node::TreeNodeRecursion>,
    >(
        &'n self,
        mut f: F,
    ) -> Result<datafusion_common::tree_node::TreeNodeRecursion> {
        if let Some(lambda) = self.0.as_any().downcast_ref::<Lambda>() {
            f(lambda.inner().into())
        } else {
            self.0
                .arc_children()
                .into_iter()
                .apply_until_stop(|p| f(p.into()))
        }
    }

    fn map_children<F: FnMut(Self) -> Result<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> Result<Transformed<Self>> {
        if let Some(lambda) = self.0.as_any().downcast_ref::<Lambda>() {
            f(PhysicalExprWithLambda(Arc::clone(lambda.inner())))
        } else {
            let children = self.0.arc_children();

            if !children.is_empty() {
                let new_children = children
                    .into_iter()
                    .cloned()
                    .map_until_stop_and_collect(|p| {
                        f(PhysicalExprWithLambda(p)).map(|t| t.update_data(|n| n.0))
                    })?;
                // Propagate up `new_children.transformed` and `new_children.tnr`
                // along with the node containing transformed children.
                if new_children.transformed {
                    let arc_self = Arc::clone(&self.0);

                    new_children.map_data(|new_children| {
                        self.0
                            .with_new_arc_children(arc_self, new_children)
                            .map(PhysicalExprWithLambda)
                    })
                } else {
                    Ok(Transformed::new(self, false, new_children.tnr))
                }
            } else {
                Ok(Transformed::no(self))
            }
        }
    }
}

#[repr(transparent)]
struct PhysicalExprWithLambda2(dyn PhysicalExpr);

fn as_with_lambda(value: &Arc<dyn PhysicalExpr>) -> &Arc<PhysicalExprWithLambda2> {
    unsafe { &*(value as *const _ as *const Arc<PhysicalExprWithLambda2>) }
}

fn as_without_lambda(value: &Arc<PhysicalExprWithLambda2>) -> &Arc<dyn PhysicalExpr> {
    unsafe { &*(value as *const _ as *const Arc<dyn PhysicalExpr>) }
}

fn into_with_lambda(value: Arc<dyn PhysicalExpr>) -> Arc<PhysicalExprWithLambda2> {
    unsafe { Arc::from_raw(Arc::into_raw(value) as _) }
}

fn into_without_lambda(value: Arc<PhysicalExprWithLambda2>) -> Arc<dyn PhysicalExpr> {
    // unsafe { Arc::from_raw(Arc::into_raw(value) as _) }
    unsafe { std::mem::transmute(value) }
}

impl DynTreeNode for PhysicalExprWithLambda2 {
    fn arc_children(&self) -> Vec<&Arc<Self>> {
        if let Some(lambda) = self.0.as_any().downcast_ref::<Lambda>() {
            vec![as_with_lambda(lambda.inner())]
        } else {
            self.0
                .arc_children()
                .into_iter()
                .map(as_with_lambda)
                .collect()
        }
    }

    fn with_new_arc_children(
        &self,
        arc_self: Arc<Self>,
        mut new_children: Vec<Arc<Self>>,
    ) -> Result<Arc<Self>> {
        if let Some(lambda) = self.0.as_any().downcast_ref::<Lambda>() {
            if new_children.len() != 1 {
                return exec_err!("Lambda::with_new_arc children called with {} new children instead of 1", new_children.len());
            }

            let new_body = new_children.remove(0);

            if !Arc::ptr_eq(as_with_lambda(lambda.inner()), &new_body) {
                Ok(into_with_lambda(Arc::new(Lambda::new(
                    into_without_lambda(new_body),
                    lambda.args().to_vec(),
                ))))
            } else {
                Ok(arc_self)
            }
        } else {
            self.0
                .with_new_arc_children(
                    into_without_lambda(arc_self),
                    new_children.into_iter().map(into_without_lambda).collect(),
                )
                .map(into_with_lambda)
        }
    }
}


fn stretch_batch<K: ArrowDictionaryKeyType>(
    batch: &RecordBatch,
    indices: &PrimitiveArray<K>,
) -> Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .map(|column| stretch(column, indices))
        .collect::<Result<Vec<_>>>()?;

    let fields = std::iter::zip(batch.schema().fields(), &columns)
        .map(|(field, column)| {
            Field::new(
                field.name(),
                column.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            batch.schema().metadata().clone(),
        )),
        columns,
    )?)
}

fn stretch<K: ArrowDictionaryKeyType>(
    values: &ArrayRef,
    indices: &PrimitiveArray<K>,
) -> Result<ArrayRef> {
    match values.data_type() {
        DataType::Binary
        | DataType::LargeBinary
        | DataType::Utf8
        | DataType::LargeUtf8 => Ok(Arc::new(DictionaryArray::try_new(
            indices.clone(),
            Arc::clone(values),
        )?)),
        DataType::Struct(fields) => {
            let arrays = values
                .as_struct()
                .columns()
                .iter()
                .map(|column| stretch(column, indices))
                .collect::<Result<Vec<_>>>()?;

            let fields = std::iter::zip(fields, &arrays)
                .map(|(field, array)| {
                    Field::new(
                        field.name(),
                        array.data_type().clone(),
                        field.is_nullable(),
                    )
                    .with_metadata(field.metadata().clone())
                })
                .collect();

            Ok(Arc::new(StructArray::try_new(
                fields,
                arrays,
                values.nulls().cloned(),
            )?))
        }
        _ => Ok(take(values, &indices, None)?),
    }
}
