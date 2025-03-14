
pub fn list_reduce<T: ArrowPrimitiveType>(
    list: GenericListArray<T::Native>,
    lambda: &dyn PhysicalExpr,
    init: ColumnarValue,
) -> Result<ArrayRef>
where
    T::Native: OffsetSizeTrait,
{
    let values = list.values();

    let mut pairs = match init {
        ColumnarValue::Array(_) => (0..list.len()).map(|i| (0, i)).collect(),
        ColumnarValue::Scalar(_) => vec![(0, 0); list.len()],
    };

    let init = init.into_array(list.len())?;

    let schema = Schema::new(vec![
        Field::new("acc", init.data_type().clone(), init.is_nullable()),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", T::DATA_TYPE, false),
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("acc", init.data_type().clone(), lambda.nullable(&schema)?),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", T::DATA_TYPE, false),
    ]));

    let mut last_indices = Arc::new(PrimitiveArray::<T>::from_iter_values(
        (0..list.len()).map(T::Native::usize_as),
    )) as ArrayRef;

    let mut lengths = length(&list)?;
    let mut offsets = Arc::new(PrimitiveArray::<T>::new(
        list.offsets().clone().into(),
        None,
    )) as ArrayRef;
    let mut partials = vec![init];

    for current_len in 0.. {
        let mask = gt(
            &lengths,
            &PrimitiveArray::<T>::new_scalar(T::Native::usize_as(current_len)),
        )?;

        lengths = filter(&lengths, &mask)?;
        offsets = filter(&offsets, &mask)?;
        last_indices = filter(&last_indices, &mask)?;

        if true {
            for (i, last_indice) in
                last_indices.as_primitive::<T>().values().iter().enumerate()
            {
                // todo: the array_index is already set: it equals to the list length
                pairs[last_indice.as_usize()].0 = current_len;
                pairs[last_indice.as_usize()].1 = i;
            }
        } else {
            for (i, length) in lengths.iter().enumerate() {
                pairs[i].0 += (length > current_len) as usize;
            }
        }

        if lengths.is_empty() {
            break;
        }

        let indices = add(
            &offsets,
            &PrimitiveArray::<T>::new_scalar(T::Native::usize_as(1)),
        )?;

        let acc = filter(partials.last().unwrap(), &mask)?;
        let current_values = take(values, &indices, None)?;
        let index = Arc::new(PrimitiveArray::<T>::from_value(
            T::Native::usize_as(current_len),
            indices.len(),
        ));

        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![acc, current_values, index])?;

        let partial = lambda.evaluate(&batch)?;

        partials.push(partial.into_array(indices.len())?);
    }

    Ok(interleave(
        &partials.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
        &pairs,
    )?)
}

pub fn list_reduce2<T: ArrowPrimitiveType>(
    list: GenericListArray<T::Native>,
    lambda: &dyn PhysicalExpr,
    init: ColumnarValue,
) -> Result<ArrayRef>
where
    T::Native: OffsetSizeTrait,
{
    let values = list.values();

    let mut reduced = init.into_array(list.len())?;
    let mut last_step = reduced.clone();

    let schema = Schema::new(vec![
        Field::new("acc", reduced.data_type().clone(), reduced.is_nullable()),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", T::DATA_TYPE, false),
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "acc",
            reduced.data_type().clone(),
            lambda.nullable(&schema)?,
        ),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", T::DATA_TYPE, false),
    ]));

    let lengths = length(&list)?;
    let offsets = Arc::new(PrimitiveArray::<T>::new(
        list.offsets().clone().into(),
        None,
    )) as ArrayRef;

    let mut last_indices = offsets.clone();

    for current_len in 0.. {
        let mask = gt(
            &lengths,
            &PrimitiveArray::<T>::new_scalar(T::Native::usize_as(current_len)),
        )?;

        if mask.true_count() == 0 {
            break;
        }

        let (acc, indices) = if mask.true_count() == last_step.len() {
            (last_step.clone(), last_indices.clone())
        } else {
            let indices = filter(&offsets, &mask)?;

            (filter(&reduced, &mask)?, indices)
        };
        let current_values = take(
            &values.slice(current_len, values.len() - current_len),
            &indices,
            None,
        )?;
        let index = Arc::new(PrimitiveArray::<T>::from_value(
            T::Native::usize_as(current_len),
            indices.len(),
        ));

        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![acc, current_values, index])?;

        let step = lambda.evaluate(&batch)?.into_array(indices.len())?;

        if step.len() == list.len() {
            reduced = step;
            last_step = reduced.clone();
        } else if step.len() == last_step.len() {
            last_step = step;
        } else {
            let mut right_index = 0;

            let indices = mask
                .values()
                .iter()
                .enumerate()
                .map(|(left_index, v)| {
                    if v {
                        right_index += 1;

                        (1, right_index)
                    } else {
                        (0, left_index)
                    }
                })
                .collect::<Vec<_>>();

            reduced = interleave(&[&reduced, &step], &indices)?;
        }

        last_indices = indices;
    }

    // if false {
    // interleave()
    // }

    let nulls = NullBuffer::union(reduced.nulls(), list.nulls());

    let data = unsafe {
        reduced
            .into_data()
            .into_builder()
            .nulls(nulls)
            .build_unchecked()
    };

    Ok(make_array(data))
}

pub fn fixed_size_list_reduce(
    list: FixedSizeListArray,
    lambda: &dyn PhysicalExpr,
    init: ColumnarValue,
) -> Result<ArrayRef> {
    let values = list.values();

    let mut reduced = init.into_array(list.len())?;

    const U8_MAX: u64 = u8::MAX as u64;
    const U16_MAX: u64 = u16::MAX as u64;
    const U32_MAX: u64 = u32::MAX as u64;

    let index_type = match list.value_length() as u64 {
        ..=U8_MAX => DataType::UInt8,
        U8_MAX..=U16_MAX => DataType::UInt16,
        U16_MAX..=U32_MAX => DataType::UInt32,
        U32_MAX..=u64::MAX => DataType::UInt64,
    };

    let init_schema = Schema::new(vec![
        Field::new("acc", reduced.data_type().clone(), reduced.is_nullable()),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", DataType::Int32, false),
    ]);

    // list_reduce([NULL], NULL, v -> coalesce(v, 0))
    let lambda_schema = Arc::new(Schema::new(vec![
        Field::new(
            "acc",
            reduced.data_type().clone(),
            lambda.nullable(&init_schema)?,
        ),
        Field::new("v", values.data_type().clone(), values.is_nullable()),
        Field::new("i", DataType::Int32, false),
    ]));

    let indices = Arc::new(
        (0..list.len() as i32)
            .map(|i| i * list.value_length())
            .collect::<Int32Array>(),
    ) as ArrayRef;

    for i in 0..list.value_length() {
        let current_values = take(
            &values.slice(i as usize, values.len() - i as usize),
            &indices,
            None,
        )?;

        let schema = if i == 0 {
            Arc::new(init_schema)
        } else {
            Arc::clone(&lambda_schema)
        };

        reduced = lambda
            .evaluate(&RecordBatch::try_new(
                schema,
                vec![
                    reduced,
                    current_values,
                    Arc::new(Int32Array::from_value(i, list.len())),
                ],
            )?)?
            .into_array(list.len())?;
    }

    let nulls = NullBuffer::union(reduced.nulls(), list.nulls());

    let data = unsafe {
        reduced
            .into_data()
            .into_builder()
            .nulls(nulls)
            .build_unchecked()
    };

    Ok(make_array(data))
}

fn interleave3(values: &[&dyn Array], i: &[usize], offsets: &[usize], j: &[&[usize]]) {
    for (i, offset) in std::iter::zip(i, offsets) {
        let x = j[*i][*offset];

        values[*i][x]
    }
}

fn optimize_union(expr: &Expr, schema: &dyn ExprSchema) {
    if let Expr::Case(Case {
        expr: Some(expr),
        when_then_expr,
        else_expr,
    }) = expr
    {
        if let Expr::ScalarFunction(union_tag) = expr {
            if union_tag.name() == "union_tag" {
                when_then_expr
                    .iter()
                    .map(|(when, _then)| {
                        if let Expr::Literal(ScalarValue::Utf8(Some(tag))) = when {
                            Ok(tag)
                        } else {
                            Err(())
                        }
                    })
                    .collect::<Result<Vec<_>>>();
            }
        }
    }
}
