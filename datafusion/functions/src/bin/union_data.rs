use std::fs::File;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch, UnionBuilder};

use arrow::array::{Int32Array, UnionArray};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{Field, UnionFields};
use arrow::datatypes::{Float64Type, Int32Type, Schema};

use datafusion_common::Result;

fn sparse_union() -> Result<()> {
    let mut builder = UnionBuilder::new_sparse();

    builder.append::<Int32Type>("a", 1)?;
    builder.append::<Float64Type>("b", 3.0)?;
    builder.append::<Int32Type>("a", 4)?;

    let union = builder.build()?;

    write("union_sparse.arrow", union)
}

fn empty_union() -> Result<()> {
    let union = UnionArray::try_new(
        UnionFields::empty(),
        ScalarBuffer::from(vec![]),
        None,
        vec![],
    )?;

    write("union_sparse_empty.arrow", union)?;

    let union = UnionArray::try_new(
        UnionFields::empty(),
        ScalarBuffer::from(vec![]),
        Some(ScalarBuffer::from(vec![])),
        vec![],
    )?;

    write("union_dense_empty.arrow", union)
}

fn dense_union_duplicated_offsets() -> Result<()> {
    let ints = Int32Array::from(vec![1, 2]);

    let fields = [(
        0,
        Arc::new(Field::new("a", ints.data_type().clone(), false)),
    )]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0, 0, 0, 0]),
        Some(ScalarBuffer::from(vec![0, 0, 1, 1])),
        vec![Arc::new(ints)],
    )?;

    write("dense_union_duplicated_offsets.arrow", union)
}

fn dense_union_non_sequential_offsets() -> Result<()> {
    let ints = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);

    let fields = [(
        0,
        Arc::new(Field::new("a", ints.data_type().clone(), false)),
    )]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0, 0, 0, 0]),
        Some(ScalarBuffer::from(vec![0, 2, 4, 6])),
        vec![Arc::new(ints)],
    )?;

    write("dense_union_non_sequential_offsets.arrow", union)
}

fn sparse_union_without_nulls() -> Result<()> {
    let ints = Int32Array::from(vec![1, 2]);
    let bools = BooleanArray::from(vec![true, false]);

    let fields = [
        (
            0,
            Arc::new(Field::new("a", ints.data_type().clone(), false)),
        ),
        (
            1,
            Arc::new(Field::new("b", bools.data_type().clone(), false)),
        ),
    ]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0, 1]),
        None,
        vec![Arc::new(ints), Arc::new(bools)],
    )?;

    write("union_sparse_without_nulls.arrow", union)
}

fn sparse_union_with_nulls() -> Result<()> {
    let ints = Int32Array::try_new(
        vec![1, 2].into(),
        Some(NullBuffer::from(vec![true, false])),
    )?;
    let bools = BooleanArray::new(
        vec![true, false].into(),
        Some(NullBuffer::from(vec![true, false])),
    );

    let fields = [
        (0, Arc::new(Field::new("a", ints.data_type().clone(), true))),
        (
            1,
            Arc::new(Field::new("b", bools.data_type().clone(), true)),
        ),
    ]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0, 1]),
        None,
        vec![Arc::new(ints), Arc::new(bools)],
    )?;

    write("union_sparse_with_nulls.arrow", union)
}

fn dense_union_empty_child() -> Result<()> {
    let ints = Int32Array::from(vec![1, 2]);
    let bools = BooleanArray::new_null(0);

    let fields = [
        (
            0,
            Arc::new(Field::new("a", ints.data_type().clone(), false)),
        ),
        (
            1,
            Arc::new(Field::new("b", bools.data_type().clone(), false)),
        ),
    ]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0]),
        Some(ScalarBuffer::from(vec![0])),
        vec![Arc::new(ints), Arc::new(bools)],
    )?;

    write("union_dense_empty_child.arrow", union)
}

fn dense_union_single_type_exact_size() -> Result<()> {
    let mut builder = UnionBuilder::new_dense();

    builder.append::<Int32Type>("a", 1)?;
    builder.append::<Int32Type>("a", 4)?;

    let union = builder.build()?;

    write("union_dense_single_type_exact_size.arrow", union)
}

fn dense_union_single_type_bigger_child() -> Result<()> {
    let ints = Int32Array::from(vec![1, 2]);

    let fields = [(
        0,
        Arc::new(Field::new("a", ints.data_type().clone(), false)),
    )]
    .into_iter()
    .collect();

    let union = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![0]),
        Some(ScalarBuffer::from(vec![0])),
        vec![Arc::new(ints)],
    )?;

    write("union_dense_bigger_child.arrow", union)
}

fn dense_union_multiple_types() -> Result<()> {
    let mut builder = UnionBuilder::new_dense();

    builder.append::<Int32Type>("a", 1)?;
    builder.append::<Float64Type>("b", 3.0)?;
    builder.append::<Int32Type>("a", 4)?;

    let union = builder.build()?;

    write("union_dense_multiple_types.arrow", union)
}

fn write(path: &'static str, union: UnionArray) -> Result<()> {
    let schema = Schema::new(vec![Field::new(
        "my_union",
        union.data_type().clone(),
        false,
    )]);

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(union)])?;

    let mut file = File::create(path)?;

    let mut writer =
        arrow::ipc::writer::FileWriter::try_new(&mut file, batch.schema_ref())?;

    writer.write(&batch)?;

    Ok(writer.finish()?)
}

fn main() {
    sparse_union().unwrap();
    empty_union().unwrap();
    sparse_union_without_nulls().unwrap();
    sparse_union_with_nulls().unwrap();
    dense_union_empty_child().unwrap();
    dense_union_single_type_exact_size().unwrap();
    dense_union_single_type_bigger_child().unwrap();
    dense_union_multiple_types().unwrap();
}
