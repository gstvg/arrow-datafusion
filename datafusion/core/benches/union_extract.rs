#[macro_use]
extern crate criterion;

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, Int8Array, StringArray, UnionArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field, UnionFields};
use criterion::black_box;
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarUDFImpl};
use datafusion_functions::core::union_extract::UnionExtractFun;
use itertools::repeat_n;

use crate::criterion::Criterion;

fn criterion_benchmark(c: &mut Criterion) {
    // c.benchmark_group("group_name");

    let union_extract = UnionExtractFun::new();

    c.bench_function("sparse union_extract single field", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(vec![1], vec![Field::new("a", DataType::Utf8, false)]),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            None,
            vec![Arc::new(StringArray::from_iter_values(repeat_n("a", 2048)))],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("sparse union_extract empty union", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(vec![1], vec![Field::new("a", DataType::Utf8, false)]),
            ScalarBuffer::from(vec![]),
            None,
            vec![Arc::new(StringArray::new_null(0))],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("sparse union_extract all null", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::Int32, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            None,
            vec![
                Arc::new(StringArray::new_null(2048)),
                Arc::new(Int32Array::from_iter(0..2048)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("sparse union_extract all types match", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::Int32, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            None,
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("a", 2048))),
                Arc::new(Int32Array::new_null(2048)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("sparse union_extract multiples types", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::Int32, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2047).chain([3])),
            None,
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("a", 2048))),
                Arc::new(Int32Array::new_null(2048)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract empty union empty child", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from(vec![]),
            Some(ScalarBuffer::from(vec![])),
            vec![
                Arc::new(StringArray::new_null(0)),
                Arc::new(Int8Array::new_null(0)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract empty union non-empty child", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from(vec![]),
            Some(ScalarBuffer::from(vec![])),
            vec![
                Arc::new(StringArray::from(vec!["a1", "s2"])),
                Arc::new(Int8Array::new_null(0)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract empty child", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from(vec![3, 3]),
            Some(ScalarBuffer::from(vec![0, 1])),
            vec![
                Arc::new(StringArray::new_null(0)),
                Arc::new(Int8Array::new_null(2)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract single field sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(vec![1], vec![Field::new("a", DataType::Int64, false)]),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            Some(ScalarBuffer::from_iter(0..2048)),
            vec![Arc::new(Int64Array::from_iter_values(0..2048))],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function(
        "dense union_extract single field non-sequential offsets",
        |b| {
            let union = UnionArray::try_new(
                UnionFields::new(vec![1], vec![Field::new("a", DataType::Int64, false)]),
                ScalarBuffer::from_iter(repeat_n(1, 2048)),
                Some(ScalarBuffer::from_iter((0..2047).chain([2046]))),
                vec![Arc::new(Int64Array::from_iter_values(0..2048))],
            )
            .unwrap();

            let args = [
                ColumnarValue::Array(Arc::new(union)),
                ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
            ];

            b.iter(|| {
                union_extract.invoke(&args).unwrap();
            })
        },
    );

    c.bench_function("dense union_extract others child empty sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            Some(ScalarBuffer::from_iter(0..2048)),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2048))),
                Arc::new(Int8Array::new_null(0)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });
    
    c.bench_function("dense union_extract others child empty non-sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            Some(ScalarBuffer::from_iter((0..2047).chain([2046]))),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2048))),
                Arc::new(Int8Array::new_null(0)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract all types match sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            Some(ScalarBuffer::from_iter(0..2048)),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2048))),
                Arc::new(Int8Array::new_null(1)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });
    
    c.bench_function("dense union_extract all types match non-sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2048)),
            Some(ScalarBuffer::from_iter((0..2047).chain(Some(2046)))),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2048))),
                Arc::new(Int8Array::new_null(1)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });

    c.bench_function("dense union_extract multiples types sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2043).chain(Some(3))),
            Some(ScalarBuffer::from_iter((0..2043).chain(Some(0)))),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2043))),
                Arc::new(Int8Array::new_null(1)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            black_box(union_extract.invoke(black_box(&args)).unwrap());
        })
    });
    
    c.bench_function("dense union_extract multiples types non-sequential offsets", |b| {
        let union = UnionArray::try_new(
            UnionFields::new(
                vec![1, 3],
                vec![
                    Field::new("a", DataType::Utf8, false),
                    Field::new("b", DataType::UInt8, false),
                ],
            ),
            ScalarBuffer::from_iter(repeat_n(1, 2047).chain([3])),
            Some(ScalarBuffer::from_iter((0..2046).chain([2045, 0]))),
            vec![
                Arc::new(StringArray::from_iter_values(repeat_n("str", 2048))),
                Arc::new(Int8Array::new_null(1)),
            ],
        )
        .unwrap();

        let args = [
            ColumnarValue::Array(Arc::new(union)),
            ColumnarValue::Scalar(ScalarValue::new_utf8("a")),
        ];

        b.iter(|| {
            union_extract.invoke(&args).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
