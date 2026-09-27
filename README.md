# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

This project implements a scalable business entity resolution pipeline for the Amazon ML Challenge 2026.

The task is to identify records in two noisy source datasets that correspond to entities in a deduplicated reference dataset.

The system performs:

1. Data normalization
2. Candidate generation / blocking
3. Similarity feature engineering
4. Rule-based entity matching
5. Test-set inference
6. Submission file generation
7. Submission validation

The implementation was designed to work under constrained local compute resources using DuckDB and SQL-based processing.

---

## Problem Statement

The challenge contains three independent sources:

- Source 1 — reference/deduplicated entities
- Source 2 — noisy entity records
- Source 3 — noisy entity records

Source 1 contains the reference entities.

For every Source 1 entity, the objective is to identify all corresponding entities from Source 2 and Source 3.

There are no shared entity IDs across the sources.

A Source 1 entity can have:

- no matching record
- one matching record
- multiple matching records

The task therefore requires both high-precision matching and handling of entities that have no corresponding record.

---

## Dataset

The challenge data contains:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
