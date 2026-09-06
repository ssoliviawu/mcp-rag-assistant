# MCP RAG Evaluation

This directory contains evaluation code for the MCP RAG Assistant.

The first evaluation stage measures the retrieval system before adding
reranking or generation evaluation.

---

## Directory Structure

```text
evaluation/
├── __init__.py
├── baseline.py
├── metrics.py
└── results/
    ├── baseline_retrieval_summary.json
    └── baseline_retrieval_details.json