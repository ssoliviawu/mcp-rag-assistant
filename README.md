# MCP RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for answering questions about the **MCP Python SDK documentation**.

This project was developed as a final project for the [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

The system retrieves relevant MCP Python SDK documentation using vector search and generates grounded answers with an LLM.

This project is not just about building a RAG Assistant—it is about understanding how to improve one. I started with a standard vector retrieval pipeline and systematically evaluated multiple optimization strategies, including hybrid search, reranking, deduplication, and document representation.

The most important finding was that more sophisticated does not always mean better. Some popular techniques increased complexity and latency without providing consistent improvements. Through error analysis and controlled experiments, I found that improving document representation and handling section quality was more effective for this MCP documentation corpus.

The result is a simpler, more explainable, and evidence-driven RAG pipeline, where design decisions are based on measured results rather than assumptions.

---

## 1. Problem Description

The MCP Python SDK documentation contains a large amount of technical information distributed across different documentation pages and sections.

A user asking a question such as:

> How do I create an MCP tool in Python?

should receive an answer based on the official MCP Python SDK documentation rather than a generic answer generated only from the language model's internal knowledge.

The goal of this project is therefore to build a RAG-based assistant that:

* retrieves relevant MCP Python SDK documentation;
* reduces duplicate results from the same source;
* provides the retrieved documentation to an LLM as context;
* generates concise, technically accurate answers;
* provides the source URLs used for the answer;
* evaluates retrieval and generation quality on a dedicated evaluation dataset.

---

## 2. System Architecture

```text
                         MCP Python SDK Documentation
                                      │
                                      ▼
                             Ingestion Pipeline
                                      │
                                      ▼
                                   Chunking
                                      │
                                      ▼
                               BGE Embeddings
                                      │
                                      ▼
                           PostgreSQL + pgvector
                                      │
                                      │
User Question ───────────────► Vector Search
                                      │
                                      ▼
                         Candidate Limit = 100
                                      │
                                      ▼
                            Source-level Dedup
                                      │
                                      ▼
                       Max 1 Chunk per Source
                                      │
                                      ▼
                                Top Results
                                      │
                                      ▼
                           Context Construction
                                      │
                                      ▼
                             agnes-2.5-flash
                                      │
                                      ▼
                              Generated Answer
                                      │
                                      ▼
                              Source References
```

---

## 3. Technology Stack

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| Language            | Python 3.12                       |
| Embedding model     | BGE embedding model               |
| Embedding dimension | 384                               |
| Vector database     | PostgreSQL + pgvector             |
| Retrieval           | Vector similarity search          |
| LLM                 | `agnes-2.5-flash`                 |
| LLM API             | OpenAI-compatible API             |
| Web UI              | Streamlit                         |
| Package management  | uv                                |
| Database driver     | psycopg                           |
| Evaluation          | Custom Python evaluation pipeline |
| Testing             | pytest                            |
| CI                  | GitHub Actions                    |

---

## 4. Project Structure

```text
MCP RAG Assistant/
│
├── app/
│   ├── rag.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── mcp-python-sdk/
│   └── chunks.jsonl
│
├── embedding/
│   └── model.py
│
├── ingestion/
│   ├── discover.py
│   ├── pipeline.py
│   └── embed.py
│
├── retrieval/
│   ├── search.py
│   └── reranker.py
│
├── evaluation/
│   ├── baseline.py
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── e2_final_comparison_generation.py
│   ├── e25_section_rule_combinations.py
│   ├── e27_*.py
│   ├── e28_*.py
│   ├── e29_robustness.py
│   ├── e30_*.py
│   └── results/
│
├── monitoring/
│   ├── __init__.py
│   └── logger.py
│
├── rag/
│   └── legacy application modules
│
├── db/
│   └── schema.sql
│
├── tests/
│
├── mcp_rag_eval_dataset.jsonl
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 5. Retrieval Pipeline

The final retrieval pipeline uses vector search with source-level deduplication.

```text
Question
   │
   ▼
BGE Embedding
   │
   ▼
Vector Search
candidate_limit = 100
   │
   ▼
Source Deduplication
max_chunks_per_source = 1
   │
   ▼
Top 10 Results
```

The final configuration is:

```python
CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1
```

The system retrieves a larger candidate set and then limits the number of chunks originating from the same source. This improves result diversity while preserving strong retrieval quality.

---

# 6. Retrieval Experiments

Several retrieval approaches were evaluated.

| Experiment | Approach                  |      Hit@1 |   Hit@10 |        MRR |  Avg Latency |
| ---------- | ------------------------- | ---------: | -------: | ---------: | -----------: |
| E0         | Vector                    |     45.83% |   93.75% |     0.5948 |    475.86 ms |
| E1         | Vector, candidate=100     |     45.83% |   93.75% |     0.5948 |     73.25 ms |
| **E2**     | **Vector + source dedup** | **45.83%** | **100%** | **0.6368** | **73.27 ms** |
| E3         | Vector + reranker         |     35.42% |   93.75% |     0.5267 |     ~17.57 s |
| E3 + dedup | Vector + reranker + dedup |     35.42% |   97.92% |     0.5634 |     ~17.79 s |
| E4         | Hybrid                    |     35.42% |   91.67% |     0.5140 |    308.90 ms |
| E5         | Hybrid + dedup            |     35.42% |     100% |     0.5561 |    286.29 ms |

### Final retrieval choice

E2 was selected as the final retrieval strategy because it provided the best balance between:

* retrieval quality;
* ranking quality;
* result diversity;
* latency;
* implementation simplicity.

The reranker was deliberately not used in the final system.

The ONNX reranker significantly increased latency while reducing overall retrieval quality.

---

# 7. Final Retrieval Evaluation

The final E2 pipeline was evaluated on 48 non-specification questions.

Final retrieval results:

| Metric                    |       Result |
| ------------------------- | -----------: |
| Hit@1                     |   **66.67%** |
| Hit@3                     |   **91.67%** |
| Hit@5                     |     **100%** |
| Hit@10                    |     **100%** |
| Recall@10                 |   **88.54%** |
| MRR                       |   **0.7976** |
| Average retrieval latency | **20.88 ms** |
| P50 latency               | **19.18 ms** |
| P95 latency               | **36.13 ms** |
| Retrieval misses          |   **0 / 48** |

The final retrieval pipeline successfully retrieved at least one relevant document for all 48 evaluation questions.

---

# 8. Generation

The retrieved documents are converted into a structured context:

```text
[Document 1]

Title: ...

Section: ...

URL: ...

Content:

...

[Document 2]

Title: ...

Section: ...

URL: ...

Content:

...
```

The context is limited to 12,000 characters.

The final generation model is:

```text
agnes-2.5-flash
```

The generation prompt instructs the model to:

1. use retrieved documentation as the primary source;
2. avoid unsupported claims;
3. acknowledge when the documentation is insufficient;
4. provide concise and technically accurate answers;
5. reference relevant documentation URLs.

---

# 9. End-to-End Evaluation

The final E2 retrieval + generation pipeline was evaluated using the project evaluation dataset.

| Metric               |              Result |
| -------------------- | ------------------: |
| Correctness          |          **86.46%** |
| Faithfulness         |          **95.84%** |
| Citation Correctness |          **95.84%** |
| Overall              |          **92.74%** |
| PASS                 | **36 / 48 (75.0%)** |
| PARTIAL              | **10 / 48 (20.8%)** |
| FAIL                 |   **2 / 48 (4.2%)** |
| Retrieval hits       |         **48 / 48** |

The results indicate that the primary remaining limitations are not document retrieval failures.

The remaining errors are mainly related to:

* incomplete answers;
* failure to use all relevant retrieved evidence;
* missing citations in some answers;
* occasional interpretation errors.

This suggests that future improvements should focus primarily on **generation and evidence utilization**, rather than further retrieval optimization.

---

# 10. Reranker Evaluation

A cross-encoder reranker based on:

```text
ms-marco-MiniLM-L-6-v2
```

was also evaluated.

The reranker was rejected for the final pipeline.

One representative example:

```text
Question: Q013

Vector rank:       5

Reranker rank:    64

Rank change:      -59
```

The reranker also increased retrieval latency from approximately tens of milliseconds to approximately 18 seconds.

Therefore:

```text
Final system:

Vector Search + Dedup
```

instead of:

```text
Vector Search + Reranker
```

This is an intentional design decision based on experimental evidence.

---

# 11. Web Interface

The project includes a Streamlit-based web interface.

Users can:

* enter an MCP Python SDK question;
* receive an LLM-generated answer;
* view total latency;
* see the number of retrieved documents;
* inspect retrieved documentation sources.

Streamlit is included as a project dependency and is installed automatically when using `uv sync`.

Run:

```bash
uv run streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# 12. Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_compatible_base_url
OPENAI_MODEL=agnes-2.5-flash
```

Do **not** commit `.env` to GitHub.

The repository should contain a safe example file such as:

```text
.env.example
```

with:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=agnes-2.5-flash
```

---

# 13. Installation

## Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd MCP-RAG-Assistant
```

## Install dependencies

The project uses `uv` for dependency management.

```bash
uv sync
```

This installs all project dependencies, including Streamlit.

Activate the environment if desired:

```powershell
.venv\Scripts\activate
```

The project uses:

```text
Python 3.12
pyproject.toml
uv.lock
```

for reproducible dependency management.

---

# 14. Database Setup

The project uses PostgreSQL with pgvector.

The database schema is defined in:

```text
db/schema.sql
```

The vector column uses:

```text
vector(384)
```

because the BGE embedding model produces 384-dimensional embeddings.

The project also includes:

```text
docker-compose.yml
```

for running the PostgreSQL + pgvector service.

---

# 15. Ingestion Pipeline

The ingestion process consists of:

```text
Discover documentation
        ↓
Download / collect pages
        ↓
Chunk documents
        ↓
Generate embeddings
        ↓
Store chunks + embeddings
```

The ingestion modules are located under:

```text
ingestion/
```

Examples:

```bash
python -m ingestion.discover

python -m ingestion.pipeline

python -m ingestion.embed
```

---

# 16. Running Evaluation

The evaluation dataset is:

```text
mcp_rag_eval_dataset.jsonl
```

The dataset contains 50 questions, with two specification questions excluded from the final retrieval/generation comparison, resulting in 48 evaluation questions.

The evaluation scripts are located under:

```text
evaluation/
```

The final E2 generation comparison can be run with:

```bash
python -m evaluation.e2_final_comparison_generation
```

Results are written to:

```text
evaluation/results/
```

---

# 17. Monitoring

The application includes lightweight request monitoring.

The monitoring implementation is located under:

```text
monitoring/
```

The logger records information including:

* timestamp;
* question;
* success/failure status;
* number of retrieved documents;
* model name;
* generation latency;
* total request latency;
* error information when applicable.

Runtime metrics are written to:

```text
monitoring/metrics.jsonl
```

The metrics file is excluded from Git because it may contain user questions and runtime data.

The current monitoring implementation is intentionally lightweight and file-based.

Future improvements could include:

* user feedback collection;
* monitoring dashboards;
* retrieval-quality tracking;
* answer-quality tracking;
* latency visualization;
* failure-case analysis.

---

# 18. Reproducibility

The project uses:

* Python 3.12;
* `uv`;
* `pyproject.toml`;
* `uv.lock`;
* a fixed evaluation dataset;
* explicit retrieval parameters;
* explicit model configuration.

The main final configuration is:

```text
Embedding:

BGE

Retrieval:

Vector Search

Candidate limit:

100

Source dedup:

Enabled

Maximum chunks per source:

1

Top K:

10

Reranker:

Disabled

Hybrid retrieval:

Disabled

LLM:

agnes-2.5-flash
```

The experimental results are stored under:

```text
evaluation/results/
```

so that the final retrieval decision can be traced back to the experiments.

---

# 19. Testing and Continuous Integration

The project includes automated tests using pytest.

Run tests locally with:

```bash
uv run pytest -q
```

Python compilation checks can be run with:

```bash
uv run python -m compileall app retrieval embedding ingestion evaluation monitoring tests
```

The repository also includes a GitHub Actions CI workflow:

```text
.github/
└── workflows/
    └── ci.yml
```

The CI workflow:

1. checks out the repository;
2. sets up Python 3.12;
3. installs `uv`;
4. installs locked project dependencies;
5. compiles project Python files;
6. runs the test suite.

This helps detect broken code and dependency issues when changes are pushed to the repository or submitted through pull requests.

---

# 20. Project Development Process

The project followed an experiment-driven development process.

```text
Initial Vector Retrieval
        ↓
Candidate Size Experiments
        ↓
Source Deduplication
        ↓
Reranker Evaluation
        ↓
Hybrid Retrieval Evaluation
        ↓
Section / Representation Experiments
        ↓
Robustness Testing
        ↓
Final E2 Selection
        ↓
End-to-End Evaluation
        ↓
Streamlit Application
        ↓
Monitoring
        ↓
Testing / CI
```

Rather than selecting additional components simply because they are commonly used in RAG systems, retrieval improvements were kept only when they demonstrated measurable benefits on the evaluation dataset.

For example, the reranker and hybrid retrieval approaches were evaluated experimentally but were not included in the final system because they did not provide sufficient improvement relative to their additional complexity and latency.

---

# 21. Future Improvements

The current evaluation shows that retrieval is no longer the primary bottleneck.

Potential future improvements include:

### Generation completeness

Improve the generation prompt so that the model systematically covers all important evidence found in the retrieved documents.

### Citation behavior

Improve citation generation so that answers consistently reference the relevant retrieved sources.

### Query rewriting

A controlled query-rewriting experiment can be used to determine whether rewriting improves difficult questions without increasing latency excessively.

### Monitoring dashboard

Extend the current lightweight logging system with a monitoring dashboard for visualizing:

* request volume;
* latency;
* retrieval statistics;
* generation statistics;
* error rates;
* user feedback.

---

# 22. Conclusion

The final MCP RAG Assistant uses a lightweight vector retrieval pipeline combined with source-level deduplication and an LLM generation layer.

The final retrieval configuration achieved:

```text
Hit@10: 100%

MRR:     0.7976

P95:     36.13 ms

Misses:  0 / 48
```

The end-to-end system achieved:

```text
Correctness:          86.46%

Faithfulness:         95.84%

Citation Correctness: 95.84%

Overall:              92.74%
```

The experiments also demonstrated that more complex retrieval components do not automatically produce better results. In particular, the evaluated reranker substantially increased latency and did not improve the final retrieval quality.

The project therefore prioritizes:

* measured performance;
* simplicity;
* reproducibility;
* evidence-grounded generation;
* experimental validation.

---

## License

This project is intended for educational and research purposes as part of the DataTalksClub LLM Zoomcamp final project.
