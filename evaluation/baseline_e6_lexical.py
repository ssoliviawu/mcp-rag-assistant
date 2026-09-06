import json
import re
import time
from pathlib import Path

from evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_recall_at_k,
)


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

SUMMARY_PATH = (
    RESULTS_DIR / "e6_lexical_alpha_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR / "e6_lexical_alpha_details.json"
)


# ============================================================
# Evaluation configuration
# ============================================================

TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1


# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    {
        "id": "E6_02",
        "name": "vector_lexical_alpha_002",
        "description": (
            "Vector + lexical boost + source diversity "
            "(alpha=0.02)"
        ),
        "mode": "vector_lexical_boost",
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
        "alpha": 0.02,
    },
    {
        "id": "E6_05",
        "name": "vector_lexical_alpha_005",
        "description": (
            "Vector + lexical boost + source diversity "
            "(alpha=0.05)"
        ),
        "mode": "vector_lexical_boost",
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
        "alpha": 0.05,
    },
    {
        "id": "E6_10",
        "name": "vector_lexical_alpha_010",
        "description": (
            "Vector + lexical boost + source diversity "
            "(alpha=0.10)"
        ),
        "mode": "vector_lexical_boost",
        "candidate_limit": CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
        "alpha": 0.10,
    },
]


# ============================================================
# Dataset
# ============================================================

def load_dataset(path: Path) -> list[dict]:
    """Load evaluation dataset from JSONL."""

    dataset = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            dataset.append(
                json.loads(line)
            )

    return dataset


# ============================================================
# Normalization
# ============================================================

def normalize_source(source: str) -> str:
    """
    Normalize source identifiers for evaluation comparison.

    URLs are normalized to exactly one trailing slash.
    """

    return (
        source.strip().rstrip("/") + "/"
    )


# ============================================================
# Tokenization
# ============================================================

def tokenize(text: str) -> set[str]:
    """
    Tokenize English technical text.

    Keeps technical identifiers such as:

        _meta
        MCPServer
        list_tools
        call_tool
    """

    if not text:
        return set()

    return {
        token.lower()
        for token in re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            text,
        )
        if len(token) >= 2
    }


# ============================================================
# Lexical score
# ============================================================

def lexical_score(
    query: str,
    content: str,
) -> float:
    """
    Calculate query-token coverage in chunk content.

    Example:

        query tokens = {mcp, tool, optional, parameter}

        matched = 3

        lexical score = 3 / 4 = 0.75
    """

    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    content_lower = content.lower()

    matched = sum(
        1
        for token in query_tokens
        if token in content_lower
    )

    return matched / len(query_tokens)


# ============================================================
# Retrieval result normalization
# ============================================================

def extract_sources(results) -> list[str]:
    """
    Extract source URLs from retrieval results.
    """

    sources = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        source = (
            result.get("url")
            or result.get("source")
        )

        if source:

            sources.append(
                normalize_source(source)
            )

    return sources


# ============================================================
# Retrieval
# ============================================================

def run_search(
    query: str,
    experiment: dict,
):
    """
    Run one lexical boost experiment.

    Pipeline:

        Vector Top100
            ↓
        lexical boost
            ↓
        source diversity
            ↓
        Top10
    """

    from retrieval.search import vector_search

    candidate_limit = (
        experiment["candidate_limit"]
    )

    top_k = experiment["top_k"]

    alpha = experiment["alpha"]

    # --------------------------------------------------------
    # Vector candidate retrieval
    # --------------------------------------------------------

    candidates = vector_search(
        query=query,
        limit=candidate_limit,
    )

    if not candidates:
        return []

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []

    for result in candidates:

        similarity = float(
            result.get(
                "similarity",
                0.0,
            )
        )

        content = result.get(
            "content",
            "",
        )

        lexical = lexical_score(
            query=query,
            content=content,
        )

        final_score = (
            similarity
            + alpha * lexical
        )

        scored.append(
            {
                "result": result,
                "similarity": similarity,
                "lexical_score": lexical,
                "final_score": final_score,
            }
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Source diversity
    # --------------------------------------------------------

    results = []

    source_counts = {}

    for item in scored:

        result = item["result"]

        source = (
            result.get("url")
            or result.get("source")
            or result.get("id")
        )

        if (
            source_counts.get(
                source,
                0,
            )
            >= MAX_CHUNKS_PER_SOURCE
        ):
            continue

        results.append(result)

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            )
            + 1
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# Evaluate one question
# ============================================================

def evaluate_question(
    item: dict,
    experiment: dict,
) -> dict:

    question_id = item["id"]

    question = item["question"]

    relevant_sources = [
        normalize_source(source)
        for source in item["relevant_sources"]
    ]

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    start = time.perf_counter()

    results = run_search(
        query=question,
        experiment=experiment,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # --------------------------------------------------------
    # Extract retrieved sources
    # --------------------------------------------------------

    retrieved_sources = extract_sources(
        results
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "hit@1": calculate_hit_at_k(
            retrieved_sources,
            relevant_sources,
            1,
        ),

        "hit@3": calculate_hit_at_k(
            retrieved_sources,
            relevant_sources,
            3,
        ),

        "hit@5": calculate_hit_at_k(
            retrieved_sources,
            relevant_sources,
            5,
        ),

        "hit@10": calculate_hit_at_k(
            retrieved_sources,
            relevant_sources,
            10,
        ),

        "recall@5": calculate_recall_at_k(
            retrieved_sources,
            relevant_sources,
            5,
        ),

        "recall@10": calculate_recall_at_k(
            retrieved_sources,
            relevant_sources,
            10,
        ),

        "mrr": calculate_mrr(
            retrieved_sources,
            relevant_sources,
        ),
    }

    # --------------------------------------------------------
    # Keep ranking details
    # --------------------------------------------------------

    retrieved_details = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        retrieved_details.append(
            {
                "rank": rank,
                "id": result.get("id"),
                "title": result.get("title"),
                "section": result.get("section"),
                "url": result.get("url"),
                "similarity": result.get(
                    "similarity"
                ),
                "lexical_score": lexical_score(
                    question,
                    result.get(
                        "content",
                        "",
                    ),
                ),
            }
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "id": question_id,

        "question": question,

        "experiment": experiment["id"],

        "mode": experiment["name"],

        "alpha": experiment["alpha"],

        "relevant_sources": relevant_sources,

        "retrieved_sources": retrieved_sources,

        "retrieved_details": retrieved_details,

        "latency_ms": round(
            latency_ms,
            2,
        ),

        "metrics": metrics,
    }


# ============================================================
# Average
# ============================================================

def average(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# Percentile
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * percentile_value
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * weight
    )


# ============================================================
# Calculate summary
# ============================================================

def calculate_summary(
    details: list[dict],
) -> dict:

    latencies = [
        x["latency_ms"]
        for x in details
    ]

    return {

        "num_questions": len(
            details
        ),

        "hit@1": average(
            [
                x["metrics"]["hit@1"]
                for x in details
            ]
        ),

        "hit@3": average(
            [
                x["metrics"]["hit@3"]
                for x in details
            ]
        ),

        "hit@5": average(
            [
                x["metrics"]["hit@5"]
                for x in details
            ]
        ),

        "hit@10": average(
            [
                x["metrics"]["hit@10"]
                for x in details
            ]
        ),

        "recall@5": average(
            [
                x["metrics"]["recall@5"]
                for x in details
            ]
        ),

        "recall@10": average(
            [
                x["metrics"]["recall@10"]
                for x in details
            ]
        ),

        "mrr": average(
            [
                x["metrics"]["mrr"]
                for x in details
            ]
        ),

        "avg_latency_ms": average(
            latencies
        ),

        "median_latency_ms": percentile(
            latencies,
            0.50,
        ),

        "p95_latency_ms": percentile(
            latencies,
            0.95,
        ),
    }


# ============================================================
# Evaluate one experiment
# ============================================================

def evaluate_experiment(
    dataset: list[dict],
    experiment: dict,
) -> tuple[list[dict], dict]:

    print()
    print("=" * 80)

    print(
        f"{experiment['id']} - "
        f"{experiment['name'].upper()}"
    )

    print("=" * 80)

    print(
        f"Description: "
        f"{experiment['description']}"
    )

    print(
        f"Candidate limit: "
        f"{experiment['candidate_limit']}"
    )

    print(
        f"Top-K: "
        f"{experiment['top_k']}"
    )

    print(
        f"Alpha: "
        f"{experiment['alpha']}"
    )

    print(
        f"Max chunks per source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print()

    details = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        result = evaluate_question(
            item=item,
            experiment=experiment,
        )

        details.append(
            result
        )

        print(
            f"[{index:03d}/{len(dataset):03d}] "
            f"{item['id']} "
            f"Hit@1="
            f"{result['metrics']['hit@1']:.0f} "
            f"Hit@5="
            f"{result['metrics']['hit@5']:.0f} "
            f"MRR="
            f"{result['metrics']['mrr']:.3f} "
            f"{result['latency_ms']:.1f}ms"
        )

    summary = calculate_summary(
        details
    )

    return details, summary


# ============================================================
# Print summary
# ============================================================

def print_summary(
    summaries: dict[str, dict],
):

    print()
    print("=" * 110)
    print("LEXICAL BOOST ALPHA COMPARISON")
    print("=" * 110)

    print()

    header = (
        f"{'Exp':<8}"
        f"{'Alpha':>8}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>11}"
        f"{'Recall@10':>13}"
        f"{'MRR':>10}"
        f"{'Avg ms':>12}"
        f"{'Median ms':>13}"
        f"{'P95 ms':>12}"
    )

    print(header)

    print("-" * len(header))

    for experiment in EXPERIMENTS:

        experiment_id = experiment["id"]

        if experiment_id not in summaries:
            continue

        summary = summaries[
            experiment_id
        ]

        print(
            f"{experiment_id:<8}"
            f"{experiment['alpha']:>8.2f}"
            f"{summary['hit@1']:>10.4f}"
            f"{summary['hit@3']:>10.4f}"
            f"{summary['hit@5']:>10.4f}"
            f"{summary['hit@10']:>11.4f}"
            f"{summary['recall@10']:>13.4f}"
            f"{summary['mrr']:>10.4f}"
            f"{summary['avg_latency_ms']:>12.2f}"
            f"{summary['median_latency_ms']:>13.2f}"
            f"{summary['p95_latency_ms']:>12.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    dataset = load_dataset(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Separate RAG questions from specification questions
    # --------------------------------------------------------

    total_questions = len(
        dataset
    )

    excluded_spec_questions = [
        item["id"]
        for item in dataset
        if item.get("domain") == "spec"
    ]

    rag_dataset = [
        item
        for item in dataset
        if item.get(
            "domain",
            "rag",
        ) != "spec"
    ]

    evaluated_questions = len(
        rag_dataset
    )

    print()
    print("=" * 80)
    print("MCP RAG LEXICAL BOOST EXPERIMENT")
    print("=" * 80)

    print(
        f"Dataset: "
        f"{DATASET_PATH}"
    )

    print(
        f"Total questions: "
        f"{total_questions}"
    )

    print(
        f"RAG questions: "
        f"{evaluated_questions}"
    )

    print(
        f"Excluded spec questions: "
        f"{len(excluded_spec_questions)} "
        f"{excluded_spec_questions}"
    )

    print(
        f"Candidate limit: "
        f"{CANDIDATE_LIMIT}"
    )

    print(
        f"Top-K: "
        f"{TOP_K}"
    )

    print(
        f"Max chunks per source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print()

    print(
        "EXPERIMENTS TO RUN:",
        [
            experiment["id"]
            for experiment in EXPERIMENTS
        ],
    )

    all_details = {}

    all_summaries = {}

    # --------------------------------------------------------
    # Run alpha experiments
    # --------------------------------------------------------

    for experiment in EXPERIMENTS:

        details, summary = evaluate_experiment(
            dataset=rag_dataset,
            experiment=experiment,
        )

        experiment_id = experiment["id"]

        all_details[
            experiment_id
        ] = details

        all_summaries[
            experiment_id
        ] = summary

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_output = {

        "dataset": DATASET_PATH.name,

        "total_questions": total_questions,

        "evaluated_questions": evaluated_questions,

        "excluded_spec_questions": len(
            excluded_spec_questions
        ),

        "excluded_spec_question_ids": (
            excluded_spec_questions
        ),

        "experiment_config": {

            "method": (
                "vector + lexical boost "
                "+ source diversity"
            ),

            "candidate_limit": (
                CANDIDATE_LIMIT
            ),

            "top_k": TOP_K,

            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),

            "alphas": [
                experiment["alpha"]
                for experiment in EXPERIMENTS
            ],
        },

        "experiments": EXPERIMENTS,

        "results": all_summaries,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save details
    # --------------------------------------------------------

    details_output = {

        "dataset": DATASET_PATH.name,

        "candidate_limit": (
            CANDIDATE_LIMIT
        ),

        "top_k": TOP_K,

        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),

        "results": all_details,
    }

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details_output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_summary(
        all_summaries
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        SUMMARY_PATH
    )

    print(
        DETAILS_PATH
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()