"""
MCP RAG Retrieval Error Analysis

Usage:

    Vector:
        uv run python -m evaluation.error_analysis

    Keyword:
        uv run python -m evaluation.error_analysis --mode keyword

    Hybrid:
        uv run python -m evaluation.error_analysis --mode hybrid

    Change retrieval limit:
        uv run python -m evaluation.error_analysis --limit 10

    Show every query:
        uv run python -m evaluation.error_analysis --show-all

    Custom output:
        uv run python -m evaluation.error_analysis \
            --output evaluation/error_analysis.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from retrieval.search import (
    vector_search,
    vector_search_diverse,
    vector_rerank_dedup,
    keyword_search,
    hybrid_search,
    hybrid_search_diverse,
)

# ============================================================
# Configuration
# ============================================================

DEFAULT_DATASET = Path(
    "mcp_rag_eval_dataset.jsonl"
)

DEFAULT_OUTPUT = Path(
    "evaluation/error_analysis.json"
)

DEFAULT_LIMIT = 50

K_VALUES = [1, 3, 5, 10, 20, 30, 50]


# ============================================================
# Dataset
# ============================================================

def load_dataset(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load JSONL evaluation dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{path.resolve()}"
        )

    dataset = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            dataset.append(item)

    return dataset


# ============================================================
# URL normalization
# ============================================================

def normalize_url(
    url: str,
) -> str:
    """
    Normalize URL for evaluation.

    Example:

        https://example.com/
        https://example.com

    are treated as the same URL.
    """

    if not url:
        return ""

    url = url.strip()

    if (
        url.endswith("/")
        and url.count("/") > 2
    ):
        url = url.rstrip("/")

    return url


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query: str,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:

    if mode == "vector":

        return vector_search(
            query,
            limit=limit,
        )

    if mode == "vector_diverse":

        return vector_search_diverse(
            query,
            limit=limit,
            candidate_limit=100,
            max_chunks_per_source=1,
        )

    if mode == "keyword":

        return keyword_search(
            query,
            limit=limit,
        )

    if mode == "hybrid":

        return hybrid_search(
            query,
            limit=limit,
            candidate_limit=10,
        )

    if mode == "hybrid_diverse":

        return hybrid_search_diverse(
            query,
            limit=limit,
            candidate_limit=30,
        )
    
    if mode == "vector_rerank_dedup":

        return vector_rerank_dedup(
            query,
            limit=limit,
            candidate_limit=max(50, limit * 5),
        )

    raise ValueError(
        f"Unsupported retrieval mode: {mode}"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> dict[str, Any]:
    """
    Calculate retrieval metrics for one query.

    Metrics:

        Hit@1
        Hit@3
        Hit@5
        Hit@10

        Recall@5
        Recall@10

        MRR
    """

    retrieved = [
        normalize_url(source)
        for source in retrieved_sources
        if source
    ]

    relevant = {
        normalize_url(source)
        for source in relevant_sources
        if source
    }

    # --------------------------------------------------------
    # First relevant rank
    # --------------------------------------------------------

    first_relevant_rank = None

    for rank, source in enumerate(
        retrieved,
        start=1,
    ):

        if source in relevant:

            first_relevant_rank = rank
            break

    # --------------------------------------------------------
    # Hit@K
    # --------------------------------------------------------

    metrics = {}

    for k in K_VALUES:

        top_k = retrieved[:k]

        hit = any(
            source in relevant
            for source in top_k
        )

        metrics[
            f"hit@{k}"
        ] = 1.0 if hit else 0.0

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    for k in K_VALUES:

        top_k = set(
            retrieved[:k]
        )

        if relevant:

            matched = len(
                top_k.intersection(
                    relevant
                )
            )

            metrics[
                f"recall@{k}"
            ] = (
                matched
                / len(relevant)
            )

        else:

            metrics[
                f"recall@{k}"
            ] = 0.0

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    if first_relevant_rank is None:

        mrr = 0.0

    else:

        mrr = (
            1.0
            / first_relevant_rank
        )

    metrics["mrr"] = mrr

    # --------------------------------------------------------
    # Relevant sources
    # --------------------------------------------------------

    retrieved_set = set(
        retrieved
    )

    found_relevant = sorted(
        retrieved_set.intersection(
            relevant
        )
    )

    missing_relevant = sorted(
        relevant.difference(
            retrieved_set
        )
    )

    metrics[
        "first_relevant_rank"
    ] = first_relevant_rank

    metrics[
        "relevant_count"
    ] = len(relevant)

    metrics[
        "found_relevant_count"
    ] = len(
        found_relevant
    )

    metrics[
        "found_relevant_sources"
    ] = found_relevant

    metrics[
        "missing_relevant_sources"
    ] = missing_relevant

    # --------------------------------------------------------
    # Duplicate sources
    # --------------------------------------------------------

    source_counter = Counter(
        retrieved
    )

    duplicate_sources = {
        source: count
        for source, count
        in source_counter.items()
        if count > 1
    }

    duplicate_count = sum(
        count - 1
        for count
        in source_counter.values()
        if count > 1
    )

    metrics[
        "unique_sources"
    ] = len(
        source_counter
    )

    metrics[
        "duplicate_count"
    ] = duplicate_count

    metrics[
        "duplicate_sources"
    ] = duplicate_sources

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    if retrieved:

        metrics[
            "source_diversity"
        ] = (
            len(source_counter)
            / len(retrieved)
        )

    else:

        metrics[
            "source_diversity"
        ] = 0.0

    return metrics


# ============================================================
# Error classification
# ============================================================

def classify_error(
    metrics: dict[str, Any],
) -> str:
    """
    Classify retrieval result by the
    rank of the first relevant source.
    """

    rank = metrics[
        "first_relevant_rank"
    ]

    if rank is None:
        return "COMPLETE_MISS"

    if rank == 1:
        return "SUCCESS_TOP1"

    if rank <= 3:
        return "LOW_RANK_HIT"

    if rank <= 5:
        return "MID_RANK_HIT"

    if rank <= 10:
        return "LOWER_RANK_HIT"

    if rank <= 20:
        return "RANK_11_20"

    if rank <= 30:
        return "RANK_21_30"

    if rank <= 50:
        return "RANK_31_50"

    return "MISS"

# ============================================================
# Analyze one question
# ============================================================

def analyze_question(
    item: dict[str, Any],
    mode: str,
    limit: int,
) -> dict[str, Any]:
    """
    Analyze one evaluation question.
    """

    question_id = item.get(
        "id",
        "unknown",
    )

    question = item.get(
        "question",
        "",
    )

    relevant_sources = item.get(
        "relevant_sources",
        [],
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    start = time.perf_counter()

    raw_results = retrieve(
        query=question,
        mode=mode,
        limit=limit,
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    retrieved_sources = []

    retrieved_details = []

    for rank, result in enumerate(
        raw_results,
        start=1,
    ):

        source = normalize_url(
            result.get(
                "url",
                "",
            )
        )

        similarity = result.get(
            "similarity"
        )

        rrf_score = result.get(
            "rrf_score"
        )

        detail = {
            "rank": rank,
            "id": result.get(
                "id"
            ),
            "title": result.get(
                "title"
            ),
            "section": result.get(
                "section"
            ),
            "url": source,
            "similarity": similarity,
            "rrf_score": rrf_score,
        }

        retrieved_sources.append(
            source
        )

        retrieved_details.append(
            detail
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        retrieved_sources,
        relevant_sources,
    )

    error_type = classify_error(
        metrics
    )

    return {
        "id": question_id,
        "question": question,
        "mode": mode,
        "relevant_sources": [
            normalize_url(source)
            for source
            in relevant_sources
        ],
        "retrieved_sources":
            retrieved_sources,
        "retrieved_details":
            retrieved_details,
        "latency_ms":
            round(
                latency_ms,
                2,
            ),
        "metrics":
            metrics,
        "error_type":
            error_type,
    }


# ============================================================
# Aggregate
# ============================================================

def aggregate_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate metrics across all questions.
    """

    if not results:
        return {}

    aggregate = {}

    metric_names = [
    f"hit@{k}"
    for k in K_VALUES
] + [
    f"recall@{k}"
    for k in K_VALUES
] + [
    "mrr",
]

    for metric_name in metric_names:

        values = [
            result[
                "metrics"
            ][metric_name]
            for result in results
        ]

        aggregate[
            metric_name
        ] = round(
            statistics.mean(
                values
            ),
            4,
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latencies = [
        result[
            "latency_ms"
        ]
        for result in results
    ]

    aggregate[
        "avg_latency_ms"
    ] = round(
        statistics.mean(
            latencies
        ),
        2,
    )

    aggregate[
        "median_latency_ms"
    ] = round(
        statistics.median(
            latencies
        ),
        2,
    )

    sorted_latencies = sorted(
        latencies
    )

    if len(sorted_latencies) == 1:

        p95 = sorted_latencies[0]

    else:

        index = int(
            0.95
            * (
                len(sorted_latencies)
                - 1
            )
        )

        p95 = sorted_latencies[
            index
        ]

    aggregate[
        "p95_latency_ms"
    ] = round(
        p95,
        2,
    )

    # --------------------------------------------------------
    # Error categories
    # --------------------------------------------------------

    error_counter = Counter(
        result["error_type"]
        for result in results
    )

    aggregate[
        "error_categories"
    ] = dict(
        error_counter
    )

    # --------------------------------------------------------
    # Duplicate statistics
    # --------------------------------------------------------

    duplicate_counts = [
        result[
            "metrics"
        ][
            "duplicate_count"
        ]
        for result in results
    ]

    aggregate[
        "queries_with_duplicates"
    ] = sum(
        count > 0
        for count
        in duplicate_counts
    )

    aggregate[
        "avg_duplicate_count"
    ] = round(
        statistics.mean(
            duplicate_counts
        ),
        2,
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    diversity_values = [
        result[
            "metrics"
        ][
            "source_diversity"
        ]
        for result in results
    ]

    aggregate[
        "avg_source_diversity"
    ] = round(
        statistics.mean(
            diversity_values
        ),
        4,
    )

    return aggregate


# ============================================================
# Source statistics
# ============================================================

def build_source_statistics(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Analyze how often every URL is retrieved.
    """

    source_stats = {}

    for result in results:

        relevant = {
            normalize_url(source)
            for source
            in result[
                "relevant_sources"
            ]
        }

        for rank, source in enumerate(
            result[
                "retrieved_sources"
            ],
            start=1,
        ):

            if not source:
                continue

            if source not in source_stats:

                source_stats[
                    source
                ] = {
                    "source": source,
                    "retrieved_count": 0,
                    "relevant_count": 0,
                    "rank_sum": 0,
                    "rank_min": None,
                }

            stats = source_stats[
                source
            ]

            stats[
                "retrieved_count"
            ] += 1

            stats[
                "rank_sum"
            ] += rank

            if (
                stats["rank_min"]
                is None
                or rank
                < stats["rank_min"]
            ):

                stats[
                    "rank_min"
                ] = rank

            if source in relevant:

                stats[
                    "relevant_count"
                ] += 1

    output = []

    for stats in (
        source_stats.values()
    ):

        count = stats[
            "retrieved_count"
        ]

        stats[
            "avg_rank"
        ] = round(
            stats["rank_sum"]
            / count,
            2,
        )

        del stats[
            "rank_sum"
        ]

        output.append(
            stats
        )

    output.sort(
        key=lambda x: (
            -x[
                "retrieved_count"
            ],
            x[
                "avg_rank"
            ],
        )
    )

    return output


# ============================================================
# Console output
# ============================================================

def print_header(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

def print_summary(
    aggregate: dict[str, Any],
) -> None:

    print_header(
        "RETRIEVAL ERROR ANALYSIS SUMMARY"
    )

    # --------------------------------------------------------
    # Hit@K
    # --------------------------------------------------------

    print("Hit@K:")

    for k in K_VALUES:

        print(
            f"  Hit@{k:<2}     : "
            f"{aggregate[f'hit@{k}']:.4f}"
        )

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    print()

    print("Recall@K:")

    for k in K_VALUES:

        print(
            f"  Recall@{k:<2}  : "
            f"{aggregate[f'recall@{k}']:.4f}"
        )

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    print()

    print(
        f"MRR         : "
        f"{aggregate['mrr']:.4f}"
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print()

    print(
        f"Avg latency : "
        f"{aggregate['avg_latency_ms']:.2f} ms"
    )

    print(
        f"Median      : "
        f"{aggregate['median_latency_ms']:.2f} ms"
    )

    print(
        f"P95 latency : "
        f"{aggregate['p95_latency_ms']:.2f} ms"
    )

    # --------------------------------------------------------
    # Duplicate statistics
    # --------------------------------------------------------

    print()

    print(
        "Queries with duplicates: "
        f"{aggregate['queries_with_duplicates']}"
    )

    print(
        "Avg duplicate count: "
        f"{aggregate['avg_duplicate_count']:.2f}"
    )

    print(
        "Avg source diversity: "
        f"{aggregate['avg_source_diversity']:.4f}"
    )

    # --------------------------------------------------------
    # Error categories
    # --------------------------------------------------------

    print()

    print(
        "Error categories:"
    )

    for category, count in (
        aggregate[
            "error_categories"
        ].items()
    ):

        print(
            f"  {category:<20} "
            f"{count}"
        )

def print_worst_queries(
    results: list[dict[str, Any]],
) -> None:
    """
    Print worst retrieval cases.
    """

    def sort_key(
        result: dict[str, Any],
    ):

        metrics = result[
            "metrics"
        ]

        rank = metrics[
            "first_relevant_rank"
        ]

        if rank is None:
            rank_value = 999
        else:
            rank_value = rank

        return (
            -int(
                result[
                    "error_type"
                ]
                == "COMPLETE_MISS"
            ),
            rank_value,
            -metrics[
                "duplicate_count"
            ],
            metrics[
                "mrr"
            ],
        )

    worst = sorted(
        results,
        key=sort_key,
    )[:10]

    print_header(
        "WORST 10 RETRIEVAL CASES"
    )

    for index, result in enumerate(
        worst,
        start=1,
    ):

        metrics = result[
            "metrics"
        ]

        rank = metrics[
            "first_relevant_rank"
        ]

        rank_text = (
            str(rank)
            if rank is not None
            else "MISS"
        )

        print(
            f"{index:2d}. "
            f"{result['id']} | "
            f"first relevant="
            f"{rank_text:>4} | "
            f"MRR="
            f"{metrics['mrr']:.3f} | "
            f"duplicates="
            f"{metrics['duplicate_count']}"
        )

        print(
            f"    {result['question']}"
        )

def print_complete_misses(
    results: list[dict[str, Any]],
) -> None:
    """
    Print all COMPLETE_MISS queries.

    These are queries where none of the relevant
    sources appeared in the retrieved results.
    """

    misses = [
        result
        for result in results
        if result["error_type"] == "COMPLETE_MISS"
    ]

    print_header(
        f"COMPLETE MISS ANALYSIS ({len(misses)} queries)"
    )

    for index, result in enumerate(
        misses,
        start=1,
    ):
        metrics = result["metrics"]

        print()
        print(
            f"[{index}/{len(misses)}] "
            f"{result['id']}"
        )

        print(
            f"Question: "
            f"{result['question']}"
        )

        print()
        print("Expected relevant sources:")

        for source in result[
            "relevant_sources"
        ]:
            print(
                f"  ✓ {source}"
            )

        print()
        print("Top retrieved sources:")

        for detail in result[
            "retrieved_details"
        ]:
            rank = detail["rank"]
            source = detail["url"]

            similarity = detail[
                "similarity"
            ]

            rrf_score = detail[
                "rrf_score"
            ]

            if similarity is not None:

                score = (
                    f"similarity="
                    f"{similarity:.4f}"
                )

            elif rrf_score is not None:

                score = (
                    f"rrf="
                    f"{rrf_score:.6f}"
                )

            else:

                score = ""

            print(
                f"  {rank:2d}. "
                f"{source}"
                f"  {score}"
            )

        print()

        print(
            f"First relevant rank: "
            f"{metrics['first_relevant_rank']}"
        )

        print(
            f"Source diversity: "
            f"{metrics['source_diversity']:.4f}"
        )

        print(
            f"Duplicate count: "
            f"{metrics['duplicate_count']}"
        )

        print()
        print("-" * 80)

def print_question_analysis(
    result: dict[str, Any],
) -> None:

    metrics = result[
        "metrics"
    ]

    print()
    print("-" * 80)

    print(
        f"[{result['id']}] "
        f"{result['error_type']}"
    )

    print("-" * 80)

    print(
        f"Question: "
        f"{result['question']}"
    )

    print()
    print(
        "Relevant sources:"
    )

    for source in (
        result[
            "relevant_sources"
        ]
    ):

        print(
            f"  ✓ {source}"
        )

    print()
    print(
        "Retrieved:"
    )

    relevant = {
        normalize_url(source)
        for source
        in result[
            "relevant_sources"
        ]
    }

    for detail in (
        result[
            "retrieved_details"
        ]
    ):

        rank = detail[
            "rank"
        ]

        source = detail[
            "url"
        ]

        similarity = detail[
            "similarity"
        ]

        rrf_score = detail[
            "rrf_score"
        ]

        marker = (
            "✓"
            if source in relevant
            else " "
        )

        score_text = ""

        if similarity is not None:

            score_text = (
                f" similarity="
                f"{similarity:.4f}"
            )

        elif rrf_score is not None:

            score_text = (
                f" rrf="
                f"{rrf_score:.6f}"
            )

        print(
            f"  {marker} "
            f"{rank:2d}. "
            f"{source}"
            f"{score_text}"
        )

    print()

    print("Metrics:")

    print(
    "  Hit@K: "
    + ", ".join(
        f"@{k}={metrics[f'hit@{k}']:.0f}"
        for k in K_VALUES
    )
)

    print(
    "  Recall@K: "
    + ", ".join(
        f"@{k}={metrics[f'recall@{k}']:.2f}"
        for k in K_VALUES
    )
)

    print(
    f"  MRR={metrics['mrr']:.3f}"
)
    print(
        f"Latency: "
        f"{result['latency_ms']:.2f} ms"
    )

    if metrics[
        "duplicate_sources"
    ]:

        print()
        print(
            "Duplicate sources:"
        )

        for source, count in (
            metrics[
                "duplicate_sources"
            ].items()
        ):

            print(
                f"  {count}x "
                f"{source}"
            )

    if metrics[
        "missing_relevant_sources"
    ]:

        print()
        print(
            "Missing relevant sources:"
        )

        for source in (
            metrics[
                "missing_relevant_sources"
            ]
        ):

            print(
                f"  ✗ {source}"
            )


# ============================================================
# Save report
# ============================================================

def save_report(
    output_path: Path,
    dataset_path: Path,
    mode: str,
    limit: int,
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:

    report = {
        "dataset": str(
            dataset_path
        ),
        "mode": mode,
        "retrieval_limit": limit,
        "question_count": len(
            results
        ),
        "summary": aggregate,
        "source_statistics":
            build_source_statistics(
                results
            ),
        "results": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze MCP RAG "
            "retrieval errors."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Path to evaluation "
            "JSONL dataset."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "vector",
            "vector_diverse",
            "vector_rerank_dedup",
            "keyword",
            "hybrid",
            "hybrid_diverse",
        ],
        default="vector",
        help=(
            "Retrieval mode."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Number of retrieved "
            "results."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSON report."
        ),
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help=(
            "Show detailed analysis "
            "for every query."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_header(
        "MCP RAG RETRIEVAL ERROR ANALYSIS"
    )

    print(
        f"Dataset        : "
        f"{args.dataset}"
    )

    print(
        f"Mode           : "
        f"{args.mode}"
    )

    print(
        f"Retrieval limit: "
        f"{args.limit}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset(
        args.dataset
    )

    print(
        f"Questions      : "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    results = []

    print()
    print(
        "Running retrieval analysis..."
    )

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question_id = item.get(
            "id",
            f"q{index:03d}",
        )

        print(
            f"\rProcessing "
            f"{index}/{len(dataset)} "
            f"({question_id})...",
            end="",
            flush=True,
        )

        result = analyze_question(
            item=item,
            mode=args.mode,
            limit=args.limit,
        )

        results.append(
            result
        )

    print()

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregate = aggregate_results(
        results
    )

    print_summary(
        aggregate
    )

    # --------------------------------------------------------
    # Worst cases
    # --------------------------------------------------------

    print_worst_queries(
        results
    )

    # --------------------------------------------------------
    # Complete misses
    # --------------------------------------------------------

    print_complete_misses(
        results
    )

    # --------------------------------------------------------
    # Detailed analysis
    # --------------------------------------------------------

    if args.show_all:

        print_header(
            "DETAILED ANALYSIS"
        )

        selected = results

    else:

        selected = [
            result
            for result
            in results
            if result[
                "error_type"
            ] != "SUCCESS_TOP1"
        ]

        print_header(
            "DETAILED FAILURE ANALYSIS"
        )

        print(
            f"Showing "
            f"{len(selected)} "
            f"non-Top1 queries."
        )

    for result in selected:

        print_question_analysis(
            result
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_report(
        output_path=args.output,
        dataset_path=args.dataset,
        mode=args.mode,
        limit=args.limit,
        results=results,
        aggregate=aggregate,
    )

    print_header(
        "REPORT SAVED"
    )

    print(
        f"Output: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()