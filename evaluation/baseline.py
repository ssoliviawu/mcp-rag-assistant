
import json
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

SUMMARY_PATH = RESULTS_DIR / "e6_lexical_boost_summary.json"
DETAILS_PATH = RESULTS_DIR / "e6_lexical_boost_details.json"


# ============================================================
# Evaluation configuration
# ============================================================

TOP_K = 10


# ============================================================
# Retrieval Matrix
# ============================================================
#
# E0:
#   Vector
#   candidate=10
#
# E1:
#   Vector
#   candidate=100
#
# E2:
#   Vector
#   candidate=100
#   source dedup
#
# E3a:
#   Vector
#   candidate=100
#   reranker
#
# E3:
#   Vector
#   candidate=100
#   reranker
#   source dedup
#
# E4:
#   Hybrid
#   candidate=30
#
# E5:
#   Hybrid
#   candidate=30
#   source dedup
#
# ============================================================

EXPERIMENTS = [
    {
        "id": "E0",
        "name": "vector",
        "description": "Vector baseline",
        "mode": "vector",
        "candidate_limit": 10,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": False,
    },
    {
        "id": "E1",
        "name": "vector_candidate100",
        "description": "Vector with larger candidate pool",
        "mode": "vector",
        "candidate_limit": 100,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": False,
    },
    {
        "id": "E2",
        "name": "vector_dedup",
        "description": "Vector + source deduplication",
        "mode": "vector_dedup",
        "candidate_limit": 100,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
    },
    {
        "id": "E3a",
        "name": "vector_rerank",
        "description": "Vector + ONNX reranker",
        "mode": "vector_rerank",
        "candidate_limit": 100,
        "top_k": TOP_K,
        "reranker": True,
        "dedup": False,
    },
    {
        "id": "E3",
        "name": "vector_rerank_dedup",
        "description": "Vector + ONNX reranker + source deduplication",
        "mode": "vector_rerank_dedup",
        "candidate_limit": 100,
        "top_k": TOP_K,
        "reranker": True,
        "dedup": True,
    },
    {
        "id": "E4",
        "name": "hybrid",
        "description": "Vector + keyword + RRF",
        "mode": "hybrid",
        "candidate_limit": 30,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": False,
    },
    {
        "id": "E5",
        "name": "hybrid_dedup",
        "description": "Hybrid + source deduplication",
        "mode": "hybrid_dedup",
        "candidate_limit": 30,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
    },
    {
        "id": "E6",
        "name": "vector_lexical_boost",
        "description": "Vector + lightweight lexical boost + source diversity",
        "mode": "vector_lexical_boost",
        "candidate_limit": 100,
        "top_k": TOP_K,
        "reranker": False,
        "dedup": True,
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
        source
        .strip()
        .rstrip("/")
        + "/"
    )
import re


def tokenize(text: str) -> set[str]:
    """
    Simple tokenization for English technical text.
    Keeps identifiers such as _meta and MCPServer.
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
# added for e6 experiment

def lexical_score(
    query: str,
    content: str,
) -> float:
    """
    Fraction of query tokens appearing in the chunk content.
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
    Run one retrieval experiment.

    All experiments return the final Top-K results.
    """

    from retrieval.search import (
        vector_search,
        vector_search_diverse,
        vector_rerank_dedup,
        hybrid_search,
        hybrid_search_diverse,
    )

    mode = experiment["mode"]
    candidate_limit = experiment["candidate_limit"]
    top_k = experiment["top_k"]

    # --------------------------------------------------------
    # E0 / E1
    #
    # Vector only
    # --------------------------------------------------------

    if mode == "vector":

        candidates = vector_search(
            query=query,
            limit=candidate_limit,
        )

        return candidates[:top_k]

    # --------------------------------------------------------
    # E2
    #
    # Vector + source dedup
    # --------------------------------------------------------

    if mode == "vector_dedup":

        return vector_search_diverse(
            query=query,
            limit=top_k,
            candidate_limit=candidate_limit,
            max_chunks_per_source=1,
        )

    # --------------------------------------------------------
    # E3a
    #
    # Vector + reranker
    # WITHOUT source deduplication
    # --------------------------------------------------------

    if mode == "vector_rerank":

        candidates = vector_search(
            query=query,
            limit=candidate_limit,
        )

        if not candidates:
            return []

        from retrieval.reranker import rerank

        reranked = rerank(
            query,
            candidates,
        )

        return reranked[:top_k]

    # --------------------------------------------------------
    # E3
    #
    # Vector + reranker + source dedup
    # --------------------------------------------------------

    if mode == "vector_rerank_dedup":

        return vector_rerank_dedup(
            query=query,
            limit=top_k,
            candidate_limit=candidate_limit,
        )

    # --------------------------------------------------------
    # E4
    #
    # Hybrid
    # --------------------------------------------------------

    if mode == "hybrid":

        return hybrid_search(
            query=query,
            limit=top_k,
            candidate_limit=candidate_limit,
            rrf_k=60,
        )

    # --------------------------------------------------------
    # E5
    #
    # Hybrid + source dedup
    # --------------------------------------------------------

    if mode == "hybrid_dedup":

        return hybrid_search_diverse(
            query=query,
            limit=top_k,
            candidate_limit=candidate_limit,
            rrf_k=60,
        )
    # --------------------------------------------------------
    # E6
    #
    # Vector + lightweight lexical boost
    # + source diversity
    # --------------------------------------------------------

    if mode == "vector_lexical_boost":

        candidates = vector_search(
            query=query,
            limit=candidate_limit,
        )

        if not candidates:
            return []

        def lexical_score(query, content):
            if not content:
                return 0.0

            query_words = set(
                word.lower()
                for word in query.split()
                if len(word) > 2
            )

            content_lower = content.lower()

            if not query_words:
                return 0.0

            matched = sum(
                1
                for word in query_words
                if word in content_lower
            )

            return matched / len(query_words)

        alpha = 0.05

        scored = []

        for result in candidates:

            similarity = float(
                result.get("similarity", 0.0)
            )

            content = result.get(
                "content",
                "",
            )

            lexical = lexical_score(
                query,
                content,
            )

            final_score = (
                similarity
                + alpha * lexical
            )

            scored.append(
                (
                    final_score,
                    result,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # Source diversity
        results = []
        source_counts = {}

        for _, result in scored:

            source = (
                result.get("url")
                or result.get("source")
                or result.get("id")
            )

            if source_counts.get(
                source,
                0,
            ) >= 1:
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

    raise ValueError(
        f"Unknown experiment mode: {mode}"
    )


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

    return {

        "id": question_id,

        "question": question,

        "experiment": experiment["id"],

        "mode": experiment["name"],

        "relevant_sources": relevant_sources,

        "retrieved_sources": retrieved_sources,

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
# Calculate summary
# ============================================================

def calculate_summary(
    details: list[dict],
) -> dict:

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
            [
                x["latency_ms"]
                for x in details
            ]
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
        f"Reranker: "
        f"{experiment['reranker']}"
    )

    print(
        f"Dedup: "
        f"{experiment['dedup']}"
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
# Print Matrix Summary
# ============================================================

def print_summary(
    summaries: dict[str, dict],
):

    print()
    print("=" * 100)
    print("RETRIEVAL MATRIX")
    print("=" * 100)

    print()

    header = (
        f"{'Exp':<6}"
        f"{'Mode':<25}"
        f"{'Hit@1':>9}"
        f"{'Hit@3':>9}"
        f"{'Hit@5':>9}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR':>9}"
        f"{'Latency':>12}"
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
            f"{experiment_id:<6}"
            f"{experiment['name']:<25}"
            f"{summary['hit@1']:>9.4f}"
            f"{summary['hit@3']:>9.4f}"
            f"{summary['hit@5']:>9.4f}"
            f"{summary['hit@10']:>10.4f}"
            f"{summary['recall@10']:>12.4f}"
            f"{summary['mrr']:>9.4f}"
            f"{summary['avg_latency_ms']:>12.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    dataset = load_dataset(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Separate RAG questions from specification questions.
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
    print("MCP RAG RETRIEVAL MATRIX")
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
        f"Top-K: "
        f"{TOP_K}"
    )

    print()

    all_details = {}

    all_summaries = {}

    # --------------------------------------------------------
    # Run only E1 and E3a
    #
    # E1:
    #   Vector 100 -> Top 10
    #
    # E3a:
    #   Vector 100 -> Reranker -> Top 10
    # --------------------------------------------------------

    experiments_to_run = [
        experiment
        for experiment in EXPERIMENTS
        if experiment["id"] in ["E6",]
    ]

    for experiment in experiments_to_run:
        print(
    "EXPERIMENTS TO RUN:",
    [experiment["id"] for experiment in experiments_to_run]
)

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

        "excluded_spec_questions": (
            len(
                excluded_spec_questions
            )
        ),

        "excluded_spec_question_ids": (
            excluded_spec_questions
        ),

        "top_k": TOP_K,

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

        "top_k": TOP_K,

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

