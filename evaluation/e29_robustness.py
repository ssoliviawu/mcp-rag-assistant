from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================================
# Project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Project imports
# ============================================================================

from embedding.model import BGEEmbeddingModel
from retrieval.search import search_chunks


# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = PROJECT_ROOT / "mcp_rag_eval_dataset.jsonl"

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

OUTPUT_JSON = RESULTS_DIR / "e29_robustness.json"

CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

# Same exclusion used by E25/E27/E28
EXCLUDE_DOMAIN = "spec"


# ============================================================================
# Normalization
# ============================================================================

def normalize_source(source: str | None) -> str:
    if not source:
        return ""

    return source.rstrip("/")


# ============================================================================
# Dataset
# ============================================================================

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Same exclusion as E25/E27/E28
            if item.get("domain") == EXCLUDE_DOMAIN:
                continue

            dataset.append(item)

    return dataset


# ============================================================================
# Gold sources
# ============================================================================

def get_relevant_sources(item: dict) -> set[str]:
    sources = item.get("relevant_sources", [])

    if isinstance(sources, str):
        sources = [sources]

    return {
        normalize_source(source)
        for source in sources
        if source
    }


# ============================================================================
# Query classification
# ============================================================================
#
# IMPORTANT:
# This classification is only for robustness analysis.
# It DOES NOT affect retrieval.
#
# Priority:
#   1. Existing dataset domain/category if useful
#   2. Keyword-based classification
#   3. Generic conceptual fallback
#
# The retrieval pipeline itself remains identical for every question.
# ============================================================================

def classify_query(item: dict) -> str:
    question = (
        item.get("question")
        or ""
    ).strip().lower()

    domain = (
        item.get("domain")
        or ""
    ).strip().lower()

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    migration_keywords = [
        "migrate",
        "migration",
        "migrating",
        "deprecated",
        "breaking change",
        "upgrade",
        "upgrading",
        "changed from",
        "previous version",
        "old version",
        "new version",
    ]

    if any(keyword in question for keyword in migration_keywords):
        return "Migration"

    # ------------------------------------------------------------------
    # Error / Troubleshooting
    # ------------------------------------------------------------------

    error_keywords = [
        "error",
        "exception",
        "fail",
        "failed",
        "failure",
        "doesn't work",
        "does not work",
        "not working",
        "cannot",
        "can't",
        "unable",
        "invalid",
        "timeout",
        "debug",
        "troubleshoot",
        "problem",
        "issue",
        "why does",
        "why is",
        "why can't",
        "why cannot",
    ]

    if any(keyword in question for keyword in error_keywords):
        return "Error / Troubleshooting"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    configuration_keywords = [
        "configure",
        "configuration",
        "config",
        "setting",
        "settings",
        "environment variable",
        "env var",
        "port",
        "host",
        "url",
        "authentication",
        "auth",
        "credential",
        "transport",
        "stdio",
        "sse",
        "streamable http",
    ]

    if any(keyword in question for keyword in configuration_keywords):
        return "Configuration"

    # ------------------------------------------------------------------
    # API / Usage
    # ------------------------------------------------------------------

    api_keywords = [
        "how do i",
        "how can i",
        "how to",
        "create",
        "define",
        "implement",
        "call",
        "use",
        "register",
        "add",
        "install",
        "setup",
        "set up",
        "example",
        "python sdk",
        "sdk",
        "api",
    ]

    if any(keyword in question for keyword in api_keywords):
        return "API / Usage"

    # ------------------------------------------------------------------
    # Behavior / Structured Output
    # ------------------------------------------------------------------

    behavior_keywords = [
        "what happens",
        "what happens when",
        "returns",
        "return type",
        "primitive type",
        "structured output",
        "output schema",
        "behavior",
        "behaviour",
        "result",
        "response",
        "output",
    ]

    if any(keyword in question for keyword in behavior_keywords):
        return "Behavior / Structured Output"

    # ------------------------------------------------------------------
    # Existing domain fallback
    # ------------------------------------------------------------------

    if domain:
        if "migration" in domain:
            return "Migration"

        if "error" in domain or "troubleshoot" in domain:
            return "Error / Troubleshooting"

        if "config" in domain:
            return "Configuration"

        if "api" in domain or "usage" in domain:
            return "API / Usage"

        if "behavior" in domain or "structured" in domain:
            return "Behavior / Structured Output"

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    return "Conceptual"


# ============================================================================
# Candidate deduplication
# ============================================================================
#
# E25 baseline uses max_chunks_per_source=1.
#
# We reproduce that behavior after the original Top100 candidate generation.
#
# IMPORTANT:
# This does NOT change candidate generation.
# It only applies the same source-level diversity constraint before metrics.
# ============================================================================

def apply_source_limit(
    candidates: list[dict],
    max_chunks_per_source: int,
) -> list[dict]:

    counts: dict[str, int] = defaultdict(int)

    selected = []

    for candidate in candidates:

        source = normalize_source(
            candidate.get("source")
            or candidate.get("url")
            or candidate.get("metadata", {}).get("source")
            or ""
        )

        # If source is missing, keep it.
        if not source:
            selected.append(candidate)
            continue

        if counts[source] >= max_chunks_per_source:
            continue

        counts[source] += 1
        selected.append(candidate)

    return selected


# ============================================================================
# Find first relevant rank
# ============================================================================

def find_first_relevant_rank(
    candidates: list[dict],
    relevant_sources: set[str],
) -> int | None:

    for rank, candidate in enumerate(candidates, start=1):

        source = normalize_source(
            candidate.get("source")
            or candidate.get("url")
            or candidate.get("metadata", {}).get("source")
            or ""
        )

        if source in relevant_sources:
            return rank

    return None


# ============================================================================
# Evaluate one question
# ============================================================================

def evaluate_question(
    model: BGEEmbeddingModel,
    item: dict,
) -> dict:

    question_id = item["id"]
    question = item["question"]

    relevant_sources = get_relevant_sources(item)

    category = classify_query(item)

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    query_embedding = model.embed_query(question)

    # ------------------------------------------------------------------
    # ORIGINAL C7 RETRIEVAL
    #
    # No reranker
    # No hybrid
    # No lexical boost
    # No representation modification
    # ------------------------------------------------------------------

    candidates = search_chunks(
        query_embedding,
        limit=CANDIDATE_LIMIT,
    )

    # ------------------------------------------------------------------
    # Apply E25 source diversity constraint
    # ------------------------------------------------------------------

    ranked_candidates = apply_source_limit(
        candidates,
        MAX_CHUNKS_PER_SOURCE,
    )

    # ------------------------------------------------------------------
    # Relevant rank
    # ------------------------------------------------------------------

    first_relevant_rank = find_first_relevant_rank(
        ranked_candidates,
        relevant_sources,
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    hit_at_1 = (
        first_relevant_rank is not None
        and first_relevant_rank <= 1
    )

    hit_at_3 = (
        first_relevant_rank is not None
        and first_relevant_rank <= 3
    )

    hit_at_5 = (
        first_relevant_rank is not None
        and first_relevant_rank <= 5
    )

    hit_at_10 = (
        first_relevant_rank is not None
        and first_relevant_rank <= TOP_K
    )

    mrr = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    # ------------------------------------------------------------------
    # Save Top10
    # ------------------------------------------------------------------

    top10 = []

    for rank, candidate in enumerate(
        ranked_candidates[:TOP_K],
        start=1,
    ):

        top10.append(
            {
                "rank": rank,
                "title": (
                    candidate.get("title")
                    or candidate.get("metadata", {}).get("title")
                    or ""
                ),
                "section": (
                    candidate.get("section")
                    or candidate.get("metadata", {}).get("section")
                    or ""
                ),
                "source": (
                    candidate.get("source")
                    or candidate.get("url")
                    or candidate.get("metadata", {}).get("source")
                    or ""
                ),
                "similarity": candidate.get("similarity"),
            }
        )

    return {
        "id": question_id,
        "question": question,
        "category": category,
        "domain": item.get("domain"),
        "relevant_sources": sorted(relevant_sources),
        "first_relevant_rank": first_relevant_rank,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
        "top10": top10,
    }


# ============================================================================
# Aggregate metrics
# ============================================================================

def aggregate_results(results: list[dict]) -> dict:

    total = len(results)

    if total == 0:
        return {
            "count": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
            "mrr": 0.0,
        }

    return {
        "count": total,

        "hit_at_1": (
            sum(item["hit_at_1"] for item in results)
            / total
        ),

        "hit_at_3": (
            sum(item["hit_at_3"] for item in results)
            / total
        ),

        "hit_at_5": (
            sum(item["hit_at_5"] for item in results)
            / total
        ),

        "hit_at_10": (
            sum(item["hit_at_10"] for item in results)
            / total
        ),

        "mrr": (
            sum(item["mrr"] for item in results)
            / total
        ),
    }


# ============================================================================
# Aggregate by category
# ============================================================================

def aggregate_by_category(
    results: list[dict],
) -> dict:

    grouped: dict[str, list[dict]] = defaultdict(list)

    for item in results:
        grouped[item["category"]].append(item)

    output = {}

    for category in sorted(grouped):

        category_results = grouped[category]

        output[category] = aggregate_results(
            category_results
        )

    return output


# ============================================================================
# Build category diagnosis
# ============================================================================

def build_category_diagnosis(
    category_metrics: dict,
) -> list[dict]:

    diagnosis = []

    for category, metrics in category_metrics.items():

        hit1 = metrics["hit_at_1"]
        hit10 = metrics["hit_at_10"]
        mrr = metrics["mrr"]
        count = metrics["count"]

        if count < 3:
            status = "SMALL_SAMPLE"

        elif hit10 >= 0.70 and mrr >= 0.50:
            status = "STRONG"

        elif hit10 >= 0.50 and mrr >= 0.30:
            status = "STABLE"

        elif hit10 >= 0.40:
            status = "WEAK"

        else:
            status = "PROBLEMATIC"

        diagnosis.append(
            {
                "category": category,
                "count": count,
                "hit_at_1": hit1,
                "hit_at_10": hit10,
                "mrr": mrr,
                "status": status,
            }
        )

    return diagnosis


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    print("=" * 100)
    print("E29 - QUERY-TYPE ROBUSTNESS / GENERALIZATION")
    print("=" * 100)

    print()
    print("Configuration:")
    print(f"  Candidate limit          : {CANDIDATE_LIMIT}")
    print(f"  Top K                    : {TOP_K}")
    print(f"  Max chunks per source   : {MAX_CHUNKS_PER_SOURCE}")
    print("  Reranker                 : NO")
    print("  Hybrid                   : NO")
    print("  Lexical boost            : NO")
    print("  Representation change    : NO")
    print()

    print("Purpose:")
    print(
        "  Test whether the selected E25 C7 baseline "
        "generalizes across query types."
    )

    print()
    print("Loading dataset...")

    dataset = load_dataset()

    print(
        f"Non-spec questions : {len(dataset)}"
    )

    print()
    print("Loading embedding model...")

    model = BGEEmbeddingModel()

    results = []

    print()
    print("=" * 100)
    print("RUNNING RETRIEVAL")
    print("=" * 100)

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{item['id']} | "
            f"{item['question']}"
        )

        result = evaluate_question(
            model,
            item,
        )

        results.append(result)

        rank = result["first_relevant_rank"]

        print(
            f"    category={result['category']} "
            f"rank={rank}"
        )

    # =========================================================================
    # Overall
    # =========================================================================

    overall = aggregate_results(results)

    # =========================================================================
    # Category metrics
    # =========================================================================

    category_metrics = aggregate_by_category(
        results
    )

    category_diagnosis = build_category_diagnosis(
        category_metrics
    )

    # =========================================================================
    # Print overall
    # =========================================================================

    print()
    print("=" * 100)
    print("OVERALL RESULTS")
    print("=" * 100)

    print(
        f"Questions : {overall['count']}"
    )

    print(
        f"Hit@1    : {overall['hit_at_1']:.4f}"
    )

    print(
        f"Hit@3    : {overall['hit_at_3']:.4f}"
    )

    print(
        f"Hit@5    : {overall['hit_at_5']:.4f}"
    )

    print(
        f"Hit@10   : {overall['hit_at_10']:.4f}"
    )

    print(
        f"MRR      : {overall['mrr']:.4f}"
    )

    # =========================================================================
    # Print category
    # =========================================================================

    print()
    print("=" * 100)
    print("CATEGORY RESULTS")
    print("=" * 100)

    header = (
        f"{'Category':<30}"
        f"{'N':>5}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print(header)
    print("-" * len(header))

    for category in sorted(category_metrics):

        metrics = category_metrics[category]

        print(
            f"{category:<30}"
            f"{metrics['count']:>5}"
            f"{metrics['hit_at_1']:>10.4f}"
            f"{metrics['hit_at_3']:>10.4f}"
            f"{metrics['hit_at_5']:>10.4f}"
            f"{metrics['hit_at_10']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
        )

    # =========================================================================
    # Diagnosis
    # =========================================================================

    print()
    print("=" * 100)
    print("CATEGORY DIAGNOSIS")
    print("=" * 100)

    for item in category_diagnosis:

        print(
            f"{item['category']:<30}"
            f"N={item['count']:<3} "
            f"Hit@1={item['hit_at_1']:.4f} "
            f"Hit@10={item['hit_at_10']:.4f} "
            f"MRR={item['mrr']:.4f} "
            f"[{item['status']}]"
        )

    # =========================================================================
    # Residual questions
    # =========================================================================

    residuals = [
        item
        for item in results
        if not item["hit_at_10"]
    ]

    print()
    print("=" * 100)
    print("HIT@10 RESIDUALS")
    print("=" * 100)

    print(
        f"Residual count : {len(residuals)}"
    )

    for item in residuals:

        print()
        print(
            f"{item['id']} | "
            f"{item['category']} | "
            f"first_relevant_rank={item['first_relevant_rank']}"
        )

        print(
            f"Q: {item['question']}"
        )

    # =========================================================================
    # Save JSON
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "E29",
        "name": "query_type_robustness_generalization",
        "dataset": str(DATASET_PATH),

        "purpose": (
            "Evaluate whether the selected E25 C7 baseline "
            "generalizes across different query types."
        ),

        "pipeline": {
            "baseline": "E25 C7",
            "candidate_limit": CANDIDATE_LIMIT,
            "top_k": TOP_K,
            "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,
            "reranker": False,
            "hybrid": False,
            "lexical_boost": False,
            "representation_change": False,
        },

        "question_count": len(results),

        "overall": overall,

        "category_metrics": category_metrics,

        "category_diagnosis": category_diagnosis,

        "hit_at_10_residual_count": len(residuals),

        "hit_at_10_residual_ids": [
            item["id"]
            for item in residuals
        ],

        "results": results,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================================
    # Save TXT
    # =========================================================================

    lines = []

    lines.append("=" * 100)
    lines.append(
        "E29 - QUERY-TYPE ROBUSTNESS / GENERALIZATION"
    )
    lines.append("=" * 100)

    lines.append("")
    lines.append("CONFIGURATION")
    lines.append("-" * 100)

    lines.append(
        f"Baseline              : E25 C7"
    )

    lines.append(
        f"Candidate limit       : {CANDIDATE_LIMIT}"
    )

    lines.append(
        f"Top K                 : {TOP_K}"
    )

    lines.append(
        f"Max chunks/source     : {MAX_CHUNKS_PER_SOURCE}"
    )

    lines.append(
        "Reranker              : NO"
    )

    lines.append(
        "Hybrid                : NO"
    )

    lines.append(
        "Lexical boost         : NO"
    )

    lines.append(
        "Representation change : NO"
    )

    lines.append("")
    lines.append("OVERALL")
    lines.append("-" * 100)

    lines.append(
        f"Questions : {overall['count']}"
    )

    lines.append(
        f"Hit@1    : {overall['hit_at_1']:.4f}"
    )

    lines.append(
        f"Hit@3    : {overall['hit_at_3']:.4f}"
    )

    lines.append(
        f"Hit@5    : {overall['hit_at_5']:.4f}"
    )

    lines.append(
        f"Hit@10   : {overall['hit_at_10']:.4f}"
    )

    lines.append(
        f"MRR      : {overall['mrr']:.4f}"
    )

    lines.append("")
    lines.append("CATEGORY METRICS")
    lines.append("-" * 100)

    for category in sorted(category_metrics):

        metrics = category_metrics[category]

        lines.append(
            f"{category:<30} "
            f"N={metrics['count']} "
            f"Hit@1={metrics['hit_at_1']:.4f} "
            f"Hit@3={metrics['hit_at_3']:.4f} "
            f"Hit@5={metrics['hit_at_5']:.4f} "
            f"Hit@10={metrics['hit_at_10']:.4f} "
            f"MRR={metrics['mrr']:.4f}"
        )

    lines.append("")
    lines.append("CATEGORY DIAGNOSIS")
    lines.append("-" * 100)

    for item in category_diagnosis:

        lines.append(
            f"{item['category']:<30} "
            f"N={item['count']} "
            f"Hit@1={item['hit_at_1']:.4f} "
            f"Hit@10={item['hit_at_10']:.4f} "
            f"MRR={item['mrr']:.4f} "
            f"STATUS={item['status']}"
        )

    lines.append("")
    lines.append("HIT@10 RESIDUALS")
    lines.append("-" * 100)

    for item in residuals:

        lines.append(
            f"{item['id']} | "
            f"{item['category']} | "
            f"rank={item['first_relevant_rank']} | "
            f"{item['question']}"
        )

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    # =========================================================================
    # Final
    # =========================================================================

    print()
    print("=" * 100)
    print("E29 COMPLETE")
    print("=" * 100)

    print()
    print(
        f"JSON : {OUTPUT_JSON}"
    )

    print(
        f"TXT  : {OUTPUT_TXT}"
    )

    print()
    print(
        "Next decision:"
    )

    print(
        "  If category performance is reasonably stable, "
        "freeze retrieval and proceed to E30."
    )

    print(
        "  Do NOT add reranker merely because a category is weaker."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()