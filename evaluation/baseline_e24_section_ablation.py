import json
import re
import time
from pathlib import Path

import numpy as np

from embedding.model import BGEEmbeddingModel


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    RESULTS_DIR / "e24_section_ablation_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR / "e24_section_ablation_details.json"
)

TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1
BATCH_SIZE = 32


# ============================================================
# Load
# ============================================================

def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def normalize_url(url):
    if not url:
        return ""

    return url.rstrip("/") + "/"


def get_relevant_sources(item):
    return {
        normalize_url(source)
        for source in item.get(
            "relevant_sources",
            [],
        )
        if source
    }


# ============================================================
# E23 rules
# ============================================================

GENERIC_SECTIONS = {
    "recap",
    "requirements",
    "create it",
    "try it",
    "example",
    "installation",
    "getting started",
    "introduction",
    "overview",
    "summary",
    "conclusion",
    "references",
    "contents",
    "table of contents",
}


# ============================================================
# Section processing
# ============================================================

def clean_section(section):
    if not section:
        return ""

    section = section.strip()

    section = re.sub(
        r"\s+",
        " ",
        section,
    )

    return section


def get_section_parts(section):
    section = clean_section(section)

    if not section:
        return []

    return [
        part.strip()
        for part in re.split(
            r"\s*>\s*",
            section,
        )
        if part.strip()
    ]


def section_is_low_information(
    section,
    disabled_rule=None,
):
    """
    Return True if section should NOT be included.

    disabled_rule:
        One generic rule that is temporarily disabled.

    Example:
        disabled_rule="recap"

    Then "Recap" will be treated as informative
    for that ablation experiment.
    """

    section = clean_section(section)

    if not section:
        return True

    lower = section.lower().strip()

    # --------------------------------------------------------
    # Exact generic section
    # --------------------------------------------------------

    if (
        lower in GENERIC_SECTIONS
        and lower != disabled_rule
    ):
        return True

    # --------------------------------------------------------
    # Very short sections
    # --------------------------------------------------------

    words = re.findall(
        r"[A-Za-z0-9_]+",
        lower,
    )

    if len(words) <= 1:
        return True

    # --------------------------------------------------------
    # Generic final breadcrumb component
    # --------------------------------------------------------

    parts = get_section_parts(section)

    if parts:

        last = parts[-1].lower()

        for generic in GENERIC_SECTIONS:

            if generic == disabled_rule:
                continue

            if last == generic:
                return True

            if last.startswith(
                generic + " "
            ):
                return True

    # --------------------------------------------------------
    # Very long breadcrumb
    # --------------------------------------------------------

    if len(section) > 300:
        return True

    return False


def section_quality(
    section,
    disabled_rule=None,
):
    return not section_is_low_information(
        section,
        disabled_rule=disabled_rule,
    )


# ============================================================
# Representation
# ============================================================

def build_representation(
    chunk,
    disabled_rule=None,
):
    content = (
        chunk.get("content") or ""
    ).strip()

    title = (
        chunk.get("title") or ""
    ).strip()

    section = (
        chunk.get("section") or ""
    ).strip()

    if section_quality(
        section,
        disabled_rule=disabled_rule,
    ):
        return (
            f"{title}\n\n"
            f"{section}\n\n"
            f"{content}"
        )

    return (
        f"{title}\n\n"
        f"{content}"
    )


# ============================================================
# Source diversity
# ============================================================

def diversify(
    results,
    limit=TOP_K,
):
    output = []
    source_counts = {}

    for result in results:

        source = normalize_url(
            result.get("url", "")
        )

        if (
            source_counts.get(
                source,
                0,
            )
            >= MAX_CHUNKS_PER_SOURCE
        ):
            continue

        output.append(result)

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            )
            + 1
        )

        if len(output) >= limit:
            break

    return output


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    results,
    relevant_sources,
):
    retrieved_sources = [
        normalize_url(
            result.get("url", "")
        )
        for result in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    metrics = {}

    for k in [1, 3, 5, 10]:

        metrics[f"hit@{k}"] = (
            1.0
            if any(hits[:k])
            else 0.0
        )

    gold_retrieved = len(
        set(
            retrieved_sources[:10]
        )
        & relevant_sources
    )

    if relevant_sources:
        metrics["recall@10"] = (
            gold_retrieved
            / len(relevant_sources)
        )
    else:
        metrics["recall@10"] = 0.0

    mrr = 0.0

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if hit:
            mrr = 1.0 / rank
            break

    metrics["mrr"] = mrr

    return metrics


def aggregate_metrics(results):
    keys = [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr",
    ]

    output = {}

    for key in keys:

        values = [
            item["metrics"][key]
            for item in results
        ]

        output[key] = float(
            np.mean(values)
        )

    return output


# ============================================================
# Gold rank
# ============================================================

def get_gold_rank(
    results,
    relevant_sources,
):
    for rank, result in enumerate(
        results,
        start=1,
    ):

        source = normalize_url(
            result.get("url", "")
        )

        if source in relevant_sources:
            return rank

    return None


def compare_rank(
    baseline_rank,
    ablation_rank,
):
    if (
        baseline_rank is None
        and ablation_rank is None
    ):
        return "BOTH_MISSING"

    if baseline_rank is None:
        return "NEW_HIT"

    if ablation_rank is None:
        return "NEW_MISS"

    if ablation_rank < baseline_rank:
        return "IMPROVED"

    if ablation_rank > baseline_rank:
        return "WORSENED"

    return "UNCHANGED"


# ============================================================
# One experiment
# ============================================================

def run_experiment(
    dataset,
    chunks,
    model,
    disabled_rule=None,
):
    """
    disabled_rule=None
        = E23 baseline

    disabled_rule="recap"
        = remove the "recap" filtering rule

    Everything else remains identical.
    """

    representation_embeddings = []

    texts = [
        build_representation(
            chunk,
            disabled_rule=disabled_rule,
        )
        for chunk in chunks
    ]

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch = texts[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        embeddings = (
            model.embed_documents(batch)
        )

        representation_embeddings.append(
            embeddings
        )

    doc_embeddings = np.vstack(
        representation_embeddings
    )

    embedding_time = (
        time.perf_counter() - start
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f}s"
    )

    question_results = []

    retrieval_start = time.perf_counter()

    for item in dataset:

        query = item["question"]

        relevant_sources = (
            get_relevant_sources(item)
        )

        query_embedding = (
            model.embed_query(query)
        )

        similarities = np.dot(
            doc_embeddings,
            query_embedding,
        )

        top_indices = np.argsort(
            -similarities
        )[:CANDIDATE_LIMIT]

        candidates = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            chunk = chunks[index]

            candidates.append({
                "rank": rank,
                "id": chunk.get("id"),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "url": normalize_url(
                    chunk.get("url", "")
                ),
                "similarity": float(
                    similarities[index]
                ),
            })

        final_results = diversify(
            candidates,
            limit=TOP_K,
        )

        metrics = calculate_metrics(
            final_results,
            relevant_sources,
        )

        gold_rank = get_gold_rank(
            final_results,
            relevant_sources,
        )

        question_results.append({
            "id": item["id"],
            "question": query,
            "gold_rank": gold_rank,
            "metrics": metrics,
            "top10": final_results,
        })

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    return {
        "metrics": aggregate_metrics(
            question_results
        ),
        "questions": question_results,
        "embedding_time_seconds": (
            embedding_time
        ),
        "retrieval_time_seconds": (
            retrieval_time
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E24 - SECTION RULE ABLATION")
    print("=" * 80)

    dataset = load_jsonl(
        DATASET_PATH
    )

    dataset = [
        item
        for item in dataset
        if item.get("domain") != "spec"
    ]

    chunks = load_jsonl(
        CHUNKS_PATH
    )

    print()
    print(
        f"Questions : {len(dataset)}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    print()

    model = BGEEmbeddingModel()

    # ========================================================
    # E23 baseline
    # ========================================================

    print()
    print("=" * 80)
    print("BASELINE: E23")
    print("=" * 80)

    baseline = run_experiment(
        dataset,
        chunks,
        model,
        disabled_rule=None,
    )

    print()
    print("E23 metrics:")

    for key, value in baseline[
        "metrics"
    ].items():
        print(
            f"{key:<12}: {value:.4f}"
        )

    # ========================================================
    # Ablations
    # ========================================================

    ablations = {}

    for rule in sorted(
        GENERIC_SECTIONS
    ):

        print()
        print("=" * 80)
        print(
            f"ABLATION: disable "
            f"'{rule}'"
        )
        print("=" * 80)

        result = run_experiment(
            dataset,
            chunks,
            model,
            disabled_rule=rule,
        )

        ablations[rule] = result

        print()
        print(
            f"{rule}:"
        )

        for key, value in result[
            "metrics"
        ].items():

            print(
                f"{key:<12}: "
                f"{value:.4f}"
            )

    # ========================================================
    # Compare each ablation against E23
    # ========================================================

    comparison = {}

    for rule, experiment in ablations.items():

        baseline_questions = {
            item["id"]: item
            for item in baseline[
                "questions"
            ]
        }

        ablation_questions = {
            item["id"]: item
            for item in experiment[
                "questions"
            ]
        }

        improved = []
        worsened = []
        unchanged = []
        new_hit = []
        new_miss = []

        for question_id in (
            baseline_questions.keys()
        ):

            a = baseline_questions[
                question_id
            ]

            b = ablation_questions[
                question_id
            ]

            transition = compare_rank(
                a["gold_rank"],
                b["gold_rank"],
            )

            record = {
                "id": question_id,
                "question": a["question"],
                "baseline_rank": a[
                    "gold_rank"
                ],
                "ablation_rank": b[
                    "gold_rank"
                ],
                "transition": transition,
            }

            if transition == "IMPROVED":
                improved.append(record)

            elif transition == "WORSENED":
                worsened.append(record)

            elif transition == "UNCHANGED":
                unchanged.append(record)

            elif transition == "NEW_HIT":
                new_hit.append(record)

            elif transition == "NEW_MISS":
                new_miss.append(record)

        comparison[rule] = {
            "metrics": {
                metric: {
                    "baseline": baseline[
                        "metrics"
                    ][metric],
                    "ablation": experiment[
                        "metrics"
                    ][metric],
                    "delta": (
                        experiment[
                            "metrics"
                        ][metric]
                        -
                        baseline[
                            "metrics"
                        ][metric]
                    ),
                }
                for metric in [
                    "hit@1",
                    "hit@3",
                    "hit@5",
                    "hit@10",
                    "recall@10",
                    "mrr",
                ]
            },

            "rank_change": {
                "improved": len(improved),
                "worsened": len(worsened),
                "unchanged": len(unchanged),
                "new_hit": len(new_hit),
                "new_miss": len(new_miss),
            },

            "improved_cases": improved,
            "worsened_cases": worsened,
        }

    # ========================================================
    # Rule ranking
    # ========================================================

    rule_ranking = []

    for rule, result in comparison.items():

        rule_ranking.append({
            "rule": rule,

            "hit@1_delta": result[
                "metrics"
            ]["hit@1"]["delta"],

            "hit@3_delta": result[
                "metrics"
            ]["hit@3"]["delta"],

            "hit@5_delta": result[
                "metrics"
            ]["hit@5"]["delta"],

            "mrr_delta": result[
                "metrics"
            ]["mrr"]["delta"],

            "improved": result[
                "rank_change"
            ]["improved"],

            "worsened": result[
                "rank_change"
            ]["worsened"],

            "net_rank_change": (
                result[
                    "rank_change"
                ]["improved"]
                -
                result[
                    "rank_change"
                ]["worsened"]
            ),
        })

    # Best rules to KEEP filtered
    #
    # If disabling a rule causes metrics to decrease,
    # that rule is useful.
    #
    # Therefore sort by positive MRR impact of
    # keeping the rule.
    rule_ranking.sort(
        key=lambda x: x[
            "mrr_delta"
        ]
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "experiment": (
            "E24_section_rule_ablation"
        ),

        "dataset": DATASET_PATH.name,

        "question_count": len(dataset),

        "chunk_count": len(chunks),

        "candidate_limit": CANDIDATE_LIMIT,

        "top_k": TOP_K,

        "max_chunks_per_source":
            MAX_CHUNKS_PER_SOURCE,

        "baseline": {
            "name": "E23",
            "metrics": baseline[
                "metrics"
            ],
        },

        "ablation_count": len(
            ablations
        ),

        "ablations": {
            rule: result[
                "metrics"
            ]
            for rule, result
            in ablations.items()
        },

        "comparisons": comparison,

        "rule_ranking": rule_ranking,
    }

    details = {
        "summary": summary,

        "baseline_questions":
            baseline[
                "questions"
            ],

        "ablations": {
            rule: {
                "metrics": result[
                    "metrics"
                ],
                "questions": result[
                    "questions"
                ],
            }
            for rule, result
            in ablations.items()
        },
    }

    # ========================================================
    # Save
    # ========================================================

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print final table
    # ========================================================

    print()
    print("=" * 80)
    print("E24 FINAL RULE ABLATION")
    print("=" * 80)

    print(
        f"{'Rule':<20}"
        f"{'Hit@1 Δ':>12}"
        f"{'Hit@3 Δ':>12}"
        f"{'MRR Δ':>12}"
        f"{'Improve':>10}"
        f"{'Worsen':>10}"
        f"{'Net':>8}"
    )

    print("-" * 80)

    # Sort by MRR delta:
    # most negative = most useful filter
    for item in rule_ranking:

        print(
            f"{item['rule']:<20}"
            f"{item['hit@1_delta']:>12.4f}"
            f"{item['hit@3_delta']:>12.4f}"
            f"{item['mrr_delta']:>12.4f}"
            f"{item['improved']:>10}"
            f"{item['worsened']:>10}"
            f"{item['net_rank_change']:>8}"
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "Negative delta = disabling the rule "
        "made performance worse -> rule is useful."
    )

    print(
        "Positive delta = disabling the rule "
        "made performance better -> rule may be unnecessary."
    )

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"Details saved to:\n"
        f"{DETAILS_PATH}"
    )


if __name__ == "__main__":
    main()