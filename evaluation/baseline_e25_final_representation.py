import json
import re
import time
from pathlib import Path

import numpy as np

from embedding.model import BGEEmbeddingModel

# ============================================================
# Import EXACT E23 section_quality logic
# ============================================================

from evaluation.baseline_e23_conditional_section import (
    section_quality as e23_section_quality,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    RESULTS_DIR
    / "e25_section_rule_combinations_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "e25_section_rule_combinations_details.json"
)


TOP_K = 10
CANDIDATE_LIMIT = 100
MAX_CHUNKS_PER_SOURCE = 1

BATCH_SIZE = 32


# ============================================================
# E25 candidate generic rules
# ============================================================

CANDIDATE_RULES = {
    "recap",
    "installation",
    "requirements",
}


# ============================================================
# 8 combinations
# ============================================================

COMBINATIONS = {
    "C0_none": set(),

    "C1_recap": {
        "recap",
    },

    "C2_installation": {
        "installation",
    },

    "C3_requirements": {
        "requirements",
    },

    "C4_recap_installation": {
        "recap",
        "installation",
    },

    "C5_recap_requirements": {
        "recap",
        "requirements",
    },

    "C6_installation_requirements": {
        "installation",
        "requirements",
    },

    "C7_all_three": {
        "recap",
        "installation",
        "requirements",
    },
}


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

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# URL
# ============================================================

def normalize_url(url):

    if not url:
        return ""

    return url.rstrip("/") + "/"


def get_relevant_sources(item):

    sources = item.get(
        "relevant_sources",
        [],
    )

    return {
        normalize_url(source)
        for source in sources
        if source
    }


# ============================================================
# Section quality
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


def section_quality_with_rules(
    section,
    generic_rules,
):
    """
    E25 section-quality logic.

    IMPORTANT:

    The non-generic E23 rules are preserved:

        1. empty section
        2. <= 1 word
        3. generic final breadcrumb component
        4. generic prefix
        5. section length > 300

    The ONLY thing changed is the generic-rule set.

    This allows E25 to test interactions between:

        recap
        installation
        requirements
    """

    section = clean_section(section)

    if not section:
        return False

    lower = section.lower().strip()

    # --------------------------------------------------------
    # 1. Exact generic sections
    # --------------------------------------------------------

    if lower in generic_rules:
        return False

    # --------------------------------------------------------
    # 2. Very short sections
    # --------------------------------------------------------

    words = re.findall(
        r"[A-Za-z0-9_]+",
        lower,
    )

    if len(words) <= 1:
        return False

    # --------------------------------------------------------
    # 3. Generic final breadcrumb component
    #
    # Example:
    #
    # MCP Python SDK > Example > Recap
    #
    # For E25, only the selected generic rules
    # are treated as low-information.
    # --------------------------------------------------------

    parts = [
        p.strip()
        for p in re.split(
            r"\s*>\s*",
            section,
        )
        if p.strip()
    ]

    if parts:

        last = parts[-1].lower()

        if last in generic_rules:
            return False

        # Handle:
        #
        # Recap — ...
        # Installation — ...
        # Requirements — ...
        #
        for generic in generic_rules:

            if last.startswith(
                generic + " "
            ):
                return False

    # --------------------------------------------------------
    # 4. Extremely long navigation paths
    # --------------------------------------------------------

    if len(section) > 300:
        return False

    return True


# ============================================================
# Representation
# ============================================================

def build_representation(
    chunk,
    generic_rules,
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

    if section_quality_with_rules(
        section,
        generic_rules,
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
# Exact E23 reproduction check
# ============================================================

def verify_e23_logic(chunks):

    """
    Sanity check:

    Compare our generic-rule framework against
    the actual E23 section_quality().

    This does NOT replace E23.

    It simply confirms that C7's generic-rule logic
    corresponds to E23's 14-rule logic only where
    applicable.

    We print the count for diagnostics.
    """

    e23_generic_sections = {
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

    mismatches = 0

    for chunk in chunks:

        section = chunk.get(
            "section",
            "",
        )

        # Reproduce E23's actual representation
        # decision using the imported function.
        actual = e23_section_quality(
            section
        )

        # Reproduce E23's generic list behavior
        # independently for sanity only.
        generic_test = section_quality_with_rules(
            section,
            e23_generic_sections,
        )

        if actual != generic_test:
            mismatches += 1

    return mismatches


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(
    query_embedding,
    document_embeddings,
):

    return np.dot(
        document_embeddings,
        query_embedding,
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
            r.get("url", "")
        )
        for r in results
    ]

    hits = [
        source in relevant_sources
        for source in retrieved_sources
    ]

    metrics = {}

    for k in [
        1,
        3,
        5,
        10,
    ]:

        top_k = hits[:k]

        metrics[
            f"hit@{k}"
        ] = (
            1.0
            if any(top_k)
            else 0.0
        )

    gold_count = len(
        relevant_sources
    )

    retrieved_gold = len(
        set(
            retrieved_sources[:10]
        )
        & relevant_sources
    )

    if gold_count:

        metrics["recall@10"] = (
            retrieved_gold
            / gold_count
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


def aggregate_metrics(
    question_results,
):

    if not question_results:
        return {}

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
            for item in question_results
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


# ============================================================
# Rank comparison
# ============================================================

def compare_rank(
    rank_a,
    rank_b,
):

    if (
        rank_a is None
        and rank_b is None
    ):
        return "BOTH_MISSING"

    if rank_a is None:
        return "NEW_HIT"

    if rank_b is None:
        return "NEW_MISS"

    if rank_b < rank_a:
        return "IMPROVED"

    if rank_b > rank_a:
        return "WORSENED"

    return "UNCHANGED"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E25 - SECTION RULE COMBINATION EXPERIMENT")
    print("=" * 80)

    dataset = load_jsonl(
        DATASET_PATH
    )

    chunks = load_jsonl(
        CHUNKS_PATH
    )

    # --------------------------------------------------------
    # Remove spec questions
    # --------------------------------------------------------

    dataset = [
        item
        for item in dataset
        if item.get("domain") != "spec"
    ]

    print()
    print(
        f"Questions : {len(dataset)}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    print()

    # --------------------------------------------------------
    # Print experiment definition
    # --------------------------------------------------------

    print(
        "Candidate generic rules:"
    )

    for rule in sorted(
        CANDIDATE_RULES
    ):
        print(
            f"  - {rule}"
        )

    print()

    print(
        "Combinations:"
    )

    for name, rules in (
        COMBINATIONS.items()
    ):

        print(
            f"  {name:<30} "
            f"{sorted(rules)}"
        )

    print()

    # --------------------------------------------------------
    # Sanity check against actual E23
    # --------------------------------------------------------

    print(
        "-" * 80
    )

    print(
        "Checking E23 section-quality reproduction..."
    )

    mismatch_count = (
        verify_e23_logic(
            chunks
        )
    )

    print(
        f"E23 reproduction mismatches: "
        f"{mismatch_count}"
    )

    if mismatch_count != 0:

        print()
        print(
            "WARNING: E23 logic mismatch detected."
        )
        print(
            "The E25 combination experiment will "
            "still run, but C7 may not exactly "
            "reproduce E23."
        )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BGEEmbeddingModel()

    # ========================================================
    # Build embeddings for all 8 combinations
    # ========================================================

    embeddings_by_combination = {}

    for name, generic_rules in (
        COMBINATIONS.items()
    ):

        print("-" * 80)

        print(
            f"Embedding representation: {name}"
        )

        print(
            f"Generic rules: "
            f"{sorted(generic_rules)}"
        )

        texts = [
            build_representation(
                chunk,
                generic_rules,
            )
            for chunk in chunks
        ]

        start_time = (
            time.perf_counter()
        )

        embeddings = []

        for start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):

            batch = texts[
                start:start + BATCH_SIZE
            ]

            batch_embeddings = (
                model.embed_documents(
                    batch
                )
            )

            embeddings.append(
                batch_embeddings
            )

        embeddings = np.vstack(
            embeddings
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Embedding finished in "
            f"{elapsed:.2f}s"
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        embeddings_by_combination[
            name
        ] = embeddings

    # ========================================================
    # Evaluate all combinations
    # ========================================================

    all_details = {}

    for name, generic_rules in (
        COMBINATIONS.items()
    ):

        print()
        print("=" * 80)
        print(
            f"EVALUATING: {name}"
        )
        print(
            f"Generic rules: "
            f"{sorted(generic_rules)}"
        )
        print("=" * 80)

        doc_embeddings = (
            embeddings_by_combination[
                name
            ]
        )

        question_results = []

        retrieval_start = (
            time.perf_counter()
        )

        for item in dataset:

            question_id = item["id"]
            question = item["question"]

            relevant_sources = (
                get_relevant_sources(
                    item
                )
            )

            query_embedding = (
                model.embed_query(
                    question
                )
            )

            similarities = (
                cosine_similarity(
                    query_embedding,
                    doc_embeddings,
                )
            )

            # ------------------------------------------------
            # Top100
            # ------------------------------------------------

            top_indices = np.argsort(
                -similarities
            )[:CANDIDATE_LIMIT]

            candidates = []

            for rank, index in enumerate(
                top_indices,
                start=1,
            ):

                chunk = chunks[index]

                candidates.append(
                    {
                        "rank": rank,
                        "id": chunk.get(
                            "id"
                        ),
                        "title": chunk.get(
                            "title"
                        ),
                        "section": chunk.get(
                            "section"
                        ),
                        "url": normalize_url(
                            chunk.get(
                                "url",
                                "",
                            )
                        ),
                        "similarity": float(
                            similarities[
                                index
                            ]
                        ),
                    }
                )

            # ------------------------------------------------
            # Source diversity
            # ------------------------------------------------

            final_results = diversify(
                candidates,
                limit=TOP_K,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                final_results,
                relevant_sources,
            )

            gold_rank = get_gold_rank(
                final_results,
                relevant_sources,
            )

            question_results.append(
                {
                    "id": question_id,
                    "question": question,
                    "combination": name,
                    "generic_rules": sorted(
                        generic_rules
                    ),
                    "gold_rank": gold_rank,
                    "metrics": metrics,
                    "top10": final_results,
                }
            )

        retrieval_elapsed = (
            time.perf_counter()
            - retrieval_start
        )

        aggregate = (
            aggregate_metrics(
                question_results
            )
        )

        print()

        print(
            f"Hit@1      : "
            f"{aggregate['hit@1']:.4f}"
        )

        print(
            f"Hit@3      : "
            f"{aggregate['hit@3']:.4f}"
        )

        print(
            f"Hit@5      : "
            f"{aggregate['hit@5']:.4f}"
        )

        print(
            f"Hit@10     : "
            f"{aggregate['hit@10']:.4f}"
        )

        print(
            f"Recall@10  : "
            f"{aggregate['recall@10']:.4f}"
        )

        print(
            f"MRR        : "
            f"{aggregate['mrr']:.4f}"
        )

        print(
            f"Retrieval time: "
            f"{retrieval_elapsed:.2f}s"
        )

        all_details[name] = {
            "metrics": aggregate,
            "questions": question_results,
        }

    # ========================================================
    # Compare all combinations against C7
    # ========================================================

    baseline_name = "C7_all_three"

    baseline_questions = (
        all_details[
            baseline_name
        ]["questions"]
    )

    comparisons = {}

    for name in COMBINATIONS:

        if name == baseline_name:
            continue

        current_questions = (
            all_details[
                name
            ]["questions"]
        )

        improved = 0
        worsened = 0
        unchanged = 0
        new_hit = 0
        new_miss = 0

        transitions = []

        for base, current in zip(
            baseline_questions,
            current_questions,
        ):

            rank_a = base["gold_rank"]
            rank_b = current["gold_rank"]

            transition = compare_rank(
                rank_a,
                rank_b,
            )

            if transition == "IMPROVED":
                improved += 1

            elif transition == "WORSENED":
                worsened += 1

            elif transition == "UNCHANGED":
                unchanged += 1

            elif transition == "NEW_HIT":
                new_hit += 1

            elif transition == "NEW_MISS":
                new_miss += 1

            transitions.append(
                {
                    "id": base["id"],
                    "question": base[
                        "question"
                    ],
                    "c7_rank": rank_a,
                    "current_rank": rank_b,
                    "transition": transition,
                }
            )

        comparisons[name] = {
            "vs": baseline_name,
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "new_hit": new_hit,
            "new_miss": new_miss,
            "transitions": transitions,
        }

    # ========================================================
    # Pairwise comparison matrix
    # ========================================================

    pairwise = {}

    combination_names = list(
        COMBINATIONS.keys()
    )

    for name_a in combination_names:

        pairwise[name_a] = {}

        questions_a = (
            all_details[
                name_a
            ]["questions"]
        )

        for name_b in combination_names:

            if name_a == name_b:
                continue

            questions_b = (
                all_details[
                    name_b
                ]["questions"]
            )

            improved = 0
            worsened = 0
            unchanged = 0
            new_hit = 0
            new_miss = 0

            for a, b in zip(
                questions_a,
                questions_b,
            ):

                transition = compare_rank(
                    a["gold_rank"],
                    b["gold_rank"],
                )

                if transition == "IMPROVED":
                    improved += 1

                elif transition == "WORSENED":
                    worsened += 1

                elif transition == "UNCHANGED":
                    unchanged += 1

                elif transition == "NEW_HIT":
                    new_hit += 1

                elif transition == "NEW_MISS":
                    new_miss += 1

            pairwise[name_a][
                name_b
            ] = {
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "new_hit": new_hit,
                "new_miss": new_miss,
            }

    # ========================================================
    # Ranking by metrics
    # ========================================================

    ranking = []

    for name, data in (
        all_details.items()
    ):

        metrics = data["metrics"]

        ranking.append(
            {
                "combination": name,
                "generic_rules": sorted(
                    COMBINATIONS[name]
                ),
                "rule_count": len(
                    COMBINATIONS[name]
                ),
                "hit@1": metrics[
                    "hit@1"
                ],
                "hit@3": metrics[
                    "hit@3"
                ],
                "hit@5": metrics[
                    "hit@5"
                ],
                "hit@10": metrics[
                    "hit@10"
                ],
                "recall@10": metrics[
                    "recall@10"
                ],
                "mrr": metrics[
                    "mrr"
                ],
            }
        )

    # Primary ranking: MRR
    # Secondary: Hit@1
    # Then fewer rules
    ranking.sort(
        key=lambda x: (
            -x["mrr"],
            -x["hit@1"],
            x["rule_count"],
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "experiment": (
            "E25_section_rule_combinations"
        ),

        "dataset": (
            DATASET_PATH.name
        ),

        "question_count": len(
            dataset
        ),

        "chunks": len(
            chunks
        ),

        "candidate_limit": (
            CANDIDATE_LIMIT
        ),

        "top_k": TOP_K,

        "max_chunks_per_source": (
            MAX_CHUNKS_PER_SOURCE
        ),

        "candidate_rules": sorted(
            CANDIDATE_RULES
        ),

        "combinations": {
            name: {
                "generic_rules": sorted(
                    rules
                )
            }
            for name, rules in (
                COMBINATIONS.items()
            )
        },

        "representations": {
            name: all_details[
                name
            ]["metrics"]
            for name in COMBINATIONS
        },

        "ranking": ranking,

        "comparison_vs_c7": (
            comparisons
        ),

        "pairwise_comparison": (
            pairwise
        ),

        "e23_logic_verification": {
            "mismatch_count": (
                mismatch_count
            )
        },

        "experimental_note": (
            "E25 keeps the non-generic "
            "section-quality heuristics "
            "from E23 and evaluates all "
            "2^3 combinations of the "
            "three candidate generic rules "
            "identified by E24."
        ),
    }

    # ========================================================
    # Details
    # ========================================================

    details = {
        "summary": summary,
        "representations": all_details,
        "comparison_vs_c7": comparisons,
        "pairwise_comparison": pairwise,
        "ranking": ranking,
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
    # Final comparison
    # ========================================================

    print()
    print("=" * 100)
    print("E25 FINAL COMPARISON")
    print("=" * 100)

    print(
        f"{'Combination':<30}"
        f"{'Rules':>7}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>12}"
        f"{'MRR':>10}"
    )

    print("-" * 100)

    for row in ranking:

        print(
            f"{row['combination']:<30}"
            f"{row['rule_count']:>7}"
            f"{row['hit@1']:>10.4f}"
            f"{row['hit@3']:>10.4f}"
            f"{row['hit@5']:>10.4f}"
            f"{row['hit@10']:>10.4f}"
            f"{row['recall@10']:>12.4f}"
            f"{row['mrr']:>10.4f}"
        )

    # ========================================================
    # C7 comparison
    # ========================================================

    print()
    print("=" * 100)
    print("COMPARISON AGAINST C7 (ALL THREE)")
    print("=" * 100)

    for name, comparison in (
        comparisons.items()
    ):

        print()
        print(name)

        print(
            f"  Improved : "
            f"{comparison['improved']}"
        )

        print(
            f"  Worsened : "
            f"{comparison['worsened']}"
        )

        print(
            f"  Unchanged: "
            f"{comparison['unchanged']}"
        )

        print(
            f"  New hit  : "
            f"{comparison['new_hit']}"
        )

        print(
            f"  New miss : "
            f"{comparison['new_miss']}"
        )

    # ========================================================
    # Files
    # ========================================================

    print()
    print("=" * 100)

    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"Details saved to:\n"
        f"{DETAILS_PATH}"
    )

    print()


if __name__ == "__main__":
    main()