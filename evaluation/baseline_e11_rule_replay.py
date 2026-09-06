from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

E6_PATH = RESULTS_DIR / "e6_lexical_alpha_details.json"
E8_PATH = RESULTS_DIR / "e8_e6_vs_e2_analysis.json"

OUTPUT_PATH = RESULTS_DIR / "e11_rule_replay_summary.json"


# ============================================================
# Fixed E6 configuration
# ============================================================

ALPHA = 0.05

SEMANTIC_GAP_THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
]

LEXICAL_GAP_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
]


# ============================================================
# Load JSON
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Build E6_05 question map
# ============================================================

def load_e6_05(e6_data):
    """
    Expected structure:

    {
        "results": {
            "E6_05": [
                {
                    "id": "q001",
                    ...
                    "retrieved_details": [
                        {
                            "rank": 1,
                            "id": "...",
                            "url": "...",
                            "similarity": ...,
                            "lexical_score": ...
                        }
                    ]
                }
            ]
        }
    }
    """

    results = e6_data.get("results")

    if not isinstance(results, dict):
        raise ValueError(
            "E6 JSON does not contain "
            "'results' dict."
        )

    if "E6_05" not in results:
        raise ValueError(
            "E6 JSON does not contain results['E6_05']."
        )

    records = results["E6_05"]

    if not isinstance(records, list):
        raise ValueError(
            "results['E6_05'] is not a list."
        )

    question_map = {}

    for record in records:
        qid = record.get("id")

        if not qid:
            continue

        question_map[qid] = record

    return question_map


# ============================================================
# Load E8 analysis
# ============================================================

def load_e8(e8_data):
    """
    Expected structure:

    {
        "questions": [
            {
                "id": "q024",
                "e2": {...},
                "e6": {...},
                "candidate_changes": {...},
                ...
            }
        ]
    }
    """

    questions = e8_data.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "E8 JSON does not contain "
            "'questions' list."
        )

    question_map = {}

    for record in questions:
        qid = record.get("id")

        if not qid:
            continue

        question_map[qid] = record

    return question_map


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    """
    Normalize URLs for source-level comparison.

    The E6 and E8 files may differ only by a trailing slash.
    """
    if not url:
        return ""

    return str(url).strip().rstrip("/")


def get_gold_sources(e8_record):
    sources = e8_record.get("gold_sources", [])

    if not isinstance(sources, list):
        return set()

    return {
        normalize_url(source)
        for source in sources
        if source
    }


def is_gold(item, gold_sources):
    url = normalize_url(
        item.get("url", "")
    )

    return url in gold_sources


def calculate_e6_score(item):
    similarity = item.get("similarity")
    lexical = item.get("lexical_score")

    if similarity is None:
        raise ValueError(
            f"Missing similarity in candidate: {item}"
        )

    if lexical is None:
        raise ValueError(
            f"Missing lexical_score in candidate: {item}"
        )

    return (
        similarity
        + ALPHA * lexical
    )


def get_gold_rank(items, gold_sources):
    for rank, item in enumerate(items, start=1):
        if is_gold(item, gold_sources):
            return rank

    return None


# ============================================================
# E11 replay rule
# ============================================================

def replay_question(
    retrieved_details,
    gold_sources,
    semantic_gap_threshold,
    lexical_gap_threshold,
):
    """
    Conditional lexical boost.

    Baseline:
        E6 score = similarity + 0.05 * lexical_score

    E11:
        Compare adjacent candidates in the original semantic
        ranking.

        Lexical boost is allowed only when:

            semantic_gap <= threshold
            AND
            lexical_gap >= threshold

        where:

            semantic_gap =
                previous_similarity - current_similarity

            lexical_gap =
                current_lexical - previous_lexical

    If the condition is not satisfied:
        keep semantic similarity only.

    This is intentionally a conservative local correction.
    """

    # E6 retrieved_details are already ranked.
    candidates = sorted(
        retrieved_details,
        key=lambda x: x["rank"],
    )

    replay_items = []

    for index, item in enumerate(candidates):

        similarity = item["similarity"]
        lexical = item["lexical_score"]

        apply_lexical = False
        semantic_gap = None
        lexical_gap = None
        replay_score = similarity

        # ----------------------------------------------------
        # Compare against immediately preceding candidate
        # ----------------------------------------------------

        if index > 0:
            previous = candidates[index - 1]

            previous_similarity = previous["similarity"]
            previous_lexical = previous["lexical_score"]

            semantic_gap = (
                previous_similarity
                - similarity
            )

            lexical_gap = (
                lexical
                - previous_lexical
            )

            if (
                semantic_gap >= 0
                and semantic_gap <= semantic_gap_threshold
                and lexical_gap >= lexical_gap_threshold
            ):
                apply_lexical = True

                replay_score = (
                    similarity
                    + ALPHA * lexical
                )

        replay_items.append(
            {
                **item,
                "original_rank": item["rank"],
                "semantic_gap": semantic_gap,
                "lexical_gap": lexical_gap,
                "apply_lexical": apply_lexical,
                "replay_score": replay_score,
                "is_gold": is_gold(
                    item,
                    gold_sources,
                ),
            }
        )

    # --------------------------------------------------------
    # Sort according to E11 replay score
    # --------------------------------------------------------

    replay_items.sort(
        key=lambda x: x["replay_score"],
        reverse=True,
    )

    for new_rank, item in enumerate(
        replay_items,
        start=1,
    ):
        item["replay_rank"] = new_rank

    return replay_items


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E11 - CONDITIONAL LEXICAL RULE REPLAY")
    print("=" * 80)

    print()
    print(f"E6 input : {E6_PATH}")
    print(f"E8 input : {E8_PATH}")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    e6_data = load_json(E6_PATH)
    e8_data = load_json(E8_PATH)

    e6_questions = load_e6_05(e6_data)
    e8_questions = load_e8(e8_data)

    common_ids = sorted(
        set(e6_questions)
        & set(e8_questions)
    )

    print(
        f"E6_05 questions : {len(e6_questions)}"
    )

    print(
        f"E8 questions    : {len(e8_questions)}"
    )

    print(
        f"Common questions: {len(common_ids)}"
    )

    if not common_ids:
        raise RuntimeError(
            "No common questions between E6_05 and E8."
        )

    print()

    # --------------------------------------------------------
    # Verify E6 baseline
    # --------------------------------------------------------

    e6_baseline = {}

    for qid in common_ids:

        e6_record = e6_questions[qid]
        e8_record = e8_questions[qid]

        details = e6_record.get(
            "retrieved_details"
        )

        if not isinstance(details, list):
            raise ValueError(
                f"{qid}: retrieved_details is missing "
                f"or is not a list."
            )

        gold_sources = get_gold_sources(
            e8_record
        )

        gold_rank = get_gold_rank(
            details,
            gold_sources,
        )

        e6_info = e8_record.get(
            "e6",
            {}
        )

        expected_rank = e6_info.get(
            "gold_rank"
        )

        if (
            expected_rank is not None
            and gold_rank != expected_rank
        ):
            raise ValueError(
                f"{qid}: E6 rank mismatch.\n"
                f"  E6 details rank : {gold_rank}\n"
                f"  E8 recorded rank: {expected_rank}"
            )

        e6_baseline[qid] = {
            "gold_rank": gold_rank,
            "hit@10": (
                gold_rank is not None
                and gold_rank <= 10
            ),
        }

    print("E6_05 baseline verification: PASSED")
    print()

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    configurations = []

    for semantic_threshold in (
        SEMANTIC_GAP_THRESHOLDS
    ):

        for lexical_threshold in (
            LEXICAL_GAP_THRESHOLDS
        ):

            improved = 0
            worsened = 0
            unchanged = 0

            hit10 = 0

            total_boosted = 0
            questions_with_boost = 0

            question_details = []

            for qid in common_ids:

                e6_record = e6_questions[qid]
                e8_record = e8_questions[qid]

                details = e6_record[
                    "retrieved_details"
                ]

                gold_sources = get_gold_sources(
                    e8_record
                )

                baseline_rank = e6_baseline[
                    qid
                ]["gold_rank"]

                replayed = replay_question(
                    details,
                    gold_sources,
                    semantic_threshold,
                    lexical_threshold,
                )

                replay_rank = get_gold_rank(
                    replayed,
                    gold_sources,
                )

                boosted = [
                    x
                    for x in replayed
                    if x["apply_lexical"]
                ]

                total_boosted += len(
                    boosted
                )

                if boosted:
                    questions_with_boost += 1

                if (
                    baseline_rank is not None
                    and replay_rank is not None
                ):
                    if replay_rank < baseline_rank:
                        improved += 1

                    elif replay_rank > baseline_rank:
                        worsened += 1

                    else:
                        unchanged += 1

                if (
                    replay_rank is not None
                    and replay_rank <= 10
                ):
                    hit10 += 1

                question_details.append(
                    {
                        "id": qid,
                        "e6_gold_rank": baseline_rank,
                        "e11_gold_rank": replay_rank,
                        "rank_change": (
                            None
                            if (
                                baseline_rank is None
                                or replay_rank is None
                            )
                            else (
                                baseline_rank
                                - replay_rank
                            )
                        ),
                        "boosted_candidates": [
                            {
                                "id": x["id"],
                                "url": x["url"],
                                "original_rank": x[
                                    "original_rank"
                                ],
                                "replay_rank": x[
                                    "replay_rank"
                                ],
                                "similarity": x[
                                    "similarity"
                                ],
                                "lexical_score": x[
                                    "lexical_score"
                                ],
                                "semantic_gap": x[
                                    "semantic_gap"
                                ],
                                "lexical_gap": x[
                                    "lexical_gap"
                                ],
                                "is_gold": x[
                                    "is_gold"
                                ],
                            }
                            for x in boosted
                        ],
                    }
                )

            configuration = {
                "semantic_gap_threshold": (
                    semantic_threshold
                ),
                "lexical_gap_threshold": (
                    lexical_threshold
                ),
                "alpha": ALPHA,
                "questions": len(common_ids),
                "hit@10": (
                    hit10 / len(common_ids)
                ),
                "hits@10": hit10,
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "net_rank_change": (
                    improved - worsened
                ),
                "questions_with_boost": (
                    questions_with_boost
                ),
                "total_boosted_candidates": (
                    total_boosted
                ),
                "details": question_details,
            }

            configurations.append(
                configuration
            )

    # --------------------------------------------------------
    # Sort
    #
    # Primary:
    #   fewer regressions
    #
    # Secondary:
    #   more improvements
    #
    # Tertiary:
    #   higher Hit@10
    # --------------------------------------------------------

    configurations.sort(
        key=lambda x: (
            -x["worsened"],
            -x["improved"],
            -x["hit@10"],
        )
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("=" * 80)
    print("E11 RESULTS")
    print("=" * 80)

    print()

    print(
        f"{'SemGap':>8} "
        f"{'LexGap':>8} "
        f"{'Hit@10':>8} "
        f"{'Improve':>8} "
        f"{'Worsen':>8} "
        f"{'Same':>8} "
        f"{'BoostQ':>8}"
    )

    print("-" * 80)

    for item in configurations:

        print(
            f"{item['semantic_gap_threshold']:>8.2f} "
            f"{item['lexical_gap_threshold']:>8.2f} "
            f"{item['hit@10']:>8.4f} "
            f"{item['improved']:>8} "
            f"{item['worsened']:>8} "
            f"{item['unchanged']:>8} "
            f"{item['questions_with_boost']:>8}"
        )

    # --------------------------------------------------------
    # Compare with E6
    # --------------------------------------------------------

    e6_improved = 4
    e6_worsened = 9
    e6_unchanged = 35

    print()
    print("=" * 80)
    print("E6 BASELINE")
    print("=" * 80)

    print(
        f"Improve : {e6_improved}"
    )

    print(
        f"Worsen  : {e6_worsened}"
    )

    print(
        f"Same    : {e6_unchanged}"
    )

    print(
        "Hit@10  : 1.0000"
    )

    print()
    print("=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    for index, item in enumerate(
        configurations[:5],
        start=1,
    ):

        print(
            f"{index}. "
            f"semantic_gap <= "
            f"{item['semantic_gap_threshold']:.2f}, "
            f"lexical_gap >= "
            f"{item['lexical_gap_threshold']:.2f} | "
            f"Hit@10={item['hit@10']:.4f}, "
            f"Improve={item['improved']}, "
            f"Worsen={item['worsened']}, "
            f"Same={item['unchanged']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "experiment": "E11_rule_replay",
        "description": (
            "Offline counterfactual replay of conditional "
            "lexical boosting on E6_05 results."
        ),
        "baseline": {
            "experiment": "E6_05",
            "alpha": ALPHA,
            "questions": len(common_ids),
            "hit@10": 1.0,
            "improved": e6_improved,
            "worsened": e6_worsened,
            "unchanged": e6_unchanged,
        },
        "rule": {
            "semantic_condition": (
                "semantic_gap <= semantic_gap_threshold"
            ),
            "lexical_condition": (
                "lexical_gap >= lexical_gap_threshold"
            ),
            "boost": (
                "similarity + alpha * lexical_score"
            ),
            "otherwise": (
                "similarity"
            ),
        },
        "thresholds": {
            "semantic_gap": (
                SEMANTIC_GAP_THRESHOLDS
            ),
            "lexical_gap": (
                LEXICAL_GAP_THRESHOLDS
            ),
        },
        "questions": len(common_ids),
        "configurations": configurations,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()