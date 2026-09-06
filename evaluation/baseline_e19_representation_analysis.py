import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e18_representation_details.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e19_representation_analysis.json"
)


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    if not url:
        return ""

    return url.rstrip("/")


def is_gold(result, relevant_sources):
    return (
        normalize_url(result.get("url"))
        in relevant_sources
    )


def get_first_gold_rank(item):
    return item.get("metrics", {}).get(
        "first_relevant_rank"
    )


def get_top1(item):
    top10 = item.get("top10", [])

    if not top10:
        return None

    return top10[0]


def get_gold_results(item):
    relevant_sources = {
        normalize_url(url)
        for url in item.get(
            "relevant_sources",
            []
        )
    }

    return [
        result
        for result in item.get("top10", [])
        if is_gold(
            result,
            relevant_sources,
        )
    ]


def source_type(url):
    url = normalize_url(url)

    if "/migration" in url:
        return "migration"

    if "/servers/" in url:
        return "servers"

    if "/handlers/" in url:
        return "handlers"

    if "/advanced/low-level-server" in url:
        return "low_level"

    if "/advanced/" in url:
        return "advanced"

    if "/structured-output" in url:
        return "structured_output"

    if "/troubleshooting" in url:
        return "troubleshooting"

    if "/run/" in url:
        return "run"

    if "/client/" in url:
        return "client"

    return "other"


def rank_delta(
    baseline_rank,
    new_rank,
):
    """
    Positive = improved
    Negative = worsened
    Zero = unchanged
    """
    if (
        baseline_rank is None
        and new_rank is None
    ):
        return 0

    if baseline_rank is None:
        return 999

    if new_rank is None:
        return -999

    return baseline_rank - new_rank


def classify_change(
    baseline_rank,
    new_rank,
):
    if (
        baseline_rank is None
        and new_rank is None
    ):
        return "BOTH_MISS"

    if baseline_rank is None:
        return "NEW_HIT"

    if new_rank is None:
        return "NEW_MISS"

    if new_rank < baseline_rank:
        return "IMPROVED"

    if new_rank > baseline_rank:
        return "WORSENED"

    return "UNCHANGED"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E19 - E18 REPRESENTATION ANALYSIS")
    print("=" * 80)

    print()
    print(f"Input : {INPUT_PATH}")

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    questions = data["questions"]

    modes = [
        "content",
        "title_content",
        "section_content",
        "title_section_content",
    ]

    print(
        f"Questions : {len(questions['content'])}"
    )

    # --------------------------------------------------------
    # Convert lists to question-id maps
    # --------------------------------------------------------

    by_mode = {}

    for mode in modes:

        by_mode[mode] = {
            item["id"]: item
            for item in questions[mode]
        }

    question_ids = list(
        by_mode["content"].keys()
    )

    # ========================================================
    # 1. Overall comparison
    # ========================================================

    overall = {}

    for mode in modes:

        items = [
            by_mode[mode][qid]
            for qid in question_ids
        ]

        n = len(items)

        hit1 = sum(
            item["metrics"]["hit@1"]
            for item in items
        ) / n

        hit3 = sum(
            item["metrics"]["hit@3"]
            for item in items
        ) / n

        hit5 = sum(
            item["metrics"]["hit@5"]
            for item in items
        ) / n

        hit10 = sum(
            item["metrics"]["hit@10"]
            for item in items
        ) / n

        recall10 = sum(
            item["metrics"]["recall@10"]
            for item in items
        ) / n

        mrr = sum(
            item["metrics"]["mrr"]
            for item in items
        ) / n

        overall[mode] = {
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "hit@10": hit10,
            "recall@10": recall10,
            "mrr": mrr,
        }

    # ========================================================
    # 2. Content -> Title+Section+Content
    # ========================================================

    baseline_mode = "content"
    best_mode = "title_section_content"

    changes = []

    improved = []
    worsened = []
    unchanged = []

    for qid in question_ids:

        base = by_mode[baseline_mode][qid]
        new = by_mode[best_mode][qid]

        base_rank = get_first_gold_rank(base)
        new_rank = get_first_gold_rank(new)

        change = classify_change(
            base_rank,
            new_rank,
        )

        delta = rank_delta(
            base_rank,
            new_rank,
        )

        record = {
            "id": qid,
            "question": base["question"],
            "baseline_rank": base_rank,
            "new_rank": new_rank,
            "rank_delta": delta,
            "change": change,
        }

        changes.append(record)

        if change in {
            "IMPROVED",
            "NEW_HIT",
        }:
            improved.append(record)

        elif change in {
            "WORSENED",
            "NEW_MISS",
        }:
            worsened.append(record)

        else:
            unchanged.append(record)

    # ========================================================
    # 3. Top1 transition analysis
    # ========================================================

    top1_transitions = {
        "non_gold_to_gold": [],
        "gold_to_non_gold": [],
        "gold_to_gold": [],
        "non_gold_to_non_gold": [],
    }

    for qid in question_ids:

        base = by_mode[baseline_mode][qid]
        new = by_mode[best_mode][qid]

        relevant_sources = {
            normalize_url(url)
            for url in base.get(
                "relevant_sources",
                [],
            )
        }

        base_top1 = get_top1(base)
        new_top1 = get_top1(new)

        base_is_gold = (
            base_top1 is not None
            and is_gold(
                base_top1,
                relevant_sources,
            )
        )

        new_is_gold = (
            new_top1 is not None
            and is_gold(
                new_top1,
                relevant_sources,
            )
        )

        record = {
            "id": qid,
            "question": base["question"],
            "baseline_top1": (
                base_top1
            ),
            "new_top1": (
                new_top1
            ),
        }

        if (
            not base_is_gold
            and new_is_gold
        ):
            top1_transitions[
                "non_gold_to_gold"
            ].append(record)

        elif (
            base_is_gold
            and not new_is_gold
        ):
            top1_transitions[
                "gold_to_non_gold"
            ].append(record)

        elif (
            base_is_gold
            and new_is_gold
        ):
            top1_transitions[
                "gold_to_gold"
            ].append(record)

        else:
            top1_transitions[
                "non_gold_to_non_gold"
            ].append(record)

    # ========================================================
    # 4. Source-type analysis
    # ========================================================

    source_analysis = {
        "baseline_top1": {},
        "new_top1": {},
    }

    for mode, target in [
        (
            baseline_mode,
            "baseline_top1",
        ),
        (
            best_mode,
            "new_top1",
        ),
    ]:

        counts = {}

        for qid in question_ids:

            item = by_mode[mode][qid]

            top1 = get_top1(item)

            if not top1:
                continue

            stype = source_type(
                top1.get("url")
            )

            counts[stype] = (
                counts.get(stype, 0) + 1
            )

        source_analysis[target] = counts

    # ========================================================
    # 5. Migration analysis
    # ========================================================

    migration_cases = []

    for qid in question_ids:

        base = by_mode[
            baseline_mode
        ][qid]

        new = by_mode[
            best_mode
        ][qid]

        base_top1 = get_top1(base)
        new_top1 = get_top1(new)

        base_type = (
            source_type(
                base_top1.get("url")
            )
            if base_top1
            else None
        )

        new_type = (
            source_type(
                new_top1.get("url")
            )
            if new_top1
            else None
        )

        if (
            base_type == "migration"
            or new_type == "migration"
        ):
            migration_cases.append(
                {
                    "id": qid,
                    "question": base["question"],
                    "baseline_rank": get_first_gold_rank(
                        base
                    ),
                    "new_rank": get_first_gold_rank(
                        new
                    ),
                    "baseline_top1_source": (
                        base_type
                    ),
                    "new_top1_source": (
                        new_type
                    ),
                    "baseline_top1": base_top1,
                    "new_top1": new_top1,
                }
            )

    # ========================================================
    # 6. Semantic rank movement
    # ========================================================

    rank_movement = []

    for qid in question_ids:

        base = by_mode[
            baseline_mode
        ][qid]

        new = by_mode[
            best_mode
        ][qid]

        base_rank = get_first_gold_rank(
            base
        )

        new_rank = get_first_gold_rank(
            new
        )

        rank_movement.append(
            {
                "id": qid,
                "question": base["question"],
                "baseline_rank": base_rank,
                "new_rank": new_rank,
                "delta": rank_delta(
                    base_rank,
                    new_rank,
                ),
            }
        )

    # Sort strongest improvements first

    strongest_improvements = sorted(
        [
            x
            for x in rank_movement
            if x["delta"] > 0
            and x["delta"] != 999
        ],
        key=lambda x: x["delta"],
        reverse=True,
    )

    strongest_regressions = sorted(
        [
            x
            for x in rank_movement
            if x["delta"] < 0
            and x["delta"] != -999
        ],
        key=lambda x: x["delta"],
    )

    # ========================================================
    # 7. Representation-specific comparison
    # ========================================================

    pairwise = {}

    comparison_pairs = [
        (
            "title_content",
            "content",
        ),
        (
            "section_content",
            "content",
        ),
        (
            "title_section_content",
            "content",
        ),
        (
            "title_section_content",
            "title_content",
        ),
        (
            "title_section_content",
            "section_content",
        ),
    ]

    for new_mode, base_mode in comparison_pairs:

        pair = {
            "improved": 0,
            "worsened": 0,
            "unchanged": 0,
            "changes": [],
        }

        for qid in question_ids:

            base = by_mode[
                base_mode
            ][qid]

            new = by_mode[
                new_mode
            ][qid]

            base_rank = get_first_gold_rank(
                base
            )

            new_rank = get_first_gold_rank(
                new
            )

            change = classify_change(
                base_rank,
                new_rank,
            )

            delta = rank_delta(
                base_rank,
                new_rank,
            )

            if change in {
                "IMPROVED",
                "NEW_HIT",
            }:
                pair["improved"] += 1

            elif change in {
                "WORSENED",
                "NEW_MISS",
            }:
                pair["worsened"] += 1

            else:
                pair["unchanged"] += 1

            if change != "UNCHANGED":
                pair["changes"].append(
                    {
                        "id": qid,
                        "question": base[
                            "question"
                        ],
                        "baseline_rank": base_rank,
                        "new_rank": new_rank,
                        "delta": delta,
                        "change": change,
                    }
                )

        pairwise[
            f"{new_mode}_vs_{base_mode}"
        ] = pair

    # ========================================================
    # 8. Important cases
    # ========================================================

    important_cases = []

    for qid in question_ids:

        base = by_mode[
            baseline_mode
        ][qid]

        new = by_mode[
            best_mode
        ][qid]

        base_rank = get_first_gold_rank(
            base
        )

        new_rank = get_first_gold_rank(
            new
        )

        base_top1 = get_top1(base)
        new_top1 = get_top1(new)

        delta = rank_delta(
            base_rank,
            new_rank,
        )

        # Include:
        # - large improvement
        # - large regression
        # - baseline Top1 miss -> new Top1 hit
        # - baseline Top1 hit -> new Top1 miss

        relevant_sources = {
            normalize_url(url)
            for url in base.get(
                "relevant_sources",
                [],
            )
        }

        base_top1_gold = (
            base_top1 is not None
            and is_gold(
                base_top1,
                relevant_sources,
            )
        )

        new_top1_gold = (
            new_top1 is not None
            and is_gold(
                new_top1,
                relevant_sources,
            )
        )

        if (
            abs(delta) >= 3
            or (
                not base_top1_gold
                and new_top1_gold
            )
            or (
                base_top1_gold
                and not new_top1_gold
            )
        ):

            important_cases.append(
                {
                    "id": qid,
                    "question": base[
                        "question"
                    ],
                    "baseline_rank": base_rank,
                    "new_rank": new_rank,
                    "delta": delta,
                    "baseline_top1": base_top1,
                    "new_top1": new_top1,
                    "baseline_top1_gold": (
                        base_top1_gold
                    ),
                    "new_top1_gold": (
                        new_top1_gold
                    ),
                }
            )

    # ========================================================
    # 9. Final result
    # ========================================================

    result = {
        "experiment": (
            "E19_representation_analysis"
        ),
        "source": str(INPUT_PATH),
        "question_count": len(
            question_ids
        ),

        "baseline": baseline_mode,
        "best_representation": best_mode,

        "overall": overall,

        "rank_change": {
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
        },

        "top1_transition": {
            key: len(value)
            for key, value
            in top1_transitions.items()
        },

        "source_analysis": source_analysis,

        "migration_cases": migration_cases,

        "strongest_improvements": (
            strongest_improvements
        ),

        "strongest_regressions": (
            strongest_regressions
        ),

        "pairwise": pairwise,

        "important_cases": important_cases,

        "details": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "top1_transitions": (
                top1_transitions
            ),
        },
    }

    # ========================================================
    # Save
    # ========================================================

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 80)
    print("E19 SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'Representation':<25}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 80)

    for mode in modes:

        s = overall[mode]

        print(
            f"{mode:<25}"
            f"{s['hit@1']:>10.4f}"
            f"{s['hit@3']:>10.4f}"
            f"{s['hit@5']:>10.4f}"
            f"{s['hit@10']:>10.4f}"
            f"{s['mrr']:>10.4f}"
        )

    print()

    print(
        "CONTENT -> TITLE + SECTION + CONTENT"
    )

    print(
        f"Improved : {len(improved)}"
    )

    print(
        f"Worsened  : {len(worsened)}"
    )

    print(
        f"Unchanged : {len(unchanged)}"
    )

    print()

    print(
        "TOP1 TRANSITION"
    )

    print(
        f"Non-gold -> Gold : "
        f"{len(top1_transitions['non_gold_to_gold'])}"
    )

    print(
        f"Gold -> Non-gold : "
        f"{len(top1_transitions['gold_to_non_gold'])}"
    )

    print(
        f"Gold -> Gold     : "
        f"{len(top1_transitions['gold_to_gold'])}"
    )

    print(
        f"Non-gold -> Non-gold : "
        f"{len(top1_transitions['non_gold_to_non_gold'])}"
    )

    print()

    print(
        "SOURCE TYPE - TOP1"
    )

    print()

    print("Content:")

    for source, count in sorted(
        source_analysis[
            "baseline_top1"
        ].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {source:<25} {count}"
        )

    print()

    print(
        "Title + Section + Content:"
    )

    for source, count in sorted(
        source_analysis[
            "new_top1"
        ].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {source:<25} {count}"
        )

    print()

    print(
        "MIGRATION CASES"
    )

    print(
        f"Cases involving migration Top1: "
        f"{len(migration_cases)}"
    )

    print()

    print(
        "STRONGEST IMPROVEMENTS"
    )

    for item in strongest_improvements[:10]:

        print(
            f"{item['id']} | "
            f"{item['baseline_rank']} -> "
            f"{item['new_rank']} | "
            f"{item['question']}"
        )

    print()

    print(
        "STRONGEST REGRESSIONS"
    )

    for item in strongest_regressions[:10]:

        print(
            f"{item['id']} | "
            f"{item['baseline_rank']} -> "
            f"{item['new_rank']} | "
            f"{item['question']}"
        )

    print()

    print(
        "IMPORTANT CASES"
    )

    print(
        f"Count: {len(important_cases)}"
    )

    for item in important_cases:

        print(
            f"{item['id']} | "
            f"{item['baseline_rank']} -> "
            f"{item['new_rank']} | "
            f"{item['question']}"
        )

    print()

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()