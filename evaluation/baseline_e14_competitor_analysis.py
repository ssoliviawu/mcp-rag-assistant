import json
from pathlib import Path
from collections import Counter
from statistics import mean, median


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e13_vector_error_profile.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e14_competitor_analysis.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def safe_median(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def source_type(url: str) -> str:
    if not url:
        return "unknown"

    u = url.lower()

    if "/migration/" in u:
        return "migration"

    if "/troubleshooting/" in u:
        return "troubleshooting"

    if "/advanced/low-level-server/" in u:
        return "low_level"

    if "/servers/structured-output/" in u:
        return "structured_output"

    if "/run/deploy/" in u:
        return "deployment"

    if "/servers/" in u:
        return "servers"

    if "/clients/" in u:
        return "clients"

    if "/advanced/" in u:
        return "advanced"

    if "/run/" in u:
        return "run"

    return "other"


def compact_case(item):
    top1 = item.get("top1", {})
    gold = item.get("best_gold", {})
    comparison = item.get("comparison", {})

    return {
        "id": item.get("id"),
        "question": item.get("question"),

        "gold_rank": item.get("gold_rank"),

        "top1_source_type": source_type(
            top1.get("url", "")
        ),

        "gold_source_type": source_type(
            gold.get("url", "")
        ),

        "top1_url": top1.get("url"),
        "gold_url": gold.get("url"),

        "top1_title": top1.get("title"),
        "gold_title": gold.get("title"),

        "top1_similarity": top1.get("similarity"),
        "gold_similarity": gold.get("similarity"),

        "semantic_gap": comparison.get(
            "semantic_gap_top1_minus_gold"
        ),

        "lexical_gap": comparison.get(
            "lexical_gap_top1_minus_gold"
        ),

        "top1_code_heavy": top1.get(
            "code_heavy",
            False
        ),

        "top1_code_ratio": top1.get(
            "code_ratio"
        ),

        "top1_content_chars": top1.get(
            "content_chars"
        ),

        "gold_content_chars": gold.get(
            "content_chars"
        ),

        "top1_longer_than_gold": comparison.get(
            "top1_longer_than_gold"
        ),

        "top1_more_code_than_gold": comparison.get(
            "top1_more_code_than_gold"
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"E13 result not found:\n{INPUT_PATH}"
        )

    data = load_json(INPUT_PATH)

    # --------------------------------------------------------
    # IMPORTANT:
    # E13 stores cases under data["questions"]
    # --------------------------------------------------------

    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError(
            "Unexpected E13 format: "
            "'questions' must be a list."
        )

    print("=" * 80)
    print("E14 - VECTOR COMPETITOR ANALYSIS")
    print("=" * 80)

    print(f"Input     : {INPUT_PATH}")
    print(f"Questions : {len(questions)}")

    # ========================================================
    # Containers
    # ========================================================

    cases = [
        compact_case(item)
        for item in questions
    ]

    # Remove unexpected cases without gold rank
    valid_cases = [
        x for x in cases
        if x["gold_rank"] is not None
    ]

    total = len(valid_cases)

    # ========================================================
    # Rank groups
    # ========================================================

    top1_cases = [
        x for x in valid_cases
        if x["gold_rank"] == 1
    ]

    top2_3_cases = [
        x for x in valid_cases
        if 2 <= x["gold_rank"] <= 3
    ]

    rank_4_10 = [
        x for x in valid_cases
        if 4 <= x["gold_rank"] <= 10
    ]

    rank_11_20 = [
        x for x in valid_cases
        if 11 <= x["gold_rank"] <= 20
    ]

    # ========================================================
    # Source type
    # ========================================================

    top1_source_counter = Counter(
        x["top1_source_type"]
        for x in valid_cases
    )

    # ========================================================
    # Code heavy
    # ========================================================

    code_heavy_cases = [
        x for x in valid_cases
        if x["top1_code_heavy"]
    ]

    # ========================================================
    # Migration
    # ========================================================

    migration_cases = [
        x for x in valid_cases
        if x["top1_source_type"] == "migration"
    ]

    migration_code_heavy = [
        x for x in migration_cases
        if x["top1_code_heavy"]
    ]

    # ========================================================
    # Similarity gaps
    # ========================================================

    semantic_gaps = [
        x["semantic_gap"]
        for x in valid_cases
        if x["semantic_gap"] is not None
    ]

    lexical_gaps = [
        x["lexical_gap"]
        for x in valid_cases
        if x["lexical_gap"] is not None
    ]

    near_tie_001 = [
        x for x in valid_cases
        if (
            x["semantic_gap"] is not None
            and x["semantic_gap"] <= 0.01
        )
    ]

    near_tie_003 = [
        x for x in valid_cases
        if (
            x["semantic_gap"] is not None
            and x["semantic_gap"] <= 0.03
        )
    ]

    near_tie_005 = [
        x for x in valid_cases
        if (
            x["semantic_gap"] is not None
            and x["semantic_gap"] <= 0.05
        )
    ]

    # ========================================================
    # Length
    # ========================================================

    length_ratios = []

    for x in valid_cases:

        top1_len = x["top1_content_chars"]
        gold_len = x["gold_content_chars"]

        if (
            top1_len is not None
            and gold_len is not None
            and gold_len > 0
        ):
            length_ratios.append(
                top1_len / gold_len
            )

    # ========================================================
    # Group summary
    # ========================================================

    def group_summary(group):

        if not group:
            return {
                "questions": 0,
                "avg_semantic_gap": None,
                "median_semantic_gap": None,
                "avg_lexical_gap": None,
                "median_lexical_gap": None,
                "code_heavy": 0,
                "code_heavy_rate": 0.0,
                "migration": 0,
                "migration_rate": 0.0,
                "top1_longer_than_gold": 0,
                "top1_more_code_than_gold": 0,
                "avg_length_ratio": None,
                "median_length_ratio": None,
            }

        gaps = [
            x["semantic_gap"]
            for x in group
            if x["semantic_gap"] is not None
        ]

        lex_gaps = [
            x["lexical_gap"]
            for x in group
            if x["lexical_gap"] is not None
        ]

        ratios = []

        for x in group:

            a = x["top1_content_chars"]
            b = x["gold_content_chars"]

            if (
                a is not None
                and b is not None
                and b > 0
            ):
                ratios.append(a / b)

        code_count = sum(
            1 for x in group
            if x["top1_code_heavy"]
        )

        migration_count = sum(
            1 for x in group
            if x["top1_source_type"] == "migration"
        )

        longer_count = sum(
            1 for x in group
            if x["top1_longer_than_gold"]
        )

        more_code_count = sum(
            1 for x in group
            if x["top1_more_code_than_gold"]
        )

        return {
            "questions": len(group),

            "avg_semantic_gap": safe_mean(gaps),
            "median_semantic_gap": safe_median(gaps),

            "avg_lexical_gap": safe_mean(lex_gaps),
            "median_lexical_gap": safe_median(lex_gaps),

            "code_heavy": code_count,
            "code_heavy_rate": (
                code_count / len(group)
            ),

            "migration": migration_count,
            "migration_rate": (
                migration_count / len(group)
            ),

            "top1_longer_than_gold": longer_count,

            "top1_more_code_than_gold": more_code_count,

            "avg_length_ratio": safe_mean(ratios),
            "median_length_ratio": safe_median(ratios),
        }

    # ========================================================
    # Special intersection
    # ========================================================

    migration_and_code_heavy = [
        x for x in valid_cases
        if (
            x["top1_source_type"] == "migration"
            and x["top1_code_heavy"]
        )
    ]

    code_heavy_and_rank_4_10 = [
        x for x in rank_4_10
        if x["top1_code_heavy"]
    ]

    code_heavy_and_rank_11_20 = [
        x for x in rank_11_20
        if x["top1_code_heavy"]
    ]

    migration_rank_4_10 = [
        x for x in rank_4_10
        if x["top1_source_type"] == "migration"
    ]

    migration_rank_11_20 = [
        x for x in rank_11_20
        if x["top1_source_type"] == "migration"
    ]

    # ========================================================
    # Output JSON
    # ========================================================

    output = {
        "experiment": "E14_competitor_analysis",

        "source": str(INPUT_PATH),

        "summary": {
            "questions": total,

            "gold_rank": {
                "average": safe_mean([
                    x["gold_rank"]
                    for x in valid_cases
                ]),
                "median": safe_median([
                    x["gold_rank"]
                    for x in valid_cases
                ]),
                "max": max([
                    x["gold_rank"]
                    for x in valid_cases
                ]) if valid_cases else None,
            },

            "semantic_gap": {
                "average": safe_mean(
                    semantic_gaps
                ),
                "median": safe_median(
                    semantic_gaps
                ),
                "<=0.01": len(
                    near_tie_001
                ),
                "<=0.03": len(
                    near_tie_003
                ),
                "<=0.05": len(
                    near_tie_005
                ),
            },

            "lexical_gap": {
                "average": safe_mean(
                    lexical_gaps
                ),
                "median": safe_median(
                    lexical_gaps
                ),
            },

            "top1_source_types": dict(
                top1_source_counter
            ),

            "code_heavy": {
                "count": len(
                    code_heavy_cases
                ),
                "rate": (
                    len(code_heavy_cases) / total
                    if total else 0.0
                ),
            },

            "migration": {
                "count": len(
                    migration_cases
                ),
                "rate": (
                    len(migration_cases) / total
                    if total else 0.0
                ),
                "code_heavy": len(
                    migration_code_heavy
                ),
            },

            "length": {
                "avg_ratio": safe_mean(
                    length_ratios
                ),
                "median_ratio": safe_median(
                    length_ratios
                ),
                "top1_longer_than_gold": sum(
                    1
                    for x in valid_cases
                    if x["top1_longer_than_gold"]
                ),
                "top1_more_code_than_gold": sum(
                    1
                    for x in valid_cases
                    if x["top1_more_code_than_gold"]
                ),
            },
        },

        "rank_groups": {
            "TOP1": group_summary(top1_cases),
            "TOP2_3": group_summary(top2_3_cases),
            "TOP4_10": group_summary(rank_4_10),
            "TOP11_20": group_summary(rank_11_20),
        },

        "intersections": {
            "migration_and_code_heavy": {
                "count": len(
                    migration_and_code_heavy
                ),
                "rate": (
                    len(migration_and_code_heavy)
                    / total
                    if total else 0.0
                ),
            },

            "code_heavy_rank_4_10": {
                "count": len(
                    code_heavy_and_rank_4_10
                ),
            },

            "code_heavy_rank_11_20": {
                "count": len(
                    code_heavy_and_rank_11_20
                ),
            },

            "migration_rank_4_10": {
                "count": len(
                    migration_rank_4_10
                ),
            },

            "migration_rank_11_20": {
                "count": len(
                    migration_rank_11_20
                ),
            },
        },

        "gold_rank_11_20_cases": [
            x for x in rank_11_20
        ],

        "migration_cases": [
            x for x in migration_cases
        ],

        "code_heavy_cases": [
            x for x in code_heavy_cases
        ],

        "migration_and_code_heavy_cases": [
            x for x in migration_and_code_heavy
        ],

        "small_similarity_gap_cases": [
            x
            for x in sorted(
                near_tie_001,
                key=lambda y: (
                    y["semantic_gap"]
                    if y["semantic_gap"] is not None
                    else 999
                ),
            )
        ],

        "all_cases": valid_cases,
    }

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)
    print("BASIC")
    print("=" * 80)

    print(f"Questions : {total}")

    print()
    print("Gold rank")
    print("----------------------------------------")
    print(
        f"Average : "
        f"{safe_mean([x['gold_rank'] for x in valid_cases]):.2f}"
    )
    print(
        f"Median  : "
        f"{safe_median([x['gold_rank'] for x in valid_cases]):.2f}"
    )
    print(
        f"Max     : "
        f"{max(x['gold_rank'] for x in valid_cases)}"
    )

    print()
    print("Semantic gap")
    print("----------------------------------------")
    print(
        f"Average       : "
        f"{safe_mean(semantic_gaps):.6f}"
    )
    print(
        f"Median        : "
        f"{safe_median(semantic_gaps):.6f}"
    )
    print(
        f"<= 0.01       : "
        f"{len(near_tie_001)}"
    )
    print(
        f"<= 0.03       : "
        f"{len(near_tie_003)}"
    )
    print(
        f"<= 0.05       : "
        f"{len(near_tie_005)}"
    )

    print()
    print("Lexical gap")
    print("----------------------------------------")
    print(
        f"Average       : "
        f"{safe_mean(lexical_gaps):.6f}"
    )
    print(
        f"Median        : "
        f"{safe_median(lexical_gaps):.6f}"
    )

    print()
    print("=" * 80)
    print("TOP1 SOURCE TYPE")
    print("=" * 80)

    for source, count in top1_source_counter.most_common():
        print(
            f"{source:<20}"
            f"{count:>3} "
            f"({count / total:.1%})"
        )

    print()
    print("=" * 80)
    print("CODE-HEAVY")
    print("=" * 80)

    print(
        f"Code-heavy Top1 : "
        f"{len(code_heavy_cases)}/{total} "
        f"({len(code_heavy_cases) / total:.1%})"
    )

    print()
    print("=" * 80)
    print("MIGRATION")
    print("=" * 80)

    print(
        f"Migration Top1 : "
        f"{len(migration_cases)}/{total} "
        f"({len(migration_cases) / total:.1%})"
    )

    print(
        f"Migration + code-heavy : "
        f"{len(migration_and_code_heavy)}"
    )

    print()
    print("=" * 80)
    print("GOLD RANK 4-10 VS 11-20")
    print("=" * 80)

    g1 = group_summary(rank_4_10)
    g2 = group_summary(rank_11_20)

    print(
        f"{'Metric':<30}"
        f"{'Rank 4-10':>15}"
        f"{'Rank 11-20':>15}"
    )

    print("-" * 60)

    print(
        f"{'Questions':<30}"
        f"{g1['questions']:>15}"
        f"{g2['questions']:>15}"
    )

    print(
        f"{'Avg semantic gap':<30}"
        f"{g1['avg_semantic_gap'] or 0:>15.4f}"
        f"{g2['avg_semantic_gap'] or 0:>15.4f}"
    )

    print(
        f"{'Median semantic gap':<30}"
        f"{g1['median_semantic_gap'] or 0:>15.4f}"
        f"{g2['median_semantic_gap'] or 0:>15.4f}"
    )

    print(
        f"{'Code-heavy':<30}"
        f"{g1['code_heavy']:>15}"
        f"{g2['code_heavy']:>15}"
    )

    print(
        f"{'Code-heavy rate':<30}"
        f"{g1['code_heavy_rate']:>14.1%}"
        f"{g2['code_heavy_rate']:>14.1%}"
    )

    print(
        f"{'Migration':<30}"
        f"{g1['migration']:>15}"
        f"{g2['migration']:>15}"
    )

    print(
        f"{'Migration rate':<30}"
        f"{g1['migration_rate']:>14.1%}"
        f"{g2['migration_rate']:>14.1%}"
    )

    print(
        f"{'Median length ratio':<30}"
        f"{g1['median_length_ratio'] or 0:>15.2f}"
        f"{g2['median_length_ratio'] or 0:>15.2f}"
    )

    print()
    print("=" * 80)
    print("GOLD RANK 11-20 CASES")
    print("=" * 80)

    if not rank_11_20:
        print("None")
    else:
        for x in rank_11_20:
            print()
            print(
                f"{x['id']} | Gold rank {x['gold_rank']}"
            )
            print(
                f"Question : {x['question']}"
            )
            print(
                f"Top1    : "
                f"{x['top1_source_type']}"
            )
            print(
                f"Top1 URL: "
                f"{x['top1_url']}"
            )
            print(
                f"Gold    : "
                f"{x['gold_url']}"
            )
            print(
                f"Semantic: "
                f"{x['semantic_gap']:.6f}"
                if x["semantic_gap"] is not None
                else "Semantic: N/A"
            )
            print(
                f"Lexical : "
                f"{x['lexical_gap']:.6f}"
                if x["lexical_gap"] is not None
                else "Lexical : N/A"
            )
            print(
                f"Code    : "
                f"{x['top1_code_heavy']}"
            )

    print()
    print("=" * 80)
    print("MIGRATION CASES")
    print("=" * 80)

    for x in migration_cases:
        print(
            f"{x['id']:<6} "
            f"gold_rank={x['gold_rank']:<3} "
            f"gap={x['semantic_gap']:.4f}"
        )

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()