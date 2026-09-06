import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

INPUT_PATH = RESULTS_DIR / "e8_e6_vs_e2_analysis.json"
OUTPUT_PATH = RESULTS_DIR / "e9_e6_diagnosis.json"


ALPHA = 0.05


# ============================================================
# Load
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Helpers
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_improvement(item):
    """
    Diagnose cases where gold rank improved.
    """

    e6 = item["e6"]
    top1 = e6.get("top1")
    gold_results = e6.get("gold_results", [])

    if not gold_results:
        return "NO_GOLD_DETAIL"

    gold = gold_results[0]

    gold_similarity = safe_float(
        gold.get("similarity")
    )

    gold_lexical = safe_float(
        gold.get("lexical_score")
    )

    gold_final = safe_float(
        gold.get("final_score")
    )

    # Find E6 top result if it is not gold.
    competitor = None

    if top1 and not top1.get("is_gold"):
        competitor = top1

    if competitor is None:
        return "GOLD_ALREADY_TOP1"

    competitor_similarity = safe_float(
        competitor.get("similarity")
    )

    competitor_lexical = safe_float(
        competitor.get("lexical_score")
    )

    if (
        gold_similarity is None
        or gold_lexical is None
        or competitor_similarity is None
        or competitor_lexical is None
    ):
        return "INSUFFICIENT_SCORE_DATA"

    semantic_gap = (
        competitor_similarity
        - gold_similarity
    )

    lexical_gap = (
        gold_lexical
        - competitor_lexical
    )

    gold_score = (
        gold_similarity
        + ALPHA * gold_lexical
    )

    competitor_score = (
        competitor_similarity
        + ALPHA * competitor_lexical
    )

    # Vector prefers competitor,
    # lexical boost makes gold win.
    if (
        semantic_gap > 0
        and lexical_gap > 0
        and gold_score > competitor_score
    ):
        return "TRUE_LEXICAL_RESCUE"

    # Both semantic and lexical favor competitor.
    if (
        semantic_gap > 0
        and lexical_gap <= 0
    ):
        return "LEXICAL_CANNOT_HELP"

    return "OTHER_IMPROVEMENT"


def classify_worsening(item):
    """
    Diagnose cases where gold rank worsened.
    """

    e2 = item["e2"]
    e6 = item["e6"]

    gold_results = e6.get("gold_results", [])

    if not gold_results:
        return "NO_GOLD_DETAIL"

    gold = gold_results[0]

    gold_similarity = safe_float(
        gold.get("similarity")
    )

    gold_lexical = safe_float(
        gold.get("lexical_score")
    )

    top1 = e6.get("top1")

    if not top1:
        return "NO_E6_TOP1"

    # If E6 top1 is actually gold, this shouldn't be a worsening.
    if top1.get("is_gold"):
        return "INCONSISTENT"

    competitor_similarity = safe_float(
        top1.get("similarity")
    )

    competitor_lexical = safe_float(
        top1.get("lexical_score")
    )

    if (
        gold_similarity is None
        or gold_lexical is None
        or competitor_similarity is None
        or competitor_lexical is None
    ):
        return "INSUFFICIENT_SCORE_DATA"

    semantic_gap = (
        competitor_similarity
        - gold_similarity
    )

    lexical_gap = (
        competitor_lexical
        - gold_lexical
    )

    gold_score = (
        gold_similarity
        + ALPHA * gold_lexical
    )

    competitor_score = (
        competitor_similarity
        + ALPHA * competitor_lexical
    )

    # The competitor wins both semantic and lexical.
    if (
        semantic_gap > 0
        and lexical_gap > 0
    ):
        return "SEMANTIC_AND_LEXICAL_FAVOR_COMPETITOR"

    # Lexical boost specifically favors competitor.
    if (
        semantic_gap > 0
        and lexical_gap > 0
        and competitor_score > gold_score
    ):
        return "TRUE_LEXICAL_REGRESSION"

    # Gold has higher lexical overlap,
    # but semantic gap is too large.
    if (
        semantic_gap > 0
        and gold_lexical > competitor_lexical
        and competitor_score > gold_score
    ):
        return "SEMANTIC_GAP_TOO_LARGE"

    # Competitor has same lexical score.
    if (
        abs(
            competitor_lexical
            - gold_lexical
        )
        < 1e-9
    ):
        return "LEXICAL_NO_DISCRIMINATION"

    return "OTHER_REGRESSION"


def analyze_case(item):
    e2_rank = item["e2"].get(
        "gold_rank"
    )

    e6_rank = item["e6"].get(
        "gold_rank"
    )

    rank_change = item.get(
        "rank_change"
    )

    classification = item.get(
        "classification"
    )

    if classification == "GOLD_RANK_IMPROVED":

        diagnosis = classify_improvement(
            item
        )

    elif classification == "GOLD_RANK_WORSENED":

        diagnosis = classify_worsening(
            item
        )

    else:

        diagnosis = "NOT_RANK_CHANGED"

    # --------------------------------------------------------
    # Score details
    # --------------------------------------------------------

    gold_results = item["e6"].get(
        "gold_results",
        []
    )

    gold = (
        gold_results[0]
        if gold_results
        else {}
    )

    top1 = item["e6"].get(
        "top1"
    ) or {}

    gold_similarity = safe_float(
        gold.get("similarity")
    )

    gold_lexical = safe_float(
        gold.get("lexical_score")
    )

    top1_similarity = safe_float(
        top1.get("similarity")
    )

    top1_lexical = safe_float(
        top1.get("lexical_score")
    )

    semantic_gap = None
    lexical_gap = None

    if (
        gold_similarity is not None
        and top1_similarity is not None
    ):
        semantic_gap = (
            top1_similarity
            - gold_similarity
        )

    if (
        gold_lexical is not None
        and top1_lexical is not None
    ):
        lexical_gap = (
            gold_lexical
            - top1_lexical
        )

    return {
        "id": item["id"],
        "question": item["question"],

        "classification": classification,

        "diagnosis": diagnosis,

        "e2_gold_rank": e2_rank,
        "e6_gold_rank": e6_rank,

        "rank_change": rank_change,

        "e2_top1": (
            item["e2"]
            .get("top1")
        ),

        "e6_top1": top1,

        "gold": {
            "similarity": gold_similarity,
            "lexical_score": gold_lexical,
            "final_score": safe_float(
                gold.get(
                    "final_score"
                )
            ),
        },

        "top1_competitor": {
            "similarity": top1_similarity,
            "lexical_score": top1_lexical,
            "final_score": safe_float(
                top1.get(
                    "final_score"
                )
            ),
        },

        "semantic_gap": semantic_gap,

        "lexical_gap": lexical_gap,

        "candidate_set": item[
            "candidate_changes"
        ],
    }


# ============================================================
# Summary
# ============================================================

def build_summary(cases):

    rank_changed = [
        x
        for x in cases
        if x["classification"]
        in (
            "GOLD_RANK_IMPROVED",
            "GOLD_RANK_WORSENED",
        )
    ]

    improved = [
        x
        for x in cases
        if x["classification"]
        == "GOLD_RANK_IMPROVED"
    ]

    worsened = [
        x
        for x in cases
        if x["classification"]
        == "GOLD_RANK_WORSENED"
    ]

    unchanged = [
        x
        for x in cases
        if x["classification"]
        == "UNCHANGED"
    ]

    same_candidate_set = sum(
        x["candidate_set"][
            "same_candidate_set"
        ]
        for x in cases
    )

    return {
        "total_questions": len(cases),

        "rank_changed": len(
            rank_changed
        ),

        "improved": len(
            improved
        ),

        "worsened": len(
            worsened
        ),

        "unchanged": len(
            unchanged
        ),

        "improvement_diagnosis": {
            key: sum(
                x["diagnosis"] == key
                for x in improved
            )
            for key in sorted(
                {
                    x["diagnosis"]
                    for x in improved
                }
            )
        },

        "worsening_diagnosis": {
            key: sum(
                x["diagnosis"] == key
                for x in worsened
            )
            for key in sorted(
                {
                    x["diagnosis"]
                    for x in worsened
                }
            )
        },

        "candidate_set": {
            "same": same_candidate_set,
            "different": (
                len(cases)
                - same_candidate_set
            ),
        },
    }


# ============================================================
# Print
# ============================================================

def print_improved(cases):

    improved = [
        x
        for x in cases
        if x["classification"]
        == "GOLD_RANK_IMPROVED"
    ]

    print()
    print("=" * 80)
    print("IMPROVED CASES")
    print("=" * 80)

    for x in improved:

        print()
        print(
            f"{x['id']} | "
            f"{x['e2_gold_rank']} -> "
            f"{x['e6_gold_rank']} | "
            f"{x['diagnosis']}"
        )

        print(
            f"Q: {x['question']}"
        )

        print(
            f"Gold similarity : "
            f"{x['gold']['similarity']}"
        )

        print(
            f"Gold lexical    : "
            f"{x['gold']['lexical_score']}"
        )

        print(
            f"Top1 similarity : "
            f"{x['top1_competitor']['similarity']}"
        )

        print(
            f"Top1 lexical    : "
            f"{x['top1_competitor']['lexical_score']}"
        )

        print(
            f"Semantic gap    : "
            f"{x['semantic_gap']}"
        )

        print(
            f"Lexical gap     : "
            f"{x['lexical_gap']}"
        )


def print_worsened(cases):

    worsened = [
        x
        for x in cases
        if x["classification"]
        == "GOLD_RANK_WORSENED"
    ]

    print()
    print("=" * 80)
    print("WORSENED CASES")
    print("=" * 80)

    for x in worsened:

        print()
        print(
            f"{x['id']} | "
            f"{x['e2_gold_rank']} -> "
            f"{x['e6_gold_rank']} | "
            f"{x['diagnosis']}"
        )

        print(
            f"Q: {x['question']}"
        )

        print(
            f"Gold similarity : "
            f"{x['gold']['similarity']}"
        )

        print(
            f"Gold lexical    : "
            f"{x['gold']['lexical_score']}"
        )

        print(
            f"Top1 similarity : "
            f"{x['top1_competitor']['similarity']}"
        )

        print(
            f"Top1 lexical    : "
            f"{x['top1_competitor']['lexical_score']}"
        )

        print(
            f"Semantic gap    : "
            f"{x['semantic_gap']}"
        )

        print(
            f"Lexical gap     : "
            f"{x['lexical_gap']}"
        )


# ============================================================
# Main
# ============================================================

def main():

    data = load_json(
        INPUT_PATH
    )

    cases = [
        analyze_case(item)
        for item in data["questions"]
    ]

    summary = build_summary(
        cases
    )

    output = {
        "experiment": "E9_E6_diagnosis",

        "source": str(
            INPUT_PATH
        ),

        "alpha": ALPHA,

        "summary": summary,

        "cases": cases,
    }

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

    print()
    print("=" * 80)
    print("E9 - E6 DIAGNOSIS")
    print("=" * 80)

    print(
        f"Questions : "
        f"{summary['total_questions']}"
    )

    print(
        f"Improved  : "
        f"{summary['improved']}"
    )

    print(
        f"Worsened  : "
        f"{summary['worsened']}"
    )

    print(
        f"Unchanged : "
        f"{summary['unchanged']}"
    )

    print()
    print("Improvement diagnosis")
    print("-" * 40)

    for k, v in summary[
        "improvement_diagnosis"
    ].items():

        print(
            f"{k:<35} {v}"
        )

    print()
    print("Worsening diagnosis")
    print("-" * 40)

    for k, v in summary[
        "worsening_diagnosis"
    ].items():

        print(
            f"{k:<35} {v}"
        )

    print_improved(
        cases
    )

    print_worsened(
        cases
    )

    print()
    print("=" * 80)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()