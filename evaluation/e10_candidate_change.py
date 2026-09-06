import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

INPUT_PATH = RESULTS_DIR / "e8_e6_vs_e2_analysis.json"
OUTPUT_PATH = RESULTS_DIR / "e10_candidate_change.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url):
    if not url:
        return ""
    return str(url).rstrip("/")


def get_rank_map(results):
    """
    results:
        [
            {
                "rank": 1,
                "url": "..."
            },
            ...
        ]
    """

    output = {}

    for result in results:
        url = normalize_url(
            result.get("url")
        )

        if url:
            output[url] = result.get(
                "rank"
            )

    return output


def get_e2_results(item):
    """
    E8 already converted E2 retrieved_sources
    into e2.top1 / e2.gold_results.

    However, E8 does not retain the complete E2 Top10.

    Therefore reconstruct the E2 Top10 from the
    original E2 data is not possible from E8 alone.

    We will use candidate_set information plus E6
    for the first diagnostic.
    """

    return []


def classify_candidate_change(item):
    candidate = item[
        "candidate_changes"
    ]

    same = candidate[
        "same_candidate_set"
    ]

    rank_change = item.get(
        "rank_change"
    )

    classification = item[
        "classification"
    ]

    if same:
        if classification == "GOLD_RANK_IMPROVED":
            return "PURE_RERANK_IMPROVEMENT"

        if classification == "GOLD_RANK_WORSENED":
            return "PURE_RERANK_REGRESSION"

        return "SAME_SET_UNCHANGED"

    # Candidate set changed.
    if classification == "GOLD_RANK_IMPROVED":
        return "CANDIDATE_REPLACEMENT_IMPROVEMENT"

    if classification == "GOLD_RANK_WORSENED":
        return "CANDIDATE_REPLACEMENT_REGRESSION"

    return "CANDIDATE_CHANGE_NO_GOLD_RANK_CHANGE"


def analyze_case(item):
    classification = classify_candidate_change(
        item
    )

    candidate = item[
        "candidate_changes"
    ]

    return {
        "id": item["id"],
        "question": item["question"],

        "classification": item[
            "classification"
        ],

        "mechanism": classification,

        "e2_gold_rank": item[
            "e2"
        ].get("gold_rank"),

        "e6_gold_rank": item[
            "e6"
        ].get("gold_rank"),

        "rank_change": item.get(
            "rank_change"
        ),

        "e2_top1": item[
            "e2"
        ].get("top1"),

        "e6_top1": item[
            "e6"
        ].get("top1"),

        "candidate_set": {
            "same": candidate[
                "same_candidate_set"
            ],

            "e2_count": candidate[
                "e2_count"
            ],

            "e6_count": candidate[
                "e6_count"
            ],

            "entered": candidate[
                "entered"
            ],

            "removed": candidate[
                "removed"
            ],
        },
    }


def main():

    data = load_json(
        INPUT_PATH
    )

    cases = [
        analyze_case(item)
        for item in data["questions"]
    ]

    # ========================================================
    # Summary
    # ========================================================

    mechanism_counts = {}

    for item in cases:

        mechanism = item[
            "mechanism"
        ]

        mechanism_counts[
            mechanism
        ] = (
            mechanism_counts.get(
                mechanism,
                0
            )
            + 1
        )

    # ========================================================
    # Important subsets
    # ========================================================

    pure_rerank = [
        x
        for x in cases
        if x["mechanism"]
        in (
            "PURE_RERANK_IMPROVEMENT",
            "PURE_RERANK_REGRESSION",
        )
    ]

    candidate_replacement = [
        x
        for x in cases
        if x["mechanism"]
        in (
            "CANDIDATE_REPLACEMENT_IMPROVEMENT",
            "CANDIDATE_REPLACEMENT_REGRESSION",
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

    # ========================================================
    # Output
    # ========================================================

    output = {
        "experiment": (
            "E10_candidate_change"
        ),

        "source": str(
            INPUT_PATH
        ),

        "summary": {
            "questions": len(cases),

            "mechanism_counts":
                mechanism_counts,

            "pure_rerank_cases":
                len(pure_rerank),

            "candidate_replacement_cases":
                len(
                    candidate_replacement
                ),

            "improved_cases":
                len(improved),

            "worsened_cases":
                len(worsened),
        },

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

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)
    print("E10 - CANDIDATE CHANGE DIAGNOSIS")
    print("=" * 80)

    print(
        f"Questions : {len(cases)}"
    )

    print()
    print("Mechanism")
    print("-" * 60)

    for key, value in sorted(
        mechanism_counts.items()
    ):
        print(
            f"{key:<45} {value}"
        )

    print()
    print("=" * 80)
    print("IMPROVED")
    print("=" * 80)

    for item in improved:

        print()
        print(
            f"{item['id']} | "
            f"{item['e2_gold_rank']} -> "
            f"{item['e6_gold_rank']} | "
            f"{item['mechanism']}"
        )

        print(
            item["question"]
        )

        candidate = item[
            "candidate_set"
        ]

        print(
            f"candidate set same: "
            f"{candidate['same']}"
        )

        if candidate["entered"]:
            print(
                "entered:"
            )

            for source in candidate[
                "entered"
            ]:
                print(
                    f"  + {source}"
                )

        if candidate["removed"]:
            print(
                "removed:"
            )

            for source in candidate[
                "removed"
            ]:
                print(
                    f"  - {source}"
                )

    print()
    print("=" * 80)
    print("WORSENED")
    print("=" * 80)

    for item in worsened:

        print()
        print(
            f"{item['id']} | "
            f"{item['e2_gold_rank']} -> "
            f"{item['e6_gold_rank']} | "
            f"{item['mechanism']}"
        )

        print(
            item["question"]
        )

        candidate = item[
            "candidate_set"
        ]

        print(
            f"candidate set same: "
            f"{candidate['same']}"
        )

        if candidate["entered"]:
            print(
                "entered:"
            )

            for source in candidate[
                "entered"
            ]:
                print(
                    f"  + {source}"
                )

        if candidate["removed"]:
            print(
                "removed:"
            )

            for source in candidate[
                "removed"
            ]:
                print(
                    f"  - {source}"
                )

    print()
    print("=" * 80)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()