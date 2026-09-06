"""
Extract PARTIAL and FAIL cases from generation_evaluation.json.

Input:
    evaluation/results/generation_evaluation.json

Output:
    evaluation/results/generation_errors.json
"""

import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "generation_evaluation.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "generation_errors.json"
)


# ============================================================
# Load
# ============================================================

def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Generation evaluation file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Extract
# ============================================================

def extract_errors(
    data: dict,
) -> list[dict]:

    results = data.get(
        "results",
        [],
    )

    errors = []

    for item in results:

        # Skip evaluation errors
        if "evaluation_error" in item:
            continue

        evaluation = item.get(
            "evaluation",
            {},
        )

        overall_label = evaluation.get(
            "overall_label"
        )

        # Keep only PARTIAL and FAIL
        if overall_label not in {
            "PARTIAL",
            "FAIL",
        }:
            continue

        errors.append(
            {
                "id": item.get("id"),

                "question": item.get(
                    "question",
                    "",
                ),

                "expected_answer": item.get(
                    "expected_answer",
                    "",
                ),

                "generated_answer": item.get(
                    "generated_answer",
                    "",
                ),

                "relevant_sources": item.get(
                    "relevant_sources",
                    [],
                ),

                "retrieval": {
                    "hit": item.get(
                        "retrieval",
                        {},
                    ).get(
                        "hit"
                    ),

                    "retrieved_urls": item.get(
                        "retrieval",
                        {},
                    ).get(
                        "retrieved_urls",
                        [],
                    ),
                },

                "evaluation": {
                    "answer_correctness": evaluation.get(
                        "answer_correctness",
                        {},
                    ),

                    "faithfulness": evaluation.get(
                        "faithfulness",
                        {},
                    ),

                    "citation_correctness": evaluation.get(
                        "citation_correctness",
                        {},
                    ),

                    "overall_score": evaluation.get(
                        "overall_score"
                    ),

                    "overall_label": overall_label,
                },

                "judge": item.get(
                    "judge",
                    {},
                ),
            }
        )

    return errors


# ============================================================
# Save
# ============================================================

def save_results(
    errors: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            errors,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Print Summary
# ============================================================

def print_summary(
    errors: list[dict],
) -> None:

    partial = [
        item
        for item in errors
        if item["evaluation"]["overall_label"]
        == "PARTIAL"
    ]

    fail = [
        item
        for item in errors
        if item["evaluation"]["overall_label"]
        == "FAIL"
    ]

    retrieval_hits = [
        item
        for item in errors
        if item["retrieval"]["hit"]
    ]

    retrieval_misses = [
        item
        for item in errors
        if not item["retrieval"]["hit"]
    ]

    print()
    print("=" * 80)
    print("GENERATION ERROR ANALYSIS EXTRACTION")
    print("=" * 80)

    print()

    print(
        f"Total PARTIAL + FAIL: "
        f"{len(errors)}"
    )

    print(
        f"PARTIAL: "
        f"{len(partial)}"
    )

    print(
        f"FAIL: "
        f"{len(fail)}"
    )

    print()

    print("Retrieval Status")

    print(
        f"  Retrieval Hit: "
        f"{len(retrieval_hits)}"
    )

    print(
        f"  Retrieval Miss: "
        f"{len(retrieval_misses)}"
    )

    print()

    print("Cases")

    for index, item in enumerate(
        errors,
        start=1,
    ):

        evaluation = item["evaluation"]

        print(
            f"{index:02d}. "
            f"{item['id']} | "
            f"{evaluation['overall_label']} | "
            f"Overall={evaluation['overall_score']} | "
            f"Retrieval="
            f"{'HIT' if item['retrieval']['hit'] else 'MISS'}"
        )

        print(
            f"    Q: "
            f"{item['question']}"
        )

        print(
            f"    Correctness: "
            f"{evaluation['answer_correctness'].get('score')}/2"
        )

        print(
            f"    Faithfulness: "
            f"{evaluation['faithfulness'].get('score')}/2"
        )

        print(
            f"    Citation: "
            f"{evaluation['citation_correctness'].get('score')}/2"
        )

        print()


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Input:  {INPUT_PATH}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    data = load_results(
        INPUT_PATH
    )

    errors = extract_errors(
        data
    )

    save_results(
        errors,
        OUTPUT_PATH,
    )

    print_summary(
        errors
    )

    print(
        f"Saved {len(errors)} cases."
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()