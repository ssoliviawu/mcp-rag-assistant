import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e2_final_generation_evaluation.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e31_final_case_review.json"
)


def load_results():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("results", [])

    return data


def get_label(item):
    """
    Support the evaluator's possible result structures.
    """

    if "label" in item:
        return item["label"]

    if "evaluation" in item:
        evaluation = item["evaluation"]

        if "label" in evaluation:
            return evaluation["label"]

        if "overall_label" in evaluation:
            return evaluation["overall_label"]

    return "UNKNOWN"


def get_score(item, metric):
    evaluation = item.get("evaluation", {})

    # Common structure:
    # evaluation["metrics"]["answer_correctness"]["score"]

    metrics = evaluation.get("metrics", {})

    if metric in metrics:
        value = metrics[metric]

        if isinstance(value, dict):
            return value.get("score")

        return value

    # Alternative flat structure
    return evaluation.get(metric)


def get_answer(item):
    generation = item.get("generation", {})
    return generation.get("answer", "")


def get_retrieval(item):
    retrieval = item.get("retrieval", {})

    return {
        "hit": retrieval.get("hit"),
        "retrieved_urls": retrieval.get("retrieved_urls", []),
        "retrieved_details": retrieval.get("retrieved_details", []),
    }


def main():

    results = load_results()

    print("=" * 90)
    print("E31 FINAL CASE REVIEW")
    print("=" * 90)

    print(f"Input: {INPUT_PATH}")
    print(f"Total results: {len(results)}")
    print()

    # --------------------------------------------------------
    # Select non-PASS cases
    # --------------------------------------------------------

    selected = []

    for item in results:

        label = get_label(item)

        if label.upper() in {"FAIL", "PARTIAL"}:
            selected.append(item)

    print(f"Non-PASS cases: {len(selected)}")
    print()

    # --------------------------------------------------------
    # Print cases
    # --------------------------------------------------------

    output = []

    for index, item in enumerate(selected, start=1):

        question_id = item.get("id")
        question = item.get("question", "")

        label = get_label(item)

        answer_correctness = get_score(
            item,
            "answer_correctness"
        )

        faithfulness = get_score(
            item,
            "faithfulness"
        )

        citation_correctness = get_score(
            item,
            "citation_correctness"
        )

        answer = get_answer(item)

        retrieval = get_retrieval(item)

        retrieved_details = retrieval["retrieved_details"]

        top_results = []

        for result in retrieved_details[:5]:

            top_results.append({
                "rank": result.get("rank"),
                "title": result.get("title"),
                "section": result.get("section"),
                "url": result.get("url"),
                "content_preview": (
                    result.get("content", "")[:500]
                ),
            })

        case = {
            "id": question_id,
            "question": question,
            "label": label,

            "scores": {
                "answer_correctness": answer_correctness,
                "faithfulness": faithfulness,
                "citation_correctness": citation_correctness,
            },

            "retrieval": {
                "hit": retrieval["hit"],
                "retrieved_urls": retrieval["retrieved_urls"],
                "top5": top_results,
            },

            "answer": answer,

            "manual_classification": {
                "error_type": "",
                "notes": "",
            },
        }

        output.append(case)

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print("=" * 90)
        print(
            f"[{index}/{len(selected)}] "
            f"{question_id} | {label}"
        )
        print("=" * 90)

        print()
        print("QUESTION:")
        print(question)

        print()
        print("SCORES:")
        print(
            f"  Answer Correctness : "
            f"{answer_correctness}"
        )
        print(
            f"  Faithfulness       : "
            f"{faithfulness}"
        )
        print(
            f"  Citation Correctness: "
            f"{citation_correctness}"
        )

        print()
        print("RETRIEVAL:")
        print(
            f"  Hit: {retrieval['hit']}"
        )

        print()
        print("TOP 5 RETRIEVED:")

        for result in top_results:

            print(
                f"  Rank {result['rank']}: "
                f"{result['title']}"
            )

            if result["section"]:
                print(
                    f"      Section: "
                    f"{result['section']}"
                )

            print(
                f"      URL: "
                f"{result['url']}"
            )

        print()
        print("ANSWER:")
        print(answer)

        print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "experiment": "E31_final_case_review",

                "purpose": (
                    "Final qualitative review of "
                    "non-PASS E2 generation cases. "
                    "No retrieval or generation "
                    "optimization is performed."
                ),

                "total_questions": len(results),

                "non_pass_questions": len(selected),

                "cases": output,

            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()