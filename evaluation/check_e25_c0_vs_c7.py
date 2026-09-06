import json
from pathlib import Path


INPUT_FILE = Path(
    "evaluation/results/e25_section_rule_combinations_details.json"
)


def find_detail_combination(obj, combination_name):
    """
    Find the combination object that contains question-level details.

    Important:
    summary.combinations.C0_none only contains generic_rules,
    so it must NOT be selected.
    """

    if isinstance(obj, dict):

        # First priority:
        # combination object that actually contains questions
        if (
            combination_name in obj
            and isinstance(obj[combination_name], dict)
        ):
            candidate = obj[combination_name]

            if isinstance(candidate.get("questions"), list):
                return candidate

            # Sometimes questions may be nested
            if contains_questions(candidate):
                return candidate

        for value in obj.values():
            result = find_detail_combination(
                value,
                combination_name,
            )
            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:
            result = find_detail_combination(
                item,
                combination_name,
            )
            if result is not None:
                return result

    return None


def contains_questions(obj):
    """
    Recursively determine whether an object contains
    a question list.
    """

    if isinstance(obj, dict):

        if isinstance(obj.get("questions"), list):
            return True

        for value in obj.values():
            if contains_questions(value):
                return True

    elif isinstance(obj, list):

        for item in obj:
            if contains_questions(item):
                return True

    return False


def extract_questions(obj):
    """
    Recursively find the actual questions list.
    """

    if isinstance(obj, dict):

        if isinstance(obj.get("questions"), list):
            return obj["questions"]

        for value in obj.values():
            result = extract_questions(value)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:
            result = extract_questions(item)

            if result is not None:
                return result

    return None


def normalize_questions(questions):
    result = {}

    for q in questions:
        if not isinstance(q, dict):
            continue

        qid = q.get("id")

        if qid:
            result[qid] = q

    return result


def get_gold_rank(q):
    """
    Extract gold_rank from the question result.
    """

    if "gold_rank" in q:
        return q["gold_rank"]

    metrics = q.get("metrics")

    if isinstance(metrics, dict):
        if "gold_rank" in metrics:
            return metrics["gold_rank"]

    return None


def get_top10(q):
    top10 = q.get("top10")

    if isinstance(top10, list):
        return top10

    return []


def print_top10(label, q):
    print(f"\n[{label}]")

    for item in get_top10(q):

        rank = item.get("rank", "?")
        chunk_id = item.get("id", "")
        title = item.get("title", "")
        section = item.get("section", "")
        url = item.get("url", "")
        similarity = item.get("similarity")

        if isinstance(similarity, (int, float)):
            sim_text = f"{similarity:.6f}"
        else:
            sim_text = str(similarity)

        print(
            f"{rank:>2} | "
            f"{chunk_id} | "
            f"{title} | "
            f"section={section} | "
            f"sim={sim_text}"
        )

        if url:
            print(f"     URL: {url}")


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find file: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # Find REAL question-level combination objects
    # ------------------------------------------------------------

    c0_obj = find_detail_combination(
        data,
        "C0_none",
    )

    c7_obj = find_detail_combination(
        data,
        "C7_all_three",
    )

    if c0_obj is None:
        raise RuntimeError(
            "Could not find question-level details for C0_none"
        )

    if c7_obj is None:
        raise RuntimeError(
            "Could not find question-level details for C7_all_three"
        )

    c0_questions_raw = extract_questions(c0_obj)
    c7_questions_raw = extract_questions(c7_obj)

    if c0_questions_raw is None:
        raise RuntimeError(
            "C0_none found, but no questions list exists."
        )

    if c7_questions_raw is None:
        raise RuntimeError(
            "C7_all_three found, but no questions list exists."
        )

    c0 = normalize_questions(c0_questions_raw)
    c7 = normalize_questions(c7_questions_raw)

    common_ids = sorted(
        set(c0.keys()) & set(c7.keys())
    )

    print("=" * 80)
    print("E25: C0_none vs C7_all_three")
    print("=" * 80)

    print(f"C0 questions : {len(c0)}")
    print(f"C7 questions : {len(c7)}")
    print(f"Common       : {len(common_ids)}")

    if not common_ids:
        print("\nERROR: No common question IDs.")

        print("\nC0 object keys:")
        print(list(c0_obj.keys()))

        print("\nC7 object keys:")
        print(list(c7_obj.keys()))

        return

    # ------------------------------------------------------------
    # Compare ranks
    # ------------------------------------------------------------

    transitions = []

    for qid in common_ids:

        q0 = c0[qid]
        q7 = c7[qid]

        rank0 = get_gold_rank(q0)
        rank7 = get_gold_rank(q7)

        if rank0 is None or rank7 is None:
            continue

        if rank7 < rank0:
            transition = "IMPROVED"

        elif rank7 > rank0:
            transition = "WORSENED"

        else:
            transition = "UNCHANGED"

        transitions.append(
            {
                "id": qid,
                "question": q7.get(
                    "question",
                    q0.get("question", ""),
                ),
                "c0_rank": rank0,
                "c7_rank": rank7,
                "delta": rank0 - rank7,
                "transition": transition,
            }
        )

    improved = [
        x for x in transitions
        if x["transition"] == "IMPROVED"
    ]

    worsened = [
        x for x in transitions
        if x["transition"] == "WORSENED"
    ]

    unchanged = [
        x for x in transitions
        if x["transition"] == "UNCHANGED"
    ]

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"IMPROVED  : {len(improved)}")
    print(f"WORSENED  : {len(worsened)}")
    print(f"UNCHANGED : {len(unchanged)}")

    # ------------------------------------------------------------
    # Improved
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPROVED")
    print("=" * 80)

    for x in sorted(
        improved,
        key=lambda item: item["delta"],
        reverse=True,
    ):

        print(
            f'\n{x["id"]} | '
            f'{x["c0_rank"]} -> {x["c7_rank"]} '
            f'| delta +{x["delta"]}'
        )

        print(x["question"])

    # ------------------------------------------------------------
    # Worsened
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("WORSENED")
    print("=" * 80)

    for x in sorted(
        worsened,
        key=lambda item: item["delta"],
    ):

        print(
            f'\n{x["id"]} | '
            f'{x["c0_rank"]} -> {x["c7_rank"]} '
            f'| delta {x["delta"]}'
        )

        print(x["question"])

    # ------------------------------------------------------------
    # Detailed changed questions
    # ------------------------------------------------------------

    changed_ids = [
        x["id"]
        for x in transitions
        if x["transition"] != "UNCHANGED"
    ]

    print("\n" + "=" * 80)
    print("DETAILED CHANGED QUESTIONS")
    print("=" * 80)

    for qid in changed_ids:

        q0 = c0[qid]
        q7 = c7[qid]

        transition = next(
            x
            for x in transitions
            if x["id"] == qid
        )

        print("\n" + "-" * 80)

        print(
            f'{qid}: {transition["question"]}'
        )

        print(
            f'Gold rank: '
            f'{transition["c0_rank"]} -> '
            f'{transition["c7_rank"]}'
        )

        print_top10(
            "C0_none",
            q0,
        )

        print_top10(
            "C7_all_three",
            q7,
        )

    # ------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------

    output = {
        "experiment": "E25_C0_vs_C7",
        "c0": "C0_none",
        "c7": "C7_all_three",
        "question_count": len(common_ids),
        "improved": len(improved),
        "worsened": len(worsened),
        "unchanged": len(unchanged),
        "transitions": transitions,
    }

    output_file = Path(
        "evaluation/results/e25_c0_vs_c7_analysis.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f"Saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()