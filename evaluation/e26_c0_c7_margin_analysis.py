import json
from pathlib import Path


RESULT_FILE = Path(
    "evaluation/results/e25_section_rule_combinations_details.json"
)


CHANGED_IDS = [
    "q013",
    "q002",
    "q006",
    "q008",
    "q027",
    "q034",
    "q016",
    "q045",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_question_list(obj, combination_id):
    """
    Robustly find the question list belonging to one E25 combination.
    """

    if isinstance(obj, dict):
        # Direct structure:
        # {
        #   "combination": "...",
        #   "questions": [...]
        # }
        if (
            obj.get("combination_id") == combination_id
            and isinstance(obj.get("questions"), list)
        ):
            return obj["questions"]

        if (
            obj.get("id") == combination_id
            and isinstance(obj.get("questions"), list)
        ):
            return obj["questions"]

        # Some result files may use the combination ID as a dict key.
        if combination_id in obj:
            found = find_question_list(obj[combination_id], combination_id)
            if found is not None:
                return found

        for value in obj.values():
            found = find_question_list(value, combination_id)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_question_list(item, combination_id)
            if found is not None:
                return found

    return None


def get_ranked_items(question):
    top10 = question.get("top10", [])

    normalized = []

    for item in top10:
        normalized.append(
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "source": item.get("source"),
                "similarity": item.get("similarity"),
                "section": item.get("section"),
                "content": item.get("content"),
            }
        )

    return normalized


def is_gold(item, relevant_sources):
    source = item.get("source")
    return source in relevant_sources


def get_first_gold(top10, relevant_sources):
    for item in top10:
        if is_gold(item, relevant_sources):
            return item
    return None


def fmt_similarity(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find E25 result file:\n{RESULT_FILE}"
        )

    data = load_json(RESULT_FILE)

    c0 = find_question_list(data, "C0_none")
    c7 = find_question_list(data, "C7_all_three")

    if c0 is None:
        raise RuntimeError(
            "Could not find questions for C0_none in E25 details."
        )

    if c7 is None:
        raise RuntimeError(
            "Could not find questions for C7_all_three in E25 details."
        )

    c0_by_id = {q["id"]: q for q in c0}
    c7_by_id = {q["id"]: q for q in c7}

    print("=" * 100)
    print("E26 - C0 vs C7 MARGIN ANALYSIS")
    print("=" * 100)

    print()
    print(f"Result file : {RESULT_FILE}")
    print(f"C0 questions: {len(c0_by_id)}")
    print(f"C7 questions: {len(c7_by_id)}")
    print()

    for qid in CHANGED_IDS:
        if qid not in c0_by_id:
            print(f"[ERROR] {qid} missing from C0")
            continue

        if qid not in c7_by_id:
            print(f"[ERROR] {qid} missing from C7")
            continue

        q0 = c0_by_id[qid]
        q7 = c7_by_id[qid]

        relevant_sources = set(
            q0.get("relevant_sources", [])
            or q7.get("relevant_sources", [])
        )

        top0 = get_ranked_items(q0)
        top7 = get_ranked_items(q7)

        gold0 = get_first_gold(top0, relevant_sources)
        gold7 = get_first_gold(top7, relevant_sources)

        top1_0 = top0[0] if top0 else None
        top1_7 = top7[0] if top7 else None

        gold_rank0 = gold0["rank"] if gold0 else None
        gold_rank7 = gold7["rank"] if gold7 else None

        gold_sim0 = gold0["similarity"] if gold0 else None
        gold_sim7 = gold7["similarity"] if gold7 else None

        top1_sim0 = top1_0["similarity"] if top1_0 else None
        top1_sim7 = top1_7["similarity"] if top1_7 else None

        margin0 = (
            top1_sim0 - gold_sim0
            if top1_sim0 is not None
            and gold_sim0 is not None
            and not is_gold(top1_0, relevant_sources)
            else 0.0
        )

        margin7 = (
            top1_sim7 - gold_sim7
            if top1_sim7 is not None
            and gold_sim7 is not None
            and not is_gold(top1_7, relevant_sources)
            else 0.0
        )

        top1_gold0 = is_gold(top1_0, relevant_sources) if top1_0 else False
        top1_gold7 = is_gold(top1_7, relevant_sources) if top1_7 else False

        rank_delta = (
            gold_rank0 - gold_rank7
            if gold_rank0 is not None and gold_rank7 is not None
            else None
        )

        gold_similarity_delta = (
            gold_sim7 - gold_sim0
            if gold_sim0 is not None and gold_sim7 is not None
            else None
        )

        print("-" * 100)
        print(f"{qid}")
        print(f"Question: {q0.get('question')}")

        print()
        print("C0")
        print(
            f"  Gold rank       : {gold_rank0}"
        )
        print(
            f"  Gold similarity : {fmt_similarity(gold_sim0)}"
        )
        print(
            f"  Top1 gold?      : {top1_gold0}"
        )
        print(
            f"  Top1            : "
            f"{top1_0.get('title') if top1_0 else 'N/A'}"
        )
        print(
            f"  Top1 similarity : {fmt_similarity(top1_sim0)}"
        )
        print(
            f"  Non-gold margin : {fmt_similarity(margin0)}"
        )

        if gold0:
            print(
                f"  Gold chunk      : {gold0.get('title')}"
            )
            print(
                f"  Gold section    : {gold0.get('section')}"
            )

        print()
        print("C7")
        print(
            f"  Gold rank       : {gold_rank7}"
        )
        print(
            f"  Gold similarity : {fmt_similarity(gold_sim7)}"
        )
        print(
            f"  Top1 gold?      : {top1_gold7}"
        )
        print(
            f"  Top1            : "
            f"{top1_7.get('title') if top1_7 else 'N/A'}"
        )
        print(
            f"  Top1 similarity : {fmt_similarity(top1_sim7)}"
        )
        print(
            f"  Non-gold margin : {fmt_similarity(margin7)}"
        )

        if gold7:
            print(
                f"  Gold chunk      : {gold7.get('title')}"
            )
            print(
                f"  Gold section    : {gold7.get('section')}"
            )

        print()
        print("CHANGE")
        print(
            f"  Gold rank delta       : "
            f"{rank_delta:+d}" if rank_delta is not None
            else "  Gold rank delta       : N/A"
        )
        print(
            f"  Gold similarity delta : "
            f"{gold_similarity_delta:+.6f}"
            if gold_similarity_delta is not None
            else "  Gold similarity delta : N/A"
        )
        print(
            f"  Top1 transition       : "
            f"{'GOLD' if top1_gold0 else 'NON-GOLD'} -> "
            f"{'GOLD' if top1_gold7 else 'NON-GOLD'}"
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    new_top1 = []
    lost_top1 = []
    unchanged_top1 = []

    for qid in CHANGED_IDS:
        if qid not in c0_by_id or qid not in c7_by_id:
            continue

        q0 = c0_by_id[qid]
        q7 = c7_by_id[qid]

        relevant_sources = set(
            q0.get("relevant_sources", [])
            or q7.get("relevant_sources", [])
        )

        top0 = get_ranked_items(q0)
        top7 = get_ranked_items(q7)

        top1_0 = top0[0] if top0 else None
        top1_7 = top7[0] if top7 else None

        gold0 = (
            is_gold(top1_0, relevant_sources)
            if top1_0
            else False
        )

        gold7 = (
            is_gold(top1_7, relevant_sources)
            if top1_7
            else False
        )

        if not gold0 and gold7:
            new_top1.append(qid)
        elif gold0 and not gold7:
            lost_top1.append(qid)
        else:
            unchanged_top1.append(qid)

    print()
    print(f"New Top1 gold : {len(new_top1)}")
    print(f"  {new_top1}")

    print()
    print(f"Lost Top1 gold: {len(lost_top1)}")
    print(f"  {lost_top1}")

    print()
    print(f"Other         : {len(unchanged_top1)}")
    print(f"  {unchanged_top1}")

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        "This analysis uses SOURCE-LEVEL relevance: a chunk is gold when "
        "its source URL is in relevant_sources."
    )

    print(
        "The margin is Top1 similarity - gold similarity only when Top1 "
        "is non-gold. If Top1 is already gold, margin is reported as 0."
    )

    print()
    print("Do NOT treat a small rank change as a robust improvement automatically.")
    print(
        "Margins close to 0 indicate near-ties and should be interpreted cautiously."
    )


if __name__ == "__main__":
    main()