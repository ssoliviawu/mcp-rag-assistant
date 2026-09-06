import json
from pathlib import Path


INPUT = Path("evaluation/results/e25_section_rule_combinations_details.json")
OUTPUT = Path("evaluation/results/e27_input.json")


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 找到 C7
    representations = data.get("representations", data)

    c7 = None

    if isinstance(representations, dict):
        for key, value in representations.items():
            key_text = str(key).lower()
            if "c7" in key_text or "all_three" in key_text:
                c7 = value
                print(f"Found C7: {key}")
                break

    if c7 is None:
        raise RuntimeError(
            "Could not find C7_all_three in the JSON structure."
        )

    questions = c7.get("questions", [])

    if not questions:
        raise RuntimeError(
            "C7 found, but no questions field was found."
        )

    # 只保留 E27 真正需要的信息
    output_questions = []

    for item in questions:
        gold_rank = item.get("gold_rank")

        # 只提取非 Top1
        if gold_rank != 1:
            output_questions.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "gold_rank": gold_rank,
                "metrics": item.get("metrics"),
                "top10": item.get("top10"),
            })

    output = {
        "source": INPUT.name,
        "configuration": "C7_all_three",
        "total_questions": len(questions),
        "non_top1_questions": len(output_questions),
        "questions": output_questions,
    }

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("E27 INPUT EXTRACTED")
    print("=" * 80)
    print(f"Total C7 questions : {len(questions)}")
    print(f"Non-Top1 questions : {len(output_questions)}")
    print(f"Output             : {OUTPUT}")

    print("\nNon-Top1:")
    for q in output_questions:
        print(
            f"{q['id']:>5} | "
            f"gold_rank={q['gold_rank']} | "
            f"{q['question']}"
        )


if __name__ == "__main__":
    main()