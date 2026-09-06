import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

E2_PATH = ROOT_DIR / "evaluation/results/retrieval_matrix_details.json"
E6_PATH = ROOT_DIR / "evaluation/results/e6_lexical_alpha_details.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rank(item):
    sources = [
        x.rstrip("/")
        for x in item.get("retrieved_sources", [])
    ]

    gold = {
        x.rstrip("/")
        for x in item.get("relevant_sources", [])
    }

    for i, source in enumerate(sources, 1):
        if source in gold:
            return i

    return None


def main():
    e2 = load(E2_PATH)["results"]["E2"]
    e6 = load(E6_PATH)["results"]["E6_05"]

    e2_map = {x["id"]: x for x in e2}
    e6_map = {x["id"]: x for x in e6}

    print("=" * 100)
    print("E2 → E6(alpha=0.05) CHANGED CASES")
    print("=" * 100)

    for qid in sorted(e2_map):
        a = e2_map[qid]
        b = e6_map[qid]

        r2 = rank(a)
        r6 = rank(b)

        if r2 == r6:
            continue

        # E6 top1
        top1 = b.get("retrieved", [{}])[0]

        similarity = top1.get("similarity")

        print()
        print(
            f"{qid}  "
            f"{r2} → {r6}  "
            f"Δ={r2-r6:+d}"
        )
        print(f"Q: {a['question']}")
        print(f"E6 Top1: {top1.get('title', '')}")
        print(f"Similarity: {similarity}")

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()