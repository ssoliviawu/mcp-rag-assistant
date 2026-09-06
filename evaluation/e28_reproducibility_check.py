from __future__ import annotations

import json
from pathlib import Path


from embedding.model import BGEEmbeddingModel
from retrieval.search import search_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "mcp_rag_eval_dataset.jsonl"
E27_INPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "e27_input.json"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "e28_reproducibility_check.json"
)

CANDIDATE_LIMIT = 100

RUN_COUNT = 2


def load_questions() -> list[dict]:
    questions = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # E27 使用的是非-spec问题
            if item.get("domain") == "spec":
                continue

            questions.append(item)

    return questions


def load_e27_residual_ids():
    """
    Load residual question IDs from E27 input.

    Supports common JSON structures:
    1. [{"id": "q003"}, {"id": "q009"}, ...]
    2. {"q003": {...}, "q009": {...}, ...}
    3. {"residual_ids": ["q003", "q009", ...]}
    4. {"questions": [{"id": "q003"}, ...]}
    5. {"items": [{"id": "q003"}, ...]}
    """

    path = Path("evaluation/results/e27_input.json")

    if not path.exists():
        raise FileNotFoundError(
            f"E27 input file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    residual_ids = []

    def add_item(item):
        if isinstance(item, str):
            residual_ids.append(item)

        elif isinstance(item, dict):
            question_id = item.get("id")
            if isinstance(question_id, str):
                residual_ids.append(question_id)

    if isinstance(data, list):
        for item in data:
            add_item(item)

    elif isinstance(data, dict):

        # Case 1: explicit residual_ids
        if isinstance(data.get("residual_ids"), list):
            for item in data["residual_ids"]:
                add_item(item)

        # Case 2: questions
        elif isinstance(data.get("questions"), list):
            for item in data["questions"]:
                add_item(item)

        # Case 3: items
        elif isinstance(data.get("items"), list):
            for item in data["items"]:
                add_item(item)

        # Case 4: dict keyed by question ID
        else:
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("q"):
                    residual_ids.append(key)

                elif isinstance(value, list):
                    for item in value:
                        add_item(item)

    else:
        raise ValueError(
            f"Unsupported e27_input.json structure: "
            f"{type(data).__name__}"
        )

    # 去重，同时保持原顺序
    residual_ids = list(dict.fromkeys(residual_ids))

    if not residual_ids:
        raise ValueError(
            f"No residual question IDs found in {path}"
        )

    print(f"Loaded {len(residual_ids)} E27 residual IDs:")
    print(", ".join(residual_ids))

    return residual_ids


def normalize_source(source: str | None) -> str:
    if not source:
        return ""

    return source.rstrip("/")


def get_rank_for_source(
    candidates: list[dict],
    target_source: str,
) -> int | None:
    target_source = normalize_source(target_source)

    for rank, item in enumerate(candidates, start=1):
        source = normalize_source(item.get("source"))

        if source == target_source:
            return rank

    return None


def get_best_source_rank(
    candidates: list[dict],
    target_source: str,
) -> tuple[int | None, dict | None]:
    target_source = normalize_source(target_source)

    for rank, item in enumerate(candidates, start=1):
        source = normalize_source(item.get("source"))

        if source == target_source:
            return rank, item

    return None, None


def build_run(
    questions: list[dict],
    residual_ids: set[str],
    model: BGEEmbeddingModel,
) -> dict:
    results = []

    migration_source = "https://py.sdk.modelcontextprotocol.io/migration/"

    for question in questions:
        question_id = question["id"]

        if question_id not in residual_ids:
            continue

        query = question["question"]

        print(f"  {question_id}: {query}")

        query_embedding = model.embed_query(query)

        candidates = search_chunks(
            query_embedding,
            limit=CANDIDATE_LIMIT,
        )

        gold_sources = {
            normalize_source(source)
            for source in question.get("relevant_sources", [])
        }

        gold_rank = None
        gold_item = None

        for rank, candidate in enumerate(candidates, start=1):
            candidate_source = normalize_source(
                candidate.get("source")
            )

            if candidate_source in gold_sources:
                gold_rank = rank
                gold_item = candidate
                break

        migration_rank, migration_item = get_best_source_rank(
            candidates,
            migration_source,
        )

        results.append(
            {
                "id": question_id,
                "question": query,
                "gold_sources": sorted(gold_sources),
                "gold_rank": gold_rank,
                "gold_title": (
                    gold_item.get("title")
                    if gold_item
                    else None
                ),
                "gold_similarity": (
                    gold_item.get("similarity")
                    if gold_item
                    else None
                ),
                "migration_rank": migration_rank,
                "migration_title": (
                    migration_item.get("title")
                    if migration_item
                    else None
                ),
                "migration_similarity": (
                    migration_item.get("similarity")
                    if migration_item
                    else None
                ),
                "top10": [
                    {
                        "rank": rank,
                        "title": item.get("title"),
                        "section": item.get("section"),
                        "source": item.get("source"),
                        "similarity": item.get("similarity"),
                    }
                    for rank, item in enumerate(
                        candidates[:10],
                        start=1,
                    )
                ],
                "top100": [
                    {
                        "rank": rank,
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "similarity": item.get("similarity"),
                    }
                    for rank, item in enumerate(
                        candidates,
                        start=1,
                    )
                ],
            }
        )

    return {
        "results": results,
    }


def compare_runs(
    run_a: dict,
    run_b: dict,
) -> dict:
    map_a = {
        item["id"]: item
        for item in run_a["results"]
    }

    map_b = {
        item["id"]: item
        for item in run_b["results"]
    }

    question_ids = sorted(
        set(map_a) | set(map_b)
    )

    comparisons = []

    migration_rank_changed = 0
    gold_rank_changed = 0
    top10_changed = 0
    top100_changed = 0

    for question_id in question_ids:
        a = map_a.get(question_id)
        b = map_b.get(question_id)

        if a is None or b is None:
            comparisons.append(
                {
                    "id": question_id,
                    "status": "MISSING_IN_ONE_RUN",
                    "run_a": a,
                    "run_b": b,
                }
            )
            continue

        migration_changed = (
            a["migration_rank"]
            != b["migration_rank"]
        )

        gold_changed = (
            a["gold_rank"]
            != b["gold_rank"]
        )

        top10_a = [
            (
                item["source"],
                item["title"],
                item["similarity"],
            )
            for item in a["top10"]
        ]

        top10_b = [
            (
                item["source"],
                item["title"],
                item["similarity"],
            )
            for item in b["top10"]
        ]

        top100_a = [
            (
                item["source"],
                item["title"],
                item["similarity"],
            )
            for item in a["top100"]
        ]

        top100_b = [
            (
                item["source"],
                item["title"],
                item["similarity"],
            )
            for item in b["top100"]
        ]

        top10_diff = top10_a != top10_b
        top100_diff = top100_a != top100_b

        if migration_changed:
            migration_rank_changed += 1

        if gold_changed:
            gold_rank_changed += 1

        if top10_diff:
            top10_changed += 1

        if top100_diff:
            top100_changed += 1

        comparisons.append(
            {
                "id": question_id,
                "migration_rank": {
                    "run_a": a["migration_rank"],
                    "run_b": b["migration_rank"],
                    "changed": migration_changed,
                },
                "gold_rank": {
                    "run_a": a["gold_rank"],
                    "run_b": b["gold_rank"],
                    "changed": gold_changed,
                },
                "top10_changed": top10_diff,
                "top100_changed": top100_diff,
                "migration_similarity": {
                    "run_a": a["migration_similarity"],
                    "run_b": b["migration_similarity"],
                },
                "gold_similarity": {
                    "run_a": a["gold_similarity"],
                    "run_b": b["gold_similarity"],
                },
            }
        )

    total = len(question_ids)

    return {
        "total_questions": total,
        "migration_rank_changed": migration_rank_changed,
        "gold_rank_changed": gold_rank_changed,
        "top10_changed": top10_changed,
        "top100_changed": top100_changed,
        "all_top100_identical": top100_changed == 0,
        "all_top10_identical": top10_changed == 0,
        "all_migration_ranks_identical": (
            migration_rank_changed == 0
        ),
        "all_gold_ranks_identical": (
            gold_rank_changed == 0
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    print("=" * 80)
    print("E28-R REPRODUCIBILITY CHECK")
    print("=" * 80)

    print()
    print("Loading questions...")

    questions = load_questions()
    residual_ids = load_e27_residual_ids()

    print(f"Non-spec questions : {len(questions)}")
    print(f"E27 residual IDs   : {len(residual_ids)}")

    print()
    print("Residual IDs:")
    print(", ".join(sorted(residual_ids)))

    print()
    print("Loading embedding model...")

    model = BGEEmbeddingModel()

    runs = []

    for run_index in range(1, RUN_COUNT + 1):
        print()
        print("=" * 80)
        print(f"RUN {run_index}")
        print("=" * 80)

        run_result = build_run(
            questions,
            residual_ids,
            model,
        )

        runs.append(run_result)

    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    comparison = compare_runs(
        runs[0],
        runs[1],
    )

    print(
        f"Total questions       : "
        f"{comparison['total_questions']}"
    )

    print(
        f"Migration rank changed: "
        f"{comparison['migration_rank_changed']}"
    )

    print(
        f"Gold rank changed     : "
        f"{comparison['gold_rank_changed']}"
    )

    print(
        f"Top10 changed         : "
        f"{comparison['top10_changed']}"
    )

    print(
        f"Top100 changed        : "
        f"{comparison['top100_changed']}"
    )

    print()
    print("Question-level changes:")

    for item in comparison["comparisons"]:
        changed = (
            item["migration_rank"]["changed"]
            or item["gold_rank"]["changed"]
            or item["top10_changed"]
            or item["top100_changed"]
        )

        if not changed:
            continue

        print()
        print(f"  {item['id']}")

        print(
            "    migration rank: "
            f"{item['migration_rank']['run_a']} "
            f"-> "
            f"{item['migration_rank']['run_b']}"
        )

        print(
            "    gold rank     : "
            f"{item['gold_rank']['run_a']} "
            f"-> "
            f"{item['gold_rank']['run_b']}"
        )

        print(
            "    migration sim : "
            f"{item['migration_similarity']['run_a']} "
            f"-> "
            f"{item['migration_similarity']['run_b']}"
        )

        print(
            "    gold sim      : "
            f"{item['gold_similarity']['run_a']} "
            f"-> "
            f"{item['gold_similarity']['run_b']}"
        )

        print(
            f"    Top10 changed : "
            f"{item['top10_changed']}"
        )

        print(
            f"    Top100 changed: "
            f"{item['top100_changed']}"
        )

    output = {
        "experiment": "E28-R",
        "description": (
            "Reproducibility check for E27 migration residual "
            "questions using the original vector retrieval pipeline."
        ),
        "config": {
            "candidate_limit": CANDIDATE_LIMIT,
            "run_count": RUN_COUNT,
            "residual_question_count": len(residual_ids),
        },
        "runs": runs,
        "comparison": comparison,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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
    print("RESULT")
    print("=" * 80)

    if comparison["all_top100_identical"]:
        print(
            "PASS: Run A and Run B have identical Top100."
        )
    else:
        print(
            "WARNING: Run A and Run B differ in Top100."
        )

    print()
    print(f"Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()