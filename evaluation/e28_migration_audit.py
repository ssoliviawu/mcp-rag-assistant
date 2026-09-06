from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from embedding.model import BGEEmbeddingModel
from retrieval.search import search_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "mcp_rag_eval_dataset.jsonl"

E27_INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "e27_input.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CANDIDATE_LIMIT = 100

MIGRATION_SOURCE = (
    "https://py.sdk.modelcontextprotocol.io/migration/"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_source(source: str | None) -> str:
    if not source:
        return ""

    return source.rstrip("/")


def get_result_source(result: dict) -> str:
    return normalize_source(
        result.get("source")
        or result.get("url")
        or result.get("metadata", {}).get("source")
        or ""
    )


def get_result_title(result: dict) -> str:
    return (
        result.get("title")
        or result.get("metadata", {}).get("title")
        or ""
    )


def get_result_section(result: dict) -> str:
    return (
        result.get("section")
        or result.get("metadata", {}).get("section")
        or ""
    )


def get_result_content(result: dict) -> str:
    return result.get("content") or ""


def get_similarity(result: dict) -> float:
    value = (
        result.get("similarity")
        if result.get("similarity") is not None
        else result.get("score")
    )

    if value is None:
        return 0.0

    return float(value)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # Match previous experiments:
            # q025 / q041 are spec questions and excluded.
            if item.get("domain") == "spec":
                continue

            dataset.append(item)

    return dataset


# ---------------------------------------------------------------------------
# E27 migration residual IDs
# ---------------------------------------------------------------------------

def load_migration_residual_ids() -> list[str]:
    if not E27_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"E27 input not found: {E27_INPUT_PATH}"
        )

    with E27_INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []

    for item in data.get("questions", []):
        top10 = item.get("top10", [])

        if not top10:
            continue

        top1 = top10[0]

        source = normalize_source(
            top1.get("source")
            or top1.get("url")
            or ""
        )

        if source == normalize_source(MIGRATION_SOURCE):
            ids.append(item["id"])

    return ids


# ---------------------------------------------------------------------------
# Relevant sources
# ---------------------------------------------------------------------------

def get_relevant_sources(item: dict) -> set[str]:
    sources = item.get("relevant_sources", [])

    if isinstance(sources, str):
        return {normalize_source(sources)}

    return {
        normalize_source(source)
        for source in sources
        if source
    }


# ---------------------------------------------------------------------------
# Candidate serialization
# ---------------------------------------------------------------------------

def serialize_candidate(
    result: dict,
    rank: int,
) -> dict:
    return {
        "rank": rank,
        "title": get_result_title(result),
        "section": get_result_section(result),
        "source": get_result_source(result),
        "similarity": get_similarity(result),
        "content": get_result_content(result),
    }


# ---------------------------------------------------------------------------
# Find gold candidates
# ---------------------------------------------------------------------------

def find_gold_candidates(
    candidates: list[dict],
    relevant_sources: set[str],
) -> list[dict]:
    matches = []

    for rank, candidate in enumerate(candidates, start=1):
        source = get_result_source(candidate)

        if source in relevant_sources:
            matches.append(
                serialize_candidate(candidate, rank)
            )

    return matches


# ---------------------------------------------------------------------------
# Find migration candidates
# ---------------------------------------------------------------------------

def find_migration_candidates(
    candidates: list[dict],
) -> list[dict]:
    matches = []

    migration_source = normalize_source(MIGRATION_SOURCE)

    for rank, candidate in enumerate(candidates, start=1):
        source = get_result_source(candidate)

        if source == migration_source:
            matches.append(
                serialize_candidate(candidate, rank)
            )

    return matches


# ---------------------------------------------------------------------------
# Retrieve one question
# ---------------------------------------------------------------------------

def retrieve_question(
    model: BGEEmbeddingModel,
    item: dict,
) -> dict:

    question_id = item["id"]
    question = item["question"]

    relevant_sources = get_relevant_sources(item)

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    query_embedding = model.embed_query(question)

    # ------------------------------------------------------------------
    # Original C7 DB retrieval
    #
    # IMPORTANT:
    # We intentionally use the database result directly.
    #
    # This is an audit of the original candidate pool.
    # No re-embedding.
    # No representation modification.
    # No reranking.
    # No lexical boost.
    # ------------------------------------------------------------------

    candidates = search_chunks(
        query_embedding,
        limit=CANDIDATE_LIMIT,
    )

    # ------------------------------------------------------------------
    # Extract migration candidates
    # ------------------------------------------------------------------

    migration_candidates = find_migration_candidates(
        candidates
    )

    # ------------------------------------------------------------------
    # Extract gold candidates
    # ------------------------------------------------------------------

    gold_candidates = find_gold_candidates(
        candidates,
        relevant_sources,
    )

    # ------------------------------------------------------------------
    # Full Top100
    # ------------------------------------------------------------------

    top100 = [
        serialize_candidate(candidate, rank)
        for rank, candidate in enumerate(
            candidates,
            start=1,
        )
    ]

    # ------------------------------------------------------------------
    # Identify migration Top1
    # ------------------------------------------------------------------

    migration_top1 = None

    for candidate in migration_candidates:
        if candidate["rank"] == 1:
            migration_top1 = candidate
            break

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    return {
        "id": question_id,
        "question": question,
        "relevant_sources": sorted(relevant_sources),

        "migration_top1": migration_top1,

        "migration_candidates": migration_candidates,

        "gold_candidates": gold_candidates,

        "top100": top100,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 100)
    print("E28-D - MIGRATION FALSE POSITIVE AUDIT")
    print("=" * 100)

    print()
    print("Purpose:")
    print(
        "  Audit whether migration Top1 results are actually "
        "false positives."
    )
    print()
    print("This experiment DOES NOT:")
    print("  - modify embeddings")
    print("  - modify representation")
    print("  - rerank")
    print("  - apply lexical boost")
    print("  - suppress migration")
    print()
    print("It only retrieves the original Top100 and saves")
    print("the complete migration/gold chunk content.")
    print()

    dataset = load_dataset()

    dataset_by_id = {
        item["id"]: item
        for item in dataset
    }

    migration_ids = load_migration_residual_ids()

    print(
        f"Questions in benchmark : {len(dataset)}"
    )

    print(
        f"Migration residuals    : {len(migration_ids)}"
    )

    print(
        "IDs:",
        ", ".join(migration_ids),
    )

    print()
    print("Loading embedding model...")

    model = BGEEmbeddingModel()

    audit_results = []

    for index, question_id in enumerate(
        migration_ids,
        start=1,
    ):

        if question_id not in dataset_by_id:
            print(
                f"[WARNING] {question_id} not found in dataset"
            )
            continue

        item = dataset_by_id[question_id]

        print()
        print("-" * 100)
        print(
            f"[{index:02d}/{len(migration_ids):02d}] "
            f"{question_id}"
        )
        print(
            f"Question: {item['question']}"
        )

        result = retrieve_question(
            model,
            item,
        )

        audit_results.append(result)

        migration = result["migration_top1"]

        if migration:
            print()
            print("MIGRATION TOP1")
            print(
                f"  Rank       : {migration['rank']}"
            )
            print(
                f"  Similarity : {migration['similarity']:.6f}"
            )
            print(
                f"  Title      : {migration['title']}"
            )
            print(
                f"  Section    : {migration['section']}"
            )

        print()
        print("GOLD CANDIDATES")

        if not result["gold_candidates"]:
            print("  NONE FOUND IN TOP100")
        else:
            for gold in result["gold_candidates"]:
                print(
                    f"  rank={gold['rank']} "
                    f"score={gold['similarity']:.6f} "
                    f"title={gold['title']}"
                )

        print()
        print("MIGRATION CANDIDATES")

        if not result["migration_candidates"]:
            print("  NONE FOUND")
        else:
            for migration_candidate in result[
                "migration_candidates"
            ]:
                print(
                    f"  rank={migration_candidate['rank']} "
                    f"score={migration_candidate['similarity']:.6f} "
                    f"title={migration_candidate['title']}"
                )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "E28-D",
        "name": "migration_false_positive_audit",
        "dataset": str(DATASET_PATH),
        "question_count": len(dataset),
        "migration_residual_count": len(audit_results),
        "candidate_limit": CANDIDATE_LIMIT,
        "migration_source": MIGRATION_SOURCE,
        "purpose": (
            "Audit whether migration Top1 residual cases "
            "are true false positives or legitimate "
            "semantic competitors."
        ),
        "questions": audit_results,
    }

    output_path = (
        RESULTS_DIR
        / "e28_migration_audit.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("E28-D AUDIT COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Audited questions : {len(audit_results)}"
    )

    print()
    print(
        "Output:"
    )

    print(
        output_path
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  Inspect migration_top1.content and "
        "gold_candidates[*].content."
    )

    print()
    print(
        "Classify each case manually as:"
    )

    print(
        "  1. TRUE_FALSE_POSITIVE"
    )

    print(
        "  2. RELEVANT_COMPETITOR"
    )

    print(
        "  3. EQUIVALENT_EVIDENCE"
    )

    print(
        "  4. GOLD_STRONGLY_PREFERRED"
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()