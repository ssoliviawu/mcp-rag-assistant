import json
from pathlib import Path

from retrieval.search import vector_search, vector_search_diverse


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path("mcp_rag_eval_dataset.jsonl")
OUTPUT_PATH = Path("evaluation/results/e2_source_chunks_analysis.json")

# 先分析之前发现答案质量有问题的几个问题
TARGET_IDS = {
    "q021",
    "q022",
    "q024",
    "q048",
}

# 如果想分析全部有 gold source 的问题：
# TARGET_IDS = None

VECTOR_CANDIDATE_LIMIT = 100
E2_LIMIT = 10
MAX_CHUNKS_PER_SOURCE = 1

CONTENT_PREVIEW_LENGTH = 500


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    """Normalize URL for comparison."""
    if not url:
        return ""

    return url.strip().rstrip("/")


def get_source(result):
    """
    Get source URL.

    Keep the same logic as E2 as much as possible,
    but fallback to id for diagnostic purposes.
    """
    source = result.get("url")

    if not source:
        source = result.get("id")

    return source or ""


def get_similarity(result):
    """
    Try several common field names for similarity score.
    """
    for key in ("similarity", "score", "distance"):
        value = result.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def get_title(result):
    return (
        result.get("title")
        or result.get("name")
        or result.get("section")
        or ""
    )


def get_content(result):
    return result.get("content") or ""


def load_dataset():
    questions = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if TARGET_IDS is not None:
                if item.get("id") not in TARGET_IDS:
                    continue

            questions.append(item)

    return questions


# ============================================================
# Analyze one question
# ============================================================

def analyze_question(item):
    question_id = item["id"]
    question = item["question"]

    gold_sources = item.get("relevant_sources", [])

    normalized_gold_sources = {
        normalize_url(source)
        for source in gold_sources
        if source
    }

    print()
    print("=" * 100)
    print(f"{question_id.upper()} - SOURCE CHUNK ANALYSIS")
    print("=" * 100)
    print(f"Question: {question}")
    print()

    print("Gold sources:")
    for source in gold_sources:
        print(f"  - {source}")

    # --------------------------------------------------------
    # 1. Vector Top 100
    # --------------------------------------------------------

    vector_results = vector_search(
        question,
        limit=VECTOR_CANDIDATE_LIMIT,
    )

    print()
    print("-" * 100)
    print(f"VECTOR TOP {VECTOR_CANDIDATE_LIMIT}")
    print("-" * 100)

    gold_chunks = []

    for rank, result in enumerate(vector_results, start=1):
        source = get_source(result)
        normalized_source = normalize_url(source)

        if normalized_source not in normalized_gold_sources:
            continue

        similarity = get_similarity(result)
        title = get_title(result)
        content = get_content(result)

        chunk_info = {
            "vector_rank": rank,
            "similarity": similarity,
            "source": source,
            "title": title,
            "content": content,
        }

        gold_chunks.append(chunk_info)

    if not gold_chunks:
        print("  No gold-source chunks found in Vector Top 100.")
    else:
        print(
            f"Found {len(gold_chunks)} chunks from gold source(s) "
            f"in Vector Top {VECTOR_CANDIDATE_LIMIT}:"
        )

        for chunk in gold_chunks:
            print()
            print(
                f"  [Vector Rank {chunk['vector_rank']}] "
                f"similarity={chunk['similarity']}"
            )
            print(f"  Source : {chunk['source']}")
            print(f"  Title  : {chunk['title']}")

            preview = chunk["content"].replace("\n", " ").strip()

            if len(preview) > CONTENT_PREVIEW_LENGTH:
                preview = preview[:CONTENT_PREVIEW_LENGTH] + "..."

            print(f"  Content: {preview}")

    # --------------------------------------------------------
    # 2. E2 Top 10
    # --------------------------------------------------------

    e2_results = vector_search_diverse(
        question,
        limit=E2_LIMIT,
        candidate_limit=VECTOR_CANDIDATE_LIMIT,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )

    print()
    print("-" * 100)
    print(
        f"E2 TOP {E2_LIMIT} "
        f"(candidate={VECTOR_CANDIDATE_LIMIT}, "
        f"max_chunks_per_source={MAX_CHUNKS_PER_SOURCE})"
    )
    print("-" * 100)

    selected_gold_chunks = []

    for e2_rank, result in enumerate(e2_results, start=1):
        source = get_source(result)
        normalized_source = normalize_url(source)

        is_gold = normalized_source in normalized_gold_sources

        vector_rank = None

        # Find original Vector rank
        for rank, vector_result in enumerate(vector_results, start=1):
            if vector_result is result:
                vector_rank = rank
                break

        # Fallback matching if object identity does not work
        if vector_rank is None:
            result_id = result.get("id")

            for rank, vector_result in enumerate(vector_results, start=1):
                if result_id and vector_result.get("id") == result_id:
                    vector_rank = rank
                    break

        similarity = get_similarity(result)
        title = get_title(result)
        content = get_content(result)

        print()
        print(
            f"  [E2 Rank {e2_rank}] "
            f"Vector Rank={vector_rank} "
            f"similarity={similarity}"
        )
        print(f"  Gold   : {'YES' if is_gold else 'NO'}")
        print(f"  Source : {source}")
        print(f"  Title  : {title}")

        preview = content.replace("\n", " ").strip()

        if len(preview) > CONTENT_PREVIEW_LENGTH:
            preview = preview[:CONTENT_PREVIEW_LENGTH] + "..."

        print(f"  Content: {preview}")

        if is_gold:
            selected_gold_chunks.append(
                {
                    "e2_rank": e2_rank,
                    "vector_rank": vector_rank,
                    "similarity": similarity,
                    "source": source,
                    "title": title,
                    "content": content,
                }
            )

    # --------------------------------------------------------
    # 3. Mark which gold chunks survived
    # --------------------------------------------------------

    selected_ids = set()

    for result in e2_results:
        result_id = result.get("id")

        if result_id:
            selected_ids.add(result_id)

    # Build a more reliable comparison using object identity,
    # id, or content/source combination.
    e2_gold_keys = set()

    for result in e2_results:
        source = normalize_url(get_source(result))
        content = get_content(result)
        result_id = result.get("id")

        e2_gold_keys.add(
            (
                result_id,
                source,
                content,
            )
        )

    gold_chunk_analysis = []

    for chunk in gold_chunks:
        key = (
            None,
            normalize_url(chunk["source"]),
            chunk["content"],
        )

        selected = False

        for e2_result in e2_results:
            e2_key = (
                e2_result.get("id"),
                normalize_url(get_source(e2_result)),
                get_content(e2_result),
            )

            if key[1:] == e2_key[1:]:
                selected = True
                break

            if (
                chunk["content"]
                and chunk["content"] == get_content(e2_result)
            ):
                selected = True
                break

        gold_chunk_analysis.append(
            {
                **chunk,
                "selected_by_e2": selected,
            }
        )

    # --------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("SUMMARY")
    print("-" * 100)

    if not gold_chunks:
        status = "NOT_IN_VECTOR_100"

        print("Result: gold source is NOT present in Vector Top 100.")

    else:
        selected = [
            chunk
            for chunk in gold_chunk_analysis
            if chunk["selected_by_e2"]
        ]

        if selected:
            status = "GOLD_CHUNK_SELECTED"

            print(
                f"Gold chunks in Vector Top 100 : {len(gold_chunks)}"
            )
            print(
                f"Gold chunks selected by E2    : {len(selected)}"
            )

            for chunk in selected:
                print(
                    f"  Vector rank {chunk['vector_rank']} "
                    f"-> E2 selected"
                )

        else:
            status = "GOLD_CHUNKS_FILTERED"

            print(
                "Gold source exists in Vector Top 100, "
                "but no gold chunk survived E2 Top 10."
            )

    return {
        "id": question_id,
        "question": question,
        "gold_sources": gold_sources,
        "status": status,
        "vector_candidate_limit": VECTOR_CANDIDATE_LIMIT,
        "e2_limit": E2_LIMIT,
        "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,
        "gold_chunks_in_vector_top100": gold_chunk_analysis,
        "e2_gold_chunks": selected_gold_chunks,
        "e2_results": [
            {
                "e2_rank": rank,
                "source": get_source(result),
                "title": get_title(result),
                "similarity": get_similarity(result),
                "content": get_content(result),
            }
            for rank, result in enumerate(e2_results, start=1)
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    questions = load_dataset()

    print("=" * 100)
    print("E2 SOURCE CHUNK ANALYSIS")
    print("=" * 100)
    print(f"Questions: {len(questions)}")
    print(f"Vector candidates: {VECTOR_CANDIDATE_LIMIT}")
    print(f"E2 limit: {E2_LIMIT}")
    print(f"Max chunks per source: {MAX_CHUNKS_PER_SOURCE}")
    print()

    results = []

    for item in questions:
        try:
            result = analyze_question(item)
            results.append(result)

        except Exception as e:
            print()
            print(f"[ERROR] {item.get('id')}: {e}")

            results.append(
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "status": "ERROR",
                    "error": str(e),
                }
            )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "config": {
            "dataset": str(DATASET_PATH),
            "vector_candidate_limit": VECTOR_CANDIDATE_LIMIT,
            "e2_limit": E2_LIMIT,
            "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,
            "target_ids": (
                sorted(TARGET_IDS)
                if TARGET_IDS is not None
                else None
            ),
        },
        "results": results,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary = {}

    for result in results:
        status = result.get("status", "UNKNOWN")
        summary[status] = summary.get(status, 0) + 1

    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    for status, count in sorted(summary.items()):
        print(f"{status}: {count}")

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()