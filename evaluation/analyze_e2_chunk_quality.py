import json
import os
import re
from collections import Counter

from retrieval.search import vector_search, vector_search_diverse


# ============================================================
# Config
# ============================================================

DATASET_PATH = "mcp_rag_eval_dataset.jsonl"
OUTPUT_PATH = "evaluation/results/e2_chunk_quality_analysis.json"

CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

# 只输出至少包含多个同 source chunk 的问题
SHOW_ONLY_MULTIPLE_SOURCE_CHUNKS = True


# ============================================================
# Helpers
# ============================================================

def normalize_url(url):
    if not url:
        return ""
    return url.strip().rstrip("/")


def get_similarity(result):
    """
    兼容不同字段命名。
    """
    for key in ("similarity", "score", "distance"):
        value = result.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    return None


def get_title(result):
    return (
        result.get("title")
        or result.get("name")
        or ""
    )


def get_section(result):
    return (
        result.get("section")
        or result.get("heading")
        or result.get("section_title")
        or ""
    )


def get_content(result):
    return result.get("content") or ""


def tokenize(text):
    """
    非严格的英文 tokenization。
    这里只作为辅助诊断，不代表真正 relevance。
    """
    text = text.lower()
    return set(re.findall(r"\b[a-zA-Z0-9_]+\b", text))


def lexical_overlap(question, result):
    """
    计算 question 与 chunk 的简单 token overlap。

    这只是诊断指标，不等同于语义相关性。
    """
    q_tokens = tokenize(question)

    if not q_tokens:
        return 0.0

    text = " ".join(
        [
            get_title(result),
            get_section(result),
            get_content(result),
        ]
    )

    chunk_tokens = tokenize(text)

    if not chunk_tokens:
        return 0.0

    return len(q_tokens & chunk_tokens) / len(q_tokens)


def content_preview(content, length=300):
    content = " ".join(content.split())

    if len(content) <= length:
        return content

    return content[:length] + "..."


def load_dataset(path):
    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            dataset.append(json.loads(line))

    return dataset


# ============================================================
# Analyze one question
# ============================================================

def analyze_question(item):
    question_id = item["id"]
    question = item["question"]

    relevant_sources = {
        normalize_url(source)
        for source in item.get("relevant_sources", [])
        if source
    }

    # --------------------------------------------------------
    # No gold source
    # --------------------------------------------------------

    if not relevant_sources:
        return {
            "id": question_id,
            "question": question,
            "status": "NO_GOLD_SOURCE",
            "gold_sources": [],
            "gold_source_candidates": [],
        }

    # --------------------------------------------------------
    # Vector Top 100
    # --------------------------------------------------------

    candidates = vector_search(
        question,
        limit=CANDIDATE_LIMIT,
    )

    # --------------------------------------------------------
    # E2 Top 10
    # --------------------------------------------------------

    e2_results = vector_search_diverse(
        question,
        limit=TOP_K,
        candidate_limit=CANDIDATE_LIMIT,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )

    # --------------------------------------------------------
    # Build candidate records
    # --------------------------------------------------------

    vector_candidates = []

    for rank, result in enumerate(candidates, start=1):
        source = normalize_url(result.get("url"))

        vector_candidates.append(
            {
                "rank": rank,
                "source": source,
                "title": get_title(result),
                "section": get_section(result),
                "similarity": get_similarity(result),
                "lexical_overlap": round(
                    lexical_overlap(question, result),
                    4,
                ),
                "content_preview": content_preview(
                    get_content(result)
                ),
            }
        )

    # --------------------------------------------------------
    # E2 selected candidates
    # --------------------------------------------------------

    e2_selected = []

    for e2_rank, result in enumerate(e2_results, start=1):
        source = normalize_url(result.get("url"))

        # 找回这个 chunk 在 Vector Top100 中的 rank
        original_rank = None

        for candidate in vector_candidates:
            if (
                candidate["source"] == source
                and candidate["title"] == get_title(result)
                and candidate["content_preview"]
                == content_preview(get_content(result))
            ):
                original_rank = candidate["rank"]
                break

        e2_selected.append(
            {
                "e2_rank": e2_rank,
                "vector_rank": original_rank,
                "source": source,
                "title": get_title(result),
                "section": get_section(result),
                "similarity": get_similarity(result),
                "lexical_overlap": round(
                    lexical_overlap(question, result),
                    4,
                ),
                "content_preview": content_preview(
                    get_content(result)
                ),
                "is_gold_source": source in relevant_sources,
            }
        )

    # --------------------------------------------------------
    # Analyze every gold source
    # --------------------------------------------------------

    gold_source_analysis = []

    for gold_source in sorted(relevant_sources):

        source_candidates = [
            candidate
            for candidate in vector_candidates
            if candidate["source"] == gold_source
        ]

        # 按 vector rank 已经排序
        source_candidates.sort(
            key=lambda x: x["rank"]
        )

        selected = None

        for result in e2_selected:
            if result["source"] == gold_source:
                selected = result
                break

        # E2 没选中该 gold source
        if selected is None:
            selected_rank = None
            selected_similarity = None
        else:
            selected_rank = selected["vector_rank"]
            selected_similarity = selected["similarity"]

        # 找该 source 相似度最高的 chunk
        similarity_candidates = [
            c
            for c in source_candidates
            if c["similarity"] is not None
        ]

        if similarity_candidates:
            best_similarity_candidate = max(
                similarity_candidates,
                key=lambda x: x["similarity"],
            )
        else:
            best_similarity_candidate = None

        # 同 source 的其他 chunk
        later_chunks = []

        if selected_rank is not None:
            later_chunks = [
                c
                for c in source_candidates
                if c["rank"] > selected_rank
            ]

        similarity_gap = None

        if (
            selected_similarity is not None
            and best_similarity_candidate is not None
            and best_similarity_candidate["similarity"] is not None
        ):
            similarity_gap = round(
                best_similarity_candidate["similarity"]
                - selected_similarity,
                6,
            )

        gold_source_analysis.append(
            {
                "source": gold_source,

                "vector_candidate_count": len(
                    source_candidates
                ),

                "selected_by_e2": selected is not None,

                "selected_vector_rank": selected_rank,

                "selected_similarity": selected_similarity,

                "best_similarity_vector_rank": (
                    best_similarity_candidate["rank"]
                    if best_similarity_candidate
                    else None
                ),

                "best_similarity": (
                    best_similarity_candidate["similarity"]
                    if best_similarity_candidate
                    else None
                ),

                "similarity_gap": similarity_gap,

                "has_multiple_chunks": (
                    len(source_candidates) > 1
                ),

                "later_same_source_chunk_count": len(
                    later_chunks
                ),

                "selected_chunk": selected,

                "all_same_source_chunks": source_candidates,
            }
        )

    # --------------------------------------------------------
    # Question-level flags
    # --------------------------------------------------------

    multiple_source_chunk_count = sum(
        1
        for source in gold_source_analysis
        if source["has_multiple_chunks"]
    )

    has_later_same_source_chunks = any(
        source["later_same_source_chunk_count"] > 0
        for source in gold_source_analysis
        if source["selected_by_e2"]
    )

    # 是否存在同 source 后面的 chunk lexical overlap 更高
    possible_better_later_chunk = False

    for source in gold_source_analysis:

        selected = source["selected_chunk"]

        if not selected:
            continue

        selected_overlap = selected["lexical_overlap"]

        for later in source["all_same_source_chunks"]:

            if later["rank"] <= selected["vector_rank"]:
                continue

            if later["lexical_overlap"] > selected_overlap:
                possible_better_later_chunk = True
                break

        if possible_better_later_chunk:
            break

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if has_later_same_source_chunks:
        status = "HAS_MULTIPLE_GOLD_SOURCE_CHUNKS"
    else:
        status = "SINGLE_GOLD_SOURCE_CHUNK"

    return {
        "id": question_id,
        "question": question,
        "status": status,

        "gold_sources": sorted(relevant_sources),

        "vector_top100_count": len(candidates),

        "e2_top10_count": len(e2_results),

        "gold_source_analysis": gold_source_analysis,

        "e2_selected": e2_selected,

        "flags": {
            "multiple_gold_source_chunks": (
                multiple_source_chunk_count > 0
            ),
            "has_later_same_source_chunks": (
                has_later_same_source_chunks
            ),
            "possible_better_later_chunk": (
                possible_better_later_chunk
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("E2 CHUNK QUALITY ANALYSIS")
    print("=" * 80)

    print(f"Dataset       : {DATASET_PATH}")
    print(f"Candidate     : {CANDIDATE_LIMIT}")
    print(f"Top K         : {TOP_K}")
    print(
        f"Max/source    : {MAX_CHUNKS_PER_SOURCE}"
    )
    print()

    dataset = load_dataset(DATASET_PATH)

    print(
        f"Loaded questions: {len(dataset)}"
    )
    print()

    results = []

    for index, item in enumerate(dataset, start=1):

        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{item['id']}"
        )

        analysis = analyze_question(item)

        results.append(analysis)

    # ========================================================
    # Summary
    # ========================================================

    status_counts = Counter(
        result["status"]
        for result in results
    )

    multiple_chunk_questions = [
        result
        for result in results
        if result.get("flags", {}).get(
            "multiple_gold_source_chunks",
            False,
        )
    ]

    later_chunk_questions = [
        result
        for result in results
        if result.get("flags", {}).get(
            "has_later_same_source_chunks",
            False,
        )
    ]

    possible_better_questions = [
        result
        for result in results
        if result.get("flags", {}).get(
            "possible_better_later_chunk",
            False,
        )
    ]

    # ========================================================
    # Save
    # ========================================================

    output = {
        "experiment": "E2 chunk quality",

        "dataset": DATASET_PATH,

        "parameters": {
            "candidate_limit": CANDIDATE_LIMIT,
            "top_k": TOP_K,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
        },

        "summary": {
            "question_count": len(results),

            "status_counts": dict(status_counts),

            "questions_with_multiple_gold_chunks": len(
                multiple_chunk_questions
            ),

            "questions_with_later_same_source_chunks": len(
                later_chunk_questions
            ),

            "questions_with_possible_better_later_chunk": len(
                possible_better_questions
            ),
        },

        "results": results,
    }

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

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
    # Console report
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Questions                                  : "
        f"{len(results)}"
    )

    print(
        f"Multiple gold-source chunks                : "
        f"{len(multiple_chunk_questions)}"
    )

    print(
        f"Later same-source chunks                   : "
        f"{len(later_chunk_questions)}"
    )

    print(
        f"Possible better later chunk                : "
        f"{len(possible_better_questions)}"
    )

    print()
    print("=" * 80)
    print("QUESTIONS WORTH INSPECTING")
    print("=" * 80)

    for result in results:

        flags = result.get("flags", {})

        if not (
            flags.get("has_later_same_source_chunks")
            or flags.get("possible_better_later_chunk")
        ):
            continue

        print()
        print("-" * 80)

        print(
            f"{result['id']}: "
            f"{result['question']}"
        )

        for source_info in result[
            "gold_source_analysis"
        ]:

            if not source_info[
                "selected_by_e2"
            ]:
                continue

            if not source_info[
                "has_multiple_chunks"
            ]:
                continue

            print()
            print(
                f"Source: {source_info['source']}"
            )

            print(
                f"E2 selected vector rank: "
                f"{source_info['selected_vector_rank']}"
            )

            print(
                f"E2 selected similarity: "
                f"{source_info['selected_similarity']}"
            )

            print(
                f"Best similarity rank: "
                f"{source_info['best_similarity_vector_rank']}"
            )

            print(
                f"Best similarity: "
                f"{source_info['best_similarity']}"
            )

            print(
                f"Similarity gap: "
                f"{source_info['similarity_gap']}"
            )

            print(
                f"Same-source chunks: "
                f"{source_info['vector_candidate_count']}"
            )

            print()
            print("Chunks:")

            for chunk in source_info[
                "all_same_source_chunks"
            ]:

                marker = (
                    " <-- E2 SELECTED"
                    if chunk["rank"]
                    == source_info[
                        "selected_vector_rank"
                    ]
                    else ""
                )

                print(
                    f"  rank={chunk['rank']:>3} "
                    f"sim={chunk['similarity']} "
                    f"overlap={chunk['lexical_overlap']:.4f} "
                    f"title={chunk['title']}"
                    f"{marker}"
                )

    print()
    print("=" * 80)

    print(
        f"Saved analysis to: {OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()