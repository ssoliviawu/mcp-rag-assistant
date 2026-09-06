import json
import os
import time
from pathlib import Path

from openai import OpenAI

from embedding.model import BGEEmbeddingModel
from retrieval.search import vector_search_diverse


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "mcp_rag_eval_dataset.jsonl"

OUTPUT_PATH = (
    ROOT_DIR
    / "evaluation"
    / "results"
    / "e2_final_comparison_generation_details.json"
)


# ============================================================
# Experiment configuration
# ============================================================

EXPERIMENT = "E2_final_comparison_generation"

RETRIEVAL_MODE = "vector_dedup"
CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1

CONTEXT_MAX_CHARS = 12000

MODEL = os.getenv("OPENAI_MODEL", "agnes-2.5-flash")
MAX_OUTPUT_TOKENS = 1500


# ============================================================
# Client
# ============================================================

client = OpenAI(
    timeout=90,
    max_retries=0,
)


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    questions = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            # Keep exactly the same 48-question subset as E30
            if item.get("id") in {"q025", "q041"}:
                continue

            questions.append(item)

    return questions


# ============================================================
# Retrieval
# ============================================================

def retrieve(question):
    """
    Historical E2 retrieval pipeline.

    This is intentionally kept unchanged:
        vector_search_diverse
        candidate_limit=100
        top_k=10
        max_chunks_per_source=1
    """

    return vector_search_diverse(
        question,
        limit=CANDIDATE_LIMIT,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )


# ============================================================
# Normalize retrieval result
# ============================================================

def normalize_result(result, rank):
    """
    Convert retrieval output into the exact schema expected
    by the common generation evaluator.
    """

    if not isinstance(result, dict):
        return {
            "rank": rank,
            "title": "",
            "section": "",
            "url": "",
            "content": str(result),
        }

    url = (
        result.get("url")
        or result.get("source")
        or result.get("source_url")
        or ""
    )

    return {
        "rank": rank,
        "title": result.get("title", ""),
        "section": result.get("section", ""),
        "url": url,
        "content": result.get("content", ""),
    }


# ============================================================
# Build retrieval details
# ============================================================

def build_retrieved_details(results):
    retrieved_details = []

    for rank, result in enumerate(results, start=1):
        item = normalize_result(result, rank)
        retrieved_details.append(item)

    return retrieved_details


# ============================================================
# Retrieved URLs
# ============================================================

def build_retrieved_urls(retrieved_details):
    urls = []

    for item in retrieved_details:
        url = item.get("url")

        if url and url not in urls:
            urls.append(url)

    return urls


# ============================================================
# Retrieval hit
# ============================================================

def calculate_retrieval_hit(item, retrieved_urls):
    """
    Determine whether retrieval found a relevant source.

    The dataset's relevant_sources are treated as the ground truth.
    """

    relevant_sources = item.get("relevant_sources", [])

    if not relevant_sources:
        return False

    relevant_set = {
        str(url).rstrip("/")
        for url in relevant_sources
        if url
    }

    retrieved_set = {
        str(url).rstrip("/")
        for url in retrieved_urls
        if url
    }

    return bool(relevant_set & retrieved_set)


# ============================================================
# Build context
# ============================================================

def build_context(retrieved_details):
    parts = []
    current_length = 0

    for item in retrieved_details:

        block = (
            f"[Document {item['rank']}]\n"
            f"Title: {item['title']}\n"
            f"Section: {item['section']}\n"
            f"URL: {item['url']}\n"
            f"Content:\n"
            f"{item['content']}\n"
        )

        # Preserve the same total context limit
        if current_length + len(block) > CONTEXT_MAX_CHARS:

            remaining = CONTEXT_MAX_CHARS - current_length

            if remaining <= 0:
                break

            block = block[:remaining]

        parts.append(block)
        current_length += len(block)

        if current_length >= CONTEXT_MAX_CHARS:
            break

    return "\n".join(parts)


# ============================================================
# Prompt
# ============================================================

def build_prompt(question, context):
    return f"""
You are an assistant answering questions about the MCP Python SDK.

Answer the user's question using the provided documentation context.

Rules:
1. Use the retrieved documentation as the primary source.
2. Do not invent information that is not supported by the context.
3. If the documentation does not provide enough information, say so.
4. Give a concise and technically accurate answer.
5. When appropriate, mention relevant URLs from the provided documentation.

Question:
{question}

Documentation context:
{context}

Answer:
""".strip()


# ============================================================
# Generation
# ============================================================

def generate_answer(question, context):

    prompt = build_prompt(question, context)

    start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    answer = response.output_text

    usage = None

    if getattr(response, "usage", None):
        usage = {
            "input_tokens": getattr(
                response.usage,
                "input_tokens",
                None,
            ),
            "output_tokens": getattr(
                response.usage,
                "output_tokens",
                None,
            ),
            "total_tokens": getattr(
                response.usage,
                "total_tokens",
                None,
            ),
        }

    return {
        "model": MODEL,
        "answer": answer,
        "usage": usage,
        "latency_ms": latency_ms,
    }


# ============================================================
# Main
# ============================================================

def main():

    questions = load_dataset()

    print("=" * 80)
    print("E2 FINAL COMPARISON GENERATION")
    print("=" * 80)

    print(f"Dataset       : {DATASET_PATH}")
    print(f"Questions     : {len(questions)}")
    print(f"Retrieval     : {RETRIEVAL_MODE}")
    print(f"Candidate     : {CANDIDATE_LIMIT}")
    print(f"Top K         : {TOP_K}")
    print(f"Max/source    : {MAX_CHUNKS_PER_SOURCE}")
    print(f"Model         : {MODEL}")
    print()

    results = []

    total_start = time.perf_counter()

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index:02d}/{len(questions)}] "
            f"{question_id} - {question}"
        )

        item_start = time.perf_counter()

        try:

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            raw_results = retrieve(question)

            retrieved_details = build_retrieved_details(
                raw_results
            )

            retrieved_urls = build_retrieved_urls(
                retrieved_details
            )

            retrieval_hit = calculate_retrieval_hit(
                item,
                retrieved_urls,
            )

            # ------------------------------------------------
            # Context
            # ------------------------------------------------

            context = build_context(
                retrieved_details
            )

            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            generation = generate_answer(
                question,
                context,
            )

            total_latency_ms = (
                time.perf_counter() - item_start
            ) * 1000

            # ------------------------------------------------
            # IMPORTANT:
            # This schema intentionally matches E30.
            # ------------------------------------------------

            output_item = {
                "id": question_id,

                "question": question,

                "retrieval": {
                    "mode": RETRIEVAL_MODE,
                    "candidate_limit": CANDIDATE_LIMIT,
                    "top_k": TOP_K,
                    "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,

                    "hit": retrieval_hit,

                    "retrieved_urls": retrieved_urls,

                    "retrieved_details": retrieved_details,

                    "context": context,
                },

                "generation": generation,

                "total_latency_ms": total_latency_ms,

                "status": "success",
            }

            results.append(output_item)

            print(
                f"    hit={retrieval_hit} "
                f"docs={len(retrieved_details)} "
                f"latency={total_latency_ms:.1f} ms"
            )

        except Exception as e:

            total_latency_ms = (
                time.perf_counter() - item_start
            ) * 1000

            output_item = {
                "id": question_id,

                "question": question,

                "retrieval": {
                    "mode": RETRIEVAL_MODE,
                    "candidate_limit": CANDIDATE_LIMIT,
                    "top_k": TOP_K,
                    "max_chunks_per_source": MAX_CHUNKS_PER_SOURCE,

                    "hit": False,

                    "retrieved_urls": [],

                    "retrieved_details": [],

                    "context": "",
                },

                "generation": {
                    "model": MODEL,
                    "answer": "",
                    "usage": None,
                    "latency_ms": 0,
                },

                "total_latency_ms": total_latency_ms,

                "status": "failed",

                "error": str(e),
            }

            results.append(output_item)

            print(
                f"    ERROR: {e}"
            )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_wall_clock_ms = (
        time.perf_counter() - total_start
    ) * 1000

    successful = sum(
        1
        for x in results
        if x.get("status") == "success"
    )

    failed = len(results) - successful

    hits = sum(
        1
        for x in results
        if x.get("retrieval", {}).get("hit") is True
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Questions       : {len(results)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Retrieval hits  : {hits}")
    print(f"Hit rate        : {hits / len(results):.4f}")
    print(f"Wall clock      : {total_wall_clock_ms:.1f} ms")
    print(f"Output          : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()