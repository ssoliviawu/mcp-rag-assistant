import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from retrieval.search import vector_search_diverse
from monitoring.logger import log_request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

RETRIEVAL_MODE = "vector_dedup"
CANDIDATE_LIMIT = 100
TOP_K = 10
MAX_CHUNKS_PER_SOURCE = 1
CONTEXT_MAX_CHARS = 12000
MAX_OUTPUT_TOKENS = 1500

MODEL = os.getenv("OPENAI_MODEL", )

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=90,
    max_retries=0,
)

# ============================================================
# Retrieval
# ============================================================

def retrieve(question):
    """
    E2 final retrieval pipeline.

    vector search
    + candidate_limit=100
    + source deduplication
    + max 1 chunk per source
    """

    return vector_search_diverse(
        question,
        limit=TOP_K,
        candidate_limit=CANDIDATE_LIMIT,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )


# ============================================================
# Normalize retrieval result
# ============================================================

def normalize_result(result, rank):

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

        item = normalize_result(
            result,
            rank,
        )

        retrieved_details.append(item)

    return retrieved_details


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

        if current_length + len(block) > CONTEXT_MAX_CHARS:

            remaining = (
                CONTEXT_MAX_CHARS
                - current_length
            )

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

    prompt = build_prompt(
        question,
        context,
    )

    start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "model": MODEL,
        "answer": response.output_text,
        "latency_ms": latency_ms,
    }


# ============================================================
# Main RAG function for Streamlit
# ============================================================

def answer_question(question):

    total_start = time.perf_counter()

    try:

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        raw_results = retrieve(question)

        retrieved_details = [
            normalize_result(result, rank)
            for rank, result in enumerate(
                raw_results,
                start=1,
            )
        ]

        # --------------------------------------------------------
        # Context
        # --------------------------------------------------------

        context = build_context(
            retrieved_details
        )

        # --------------------------------------------------------
        # Generation
        # --------------------------------------------------------

        generation = generate_answer(
            question,
            context,
        )

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000

        # --------------------------------------------------------
        # Monitoring
        # --------------------------------------------------------

        log_request(
            question=question,
            success=True,
            result_count=len(retrieved_details),
            model=generation["model"],
            generation_latency_ms=(
                generation["latency_ms"]
            ),
            total_latency_ms=total_latency_ms,
        )

        # --------------------------------------------------------
        # Return everything UI needs
        # --------------------------------------------------------

        return {
            "question": question,

            "answer": generation["answer"],

            "model": generation["model"],

            "latency_ms": total_latency_ms,

            "generation_latency_ms": (
                generation["latency_ms"]
            ),

            "retrieval": {
                "mode": RETRIEVAL_MODE,

                "candidate_limit": CANDIDATE_LIMIT,

                "top_k": TOP_K,

                "max_chunks_per_source": (
                    MAX_CHUNKS_PER_SOURCE
                ),

                "results": retrieved_details,

                "context": context,
            },
        }

    except Exception as error:

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000

        # --------------------------------------------------------
        # Monitoring failed request
        # --------------------------------------------------------

        log_request(
            question=question,
            success=False,
            total_latency_ms=total_latency_ms,
            error=str(error),
        )

        raise