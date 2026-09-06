
"""
End-to-End RAG Generation Evaluation

Pipeline:

    Evaluation Dataset
            ↓
    E2: Vector + Source Deduplication
            ↓
    candidate_limit = 100
            ↓
    top_k = 10
            ↓
       Build Context
            ↓
          LLM
            ↓
        Final Answer
            ↓
    Save Evaluation Results

E2 Retrieval:

    vector_search_diverse(
        query,
        limit=10,
        candidate_limit=100,
        max_chunks_per_source=1,
    )

Execution strategy:

    - Process questions in small batches
    - Sequential requests inside each batch
    - Wait between batches
    - Save results after every batch
    - One failed question does not stop the evaluation
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.search import vector_search_diverse


# ============================================================
# Configuration
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Dataset
# ============================================================

DATASET_PATH = (
    BASE_DIR
    / "mcp_rag_eval_dataset.jsonl"
)


# ============================================================
# Results
# ============================================================

RESULTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "results"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "rag_generation_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "rag_generation_details.json"
)


# ============================================================
# Model
# ============================================================

MODEL = os.getenv("OPENAI_MODEL")

if not MODEL:
    raise RuntimeError(
        "OPENAI_MODEL is not set in the environment."
    )


# ============================================================
# Retrieval
# ============================================================

# E2:
# Vector search + source-level deduplication

RETRIEVAL_MODE = "vector_dedup"

RETRIEVAL_LIMIT = 10

RETRIEVAL_CANDIDATE_LIMIT = 100

MAX_CHUNKS_PER_SOURCE = 1


# ============================================================
# Context
# ============================================================

MAX_CONTEXT_CHARS = 12_000


# ============================================================
# Generation
# ============================================================

MAX_OUTPUT_TOKENS = 1000


# ============================================================
# Request
# ============================================================

REQUEST_TIMEOUT_SECONDS = 90


# ============================================================
# Batch
# ============================================================

BATCH_SIZE = 3

BATCH_DELAY_SECONDS = 4


# ============================================================
# Dataset
# ============================================================

def load_dataset(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# Retrieval
# ============================================================

def retrieve_context(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    candidate_limit: int = RETRIEVAL_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """
    E2 retrieval:

        Vector Search
              ↓
        100 candidates
              ↓
        Source Deduplication
              ↓
        Top 10
    """

    return vector_search_diverse(
        query=question,
        limit=limit,
        candidate_limit=candidate_limit,
        max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
    )


# ============================================================
# Context
# ============================================================

def build_context(
    retrieved_details: list[dict[str, Any]],
) -> str:

    context_parts = []

    total_chars = 0

    for item in retrieved_details:

        rank = item.get(
            "rank",
            "",
        )

        title = item.get(
            "title",
            "",
        )

        section = item.get(
            "section",
            "",
        )

        url = item.get(
            "url",
            "",
        )

        content = item.get(
            "content",
            "",
        )

        if not content:

            content = item.get(
                "content_preview",
                "",
            )

        chunk_text = (
            f"[Source {rank}]\n"
            f"Title: {title}\n"
            f"Section: {section}\n"
            f"URL: {url}\n\n"
            f"{content}\n"
        )

        if (
            total_chars
            + len(chunk_text)
            > MAX_CONTEXT_CHARS
        ):
            break

        context_parts.append(
            chunk_text
        )

        total_chars += len(
            chunk_text
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an MCP documentation assistant.

Answer the user's question using ONLY the provided documentation
context.

Rules:

1. Do not invent information that is not supported by the context.

2. Prefer precise technical explanations and use the terminology
   from the provided documentation.

3. Answer the user's question directly before providing additional
   explanation.

   * If the question asks "why", "what is the purpose", or "what is
     it for", state the purpose first.
   * If the question asks "how", explain the procedure or mechanism
     first.

4. If the documentation explicitly lists multiple items, include the
   complete list rather than replacing it with a generic summary.

5. Do not omit important details that are explicitly stated in the
   provided documentation, especially required parameters, optional
   parameters, supported types, limitations, or configuration options.

6. If the context does not contain enough information to answer,
   explicitly say that the provided documentation is insufficient.
   Do not guess or rely on outside knowledge.

7. Cite the source URL(s) that directly support your answer.
   Every factual answer should include at least one relevant source URL
   when such a URL is available in the provided context.

8. Preserve source URLs exactly as they appear in the provided context.
   Do not shorten, modify, or truncate URLs.

9. Keep the answer focused on the user's question. Avoid unnecessary
   background information.

10. When the question asks for a specific fact, give the specific fact
    first, followed by a concise explanation if needed.
"""


def build_prompt(
    question: str,
    context: str,
) -> str:

    return f"""
{SYSTEM_PROMPT}

================ DOCUMENTATION CONTEXT ================

{context}

================ END DOCUMENTATION CONTEXT ================

User question:

{question}

Answer the question based only on the documentation context.
"""


# ============================================================
# LLM Generation
# ============================================================

def generate_answer(
    client: OpenAI,
    question: str,
    context: str,
) -> tuple[str, dict[str, Any]]:

    prompt = build_prompt(
        question=question,
        context=context,
    )

    print(
        f"    Prompt size: "
        f"{len(prompt):,} chars"
    )

    api_start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    api_latency_ms = (
        time.perf_counter()
        - api_start
    ) * 1000

    print(
        f"    API call: "
        f"{api_latency_ms:.2f} ms"
    )

    answer = response.output_text.strip()

    usage = {}

    if response.usage:

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

    return answer, usage


# ============================================================
# Single Question Evaluation
# ============================================================

def evaluate_question(
    client: OpenAI,
    record: dict[str, Any],
) -> dict[str, Any]:

    question_id = record["id"]

    question = record["question"]

    relevant_sources = record.get(
        "relevant_sources",
        [],
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    retrieved_details = retrieve_context(
        question=question,
        limit=RETRIEVAL_LIMIT,
        candidate_limit=RETRIEVAL_CANDIDATE_LIMIT,
    )

    retrieval_latency_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_start = time.perf_counter()

    context = build_context(
        retrieved_details
    )

    context_latency_ms = (
        time.perf_counter()
        - context_start
    ) * 1000

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_start = time.perf_counter()

    answer, usage = generate_answer(
        client=client,
        question=question,
        context=context,
    )

    generation_latency_ms = (
        time.perf_counter()
        - generation_start
    ) * 1000

    # --------------------------------------------------------
    # Retrieval Hit
    # --------------------------------------------------------

    processing_start = time.perf_counter()

    retrieved_urls = []

    for item in retrieved_details:

        url = item.get("url")

        if url:
            retrieved_urls.append(url)

    relevant_source_set = set(
        relevant_sources
    )

    retrieved_source_set = set(
        retrieved_urls
    )

    retrieval_hit = bool(
        relevant_source_set
        & retrieved_source_set
    )

    processing_latency_ms = (
        time.perf_counter()
        - processing_start
    ) * 1000

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        f"    Retrieval: "
        f"{retrieval_latency_ms:.2f} ms"
    )

    print(
        f"    Context: "
        f"{context_latency_ms:.2f} ms"
    )

    print(
        f"    Generation: "
        f"{generation_latency_ms:.2f} ms"
    )

    print(
        f"    Processing: "
        f"{processing_latency_ms:.2f} ms"
    )

    print(
        f"    Context size: "
        f"{len(context):,} chars"
    )

    print(
        f"    Retrieved chunks: "
        f"{len(retrieved_details)}"
    )

    print(
        f"    Retrieval hit: "
        f"{retrieval_hit}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "id": question_id,

        "question": question,

        "relevant_sources": relevant_sources,

        "retrieval": {
            "mode": RETRIEVAL_MODE,

            "candidate_limit": (
                RETRIEVAL_CANDIDATE_LIMIT
            ),

            "top_k": RETRIEVAL_LIMIT,

            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),

            "latency_ms": round(
                retrieval_latency_ms,
                2,
            ),

            "hit": retrieval_hit,

            "retrieved_urls": retrieved_urls,

            "retrieved_details": retrieved_details,
        },

        "context": context,

        "generation": {
            "model": MODEL,

            "latency_ms": round(
                generation_latency_ms,
                2,
            ),

            "answer": answer,

            "usage": usage,
        },

        "timing": {
            "retrieval_ms": round(
                retrieval_latency_ms,
                2,
            ),

            "context_ms": round(
                context_latency_ms,
                2,
            ),

            "generation_ms": round(
                generation_latency_ms,
                2,
            ),

            "processing_ms": round(
                processing_latency_ms,
                2,
            ),
        },
    }


# ============================================================
# Error Result
# ============================================================

def build_error_result(
    record: dict[str, Any],
    error: Exception,
    latency_ms: float,
) -> dict[str, Any]:

    return {
        "id": record["id"],

        "question": record["question"],

        "relevant_sources": record.get(
            "relevant_sources",
            [],
        ),

        "error": {
            "type": type(error).__name__,

            "message": str(error),

            "latency_ms": round(
                latency_ms,
                2,
            ),
        },
    }


# ============================================================
# Save Details
# ============================================================

def save_details(
    details: list[dict[str, Any]],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Summary
# ============================================================

def build_summary(
    details: list[dict[str, Any]],
    total_latency_ms: float,
) -> dict[str, Any]:

    question_count = len(
        details
    )

    successful = [
        item
        for item in details
        if "error" not in item
    ]

    errors = [
        item
        for item in details
        if "error" in item
    ]

    retrieval_hits = sum(
        1
        for item in successful
        if item["retrieval"]["hit"]
    )

    retrieval_hit_rate = (
        retrieval_hits / len(successful)
        if successful
        else 0
    )

    retrieval_latencies = [
        item["retrieval"]["latency_ms"]
        for item in successful
    ]

    generation_latencies = [
        item["generation"]["latency_ms"]
        for item in successful
    ]

    return {
        "dataset": str(
            DATASET_PATH
        ),

        "question_count": question_count,

        "successful_questions": len(
            successful
        ),

        "failed_questions": len(
            errors
        ),

        "retrieval": {
            "mode": RETRIEVAL_MODE,

            "candidate_limit": (
                RETRIEVAL_CANDIDATE_LIMIT
            ),

            "top_k": RETRIEVAL_LIMIT,

            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),

            "retrieval_hit_rate": round(
                retrieval_hit_rate,
                4,
            ),

            "retrieval_hits": retrieval_hits,

            "retrieval_misses": (
                len(successful)
                - retrieval_hits
            ),

            "avg_latency_ms": round(
                sum(retrieval_latencies)
                / len(retrieval_latencies),
                2,
            )
            if retrieval_latencies
            else 0,
        },

        "generation": {
            "model": MODEL,

            "avg_latency_ms": round(
                sum(generation_latencies)
                / len(generation_latencies),
                2,
            )
            if generation_latencies
            else 0,
        },

        "errors": [
            {
                "id": item["id"],

                "type": item["error"]["type"],

                "message": item["error"]["message"],
            }
            for item in errors
        ],

        "end_to_end": {
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),

            "avg_latency_ms": round(
                total_latency_ms
                / question_count,
                2,
            )
            if question_count
            else 0,
        },

        "batch": {
            "batch_size": BATCH_SIZE,

            "batch_delay_seconds": (
                BATCH_DELAY_SECONDS
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("MCP RAG END-TO-END GENERATION EVALUATION")
    print("=" * 80)

    print()

    print(
        f"Dataset : {DATASET_PATH}"
    )

    print(
        f"Model   : {MODEL}"
    )

    print(
        f"Retrieval: {RETRIEVAL_MODE}"
    )

    print(
        f"Candidate limit: "
        f"{RETRIEVAL_CANDIDATE_LIMIT}"
    )

    print(
        f"Top-K   : {RETRIEVAL_LIMIT}"
    )

    print(
        f"Max chunks/source: "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Batch delay: "
        f"{BATCH_DELAY_SECONDS}s"
    )

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"Loaded {len(dataset)} "
        f"evaluation questions."
    )

    print()

    # --------------------------------------------------------
    # OpenAI client
    # --------------------------------------------------------

    client = OpenAI(
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    details = []

    evaluation_start = (
        time.perf_counter()
    )

    total_questions = len(
        dataset
    )

    total_batches = (
        total_questions
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_index in range(
        total_batches
    ):

        batch_start = (
            batch_index
            * BATCH_SIZE
        )

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_questions,
        )

        batch = dataset[
            batch_start:batch_end
        ]

        print()
        print("=" * 80)

        print(
            f"BATCH "
            f"{batch_index + 1}/"
            f"{total_batches}"
        )

        print(
            f"Questions "
            f"{batch_start + 1}-"
            f"{batch_end}"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # Sequential questions inside batch
        # ----------------------------------------------------

        for local_index, record in enumerate(
            batch,
            start=1,
        ):

            global_index = (
                batch_start
                + local_index
            )

            question_id = record["id"]

            question = record["question"]

            print()

            print(
                f"[{global_index}/"
                f"{total_questions}] "
                f"{question_id}: "
                f"{question}"
            )

            question_start = (
                time.perf_counter()
            )

            try:

                result = evaluate_question(
                    client=client,
                    record=record,
                )

                details.append(
                    result
                )

                question_latency_ms = (
                    time.perf_counter()
                    - question_start
                ) * 1000

                print(
                    f"    Total question: "
                    f"{question_latency_ms:.2f} ms"
                )

            except Exception as exc:

                question_latency_ms = (
                    time.perf_counter()
                    - question_start
                ) * 1000

                print(
                    f"    ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                error_result = (
                    build_error_result(
                        record=record,
                        error=exc,
                        latency_ms=(
                            question_latency_ms
                        ),
                    )
                )

                details.append(
                    error_result
                )

            # ------------------------------------------------
            # Small delay between individual requests
            # ------------------------------------------------

            if local_index < len(batch):

                time.sleep(1)

        # ----------------------------------------------------
        # Save immediately after each batch
        # ----------------------------------------------------

        save_details(
            details
        )

        print()

        print(
            f"Batch {batch_index + 1} "
            f"completed."
        )

        print(
            f"Saved {len(details)} "
            f"results."
        )

        # ----------------------------------------------------
        # Delay between batches
        # ----------------------------------------------------

        if batch_index + 1 < total_batches:

            print(
                f"Waiting "
                f"{BATCH_DELAY_SECONDS}s "
                f"before next batch..."
            )

            time.sleep(
                BATCH_DELAY_SECONDS
            )

    # ========================================================
    # Final Summary
    # ========================================================

    total_latency_ms = (
        time.perf_counter()
        - evaluation_start
    ) * 1000

    summary = build_summary(
        details=details,
        total_latency_ms=(
            total_latency_ms
        ),
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print Summary
    # ========================================================

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Questions: "
        f"{summary['question_count']}"
    )

    print(
        f"Successful: "
        f"{summary['successful_questions']}"
    )

    print(
        f"Failed: "
        f"{summary['failed_questions']}"
    )

    print()

    print("Retrieval")

    print(
        f"  Mode: "
        f"{summary['retrieval']['mode']}"
    )

    print(
        f"  Candidate limit: "
        f"{summary['retrieval']['candidate_limit']}"
    )

    print(
        f"  Top-K: "
        f"{summary['retrieval']['top_k']}"
    )

    print(
        f"  Max chunks/source: "
        f"{summary['retrieval']['max_chunks_per_source']}"
    )

    print(
        f"  Hit Rate: "
        f"{summary['retrieval']['retrieval_hit_rate']:.4f}"
    )

    print(
        f"  Avg Latency: "
        f"{summary['retrieval']['avg_latency_ms']:.2f} ms"
    )

    print()

    print("Generation")

    print(
        f"  Model: "
        f"{summary['generation']['model']}"
    )

    print(
        f"  Avg Latency: "
        f"{summary['generation']['avg_latency_ms']:.2f} ms"
    )

    print()

    print("End-to-End")

    print(
        f"  Avg Latency: "
        f"{summary['end_to_end']['avg_latency_ms']:.2f} ms"
    )

    print()

    print(
        f"Details: "
        f"{DETAILS_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()

