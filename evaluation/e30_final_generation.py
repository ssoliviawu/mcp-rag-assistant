"""
E30-B - FINAL END-TO-END RAG GENERATION BENCHMARK

Frozen Retrieval Pipeline:
    E25 C7
        ↓
    BGEEmbeddingModel
        ↓
    Vector Similarity
        ↓
    Candidate Limit = 100
        ↓
    Max 1 Chunk / Source
        ↓
    Top-K = 10
        ↓
    Context
        ↓
    LLM Generation
        ↓
    Save Results

Important:
    This is the FINAL retrieval pipeline.

    No reranker
    No hybrid retrieval
    No lexical boost
    No query expansion
    No new heuristic

Dataset:
    Exclude domain == "spec"
    Expected evaluation size = 48 questions
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from embedding.model import BGEEmbeddingModel


# ============================================================
# Configuration
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Paths
# ============================================================

DATASET_PATH = (
    BASE_DIR / "mcp_rag_eval_dataset.jsonl"
)

CHUNKS_PATH = (
    BASE_DIR / "data" / "chunks.jsonl"
)

RESULTS_DIR = (
    BASE_DIR / "evaluation" / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    RESULTS_DIR / "e30_final_generation_summary.json"
)

DETAILS_PATH = (
    RESULTS_DIR / "e30_final_generation_details.json"
)


# ============================================================
# FINAL RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 10

CANDIDATE_LIMIT = 100

MAX_CHUNKS_PER_SOURCE = 1

EMBED_BATCH_SIZE = 32


# ============================================================
# EXACT E25 C7
# ============================================================

FINAL_COMBINATION = "E25_C7_all_three"

FINAL_GENERIC_RULES = {
    "recap",
    "installation",
    "requirements",
}


# ============================================================
# Context
# ============================================================

MAX_CONTEXT_CHARS = 12_000


# ============================================================
# Model
# ============================================================

MODEL = os.getenv("OPENAI_MODEL")

if not MODEL:
    raise RuntimeError(
        "OPENAI_MODEL is not set in the environment."
    )


# ============================================================
# Generation
# ============================================================

MAX_OUTPUT_TOKENS = 1000

REQUEST_TIMEOUT_SECONDS = 90


# ============================================================
# Batch
# ============================================================

BATCH_SIZE = 3

BATCH_DELAY_SECONDS = 4

QUESTION_DELAY_SECONDS = 1


# ============================================================
# Dataset
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
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


def load_evaluation_dataset() -> list[dict[str, Any]]:

    dataset = load_jsonl(
        DATASET_PATH
    )

    original_count = len(dataset)

    dataset = [
        item
        for item in dataset
        if item.get("domain") != "spec"
    ]

    print(
        f"Original questions : "
        f"{original_count}"
    )

    print(
        f"Evaluated questions: "
        f"{len(dataset)}"
    )

    print(
        f"Excluded spec      : "
        f"{original_count - len(dataset)}"
    )

    return dataset


# ============================================================
# URL normalization
# ============================================================

def normalize_url(
    url: str | None,
) -> str:

    if not url:
        return ""

    return url.rstrip("/") + "/"


# ============================================================
# EXACT E25 C7 SECTION QUALITY
# ============================================================

def clean_section(
    section: str | None,
) -> str:

    if not section:
        return ""

    section = section.strip()

    section = re.sub(
        r"\s+",
        " ",
        section,
    )

    return section


def section_quality_with_rules(
    section: str | None,
    generic_rules: set[str],
) -> bool:

    section = clean_section(
        section
    )

    if not section:
        return False

    lower = section.lower().strip()

    # --------------------------------------------------------
    # 1. Exact generic sections
    # --------------------------------------------------------

    if lower in generic_rules:
        return False

    # --------------------------------------------------------
    # 2. Very short sections
    # --------------------------------------------------------

    words = re.findall(
        r"[A-Za-z0-9_]+",
        lower,
    )

    if len(words) <= 1:
        return False

    # --------------------------------------------------------
    # 3. Generic final breadcrumb component
    # --------------------------------------------------------

    parts = [
        p.strip()
        for p in re.split(
            r"\s*>\s*",
            section,
        )
        if p.strip()
    ]

    if parts:

        last = parts[-1].lower()

        if last in generic_rules:
            return False

        for generic in generic_rules:

            if last.startswith(
                generic + " "
            ):
                return False

    # --------------------------------------------------------
    # 4. Extremely long navigation paths
    # --------------------------------------------------------

    if len(section) > 300:
        return False

    return True


# ============================================================
# EXACT E25 C7 REPRESENTATION
# ============================================================

def build_representation(
    chunk: dict[str, Any],
    generic_rules: set[str],
) -> str:

    content = (
        chunk.get("content") or ""
    ).strip()

    title = (
        chunk.get("title") or ""
    ).strip()

    section = (
        chunk.get("section") or ""
    ).strip()

    if section_quality_with_rules(
        section,
        generic_rules,
    ):

        return (
            f"{title}\n\n"
            f"{section}\n\n"
            f"{content}"
        )

    return (
        f"{title}\n\n"
        f"{content}"
    )


# ============================================================
# FINAL RETRIEVAL ENGINE
# ============================================================

class FinalC7Retriever:
    """
    Exact E25 C7 retrieval implementation.

    This intentionally does NOT call retrieval.search.

    The purpose is to guarantee that E30-B uses exactly
    the same frozen retrieval pipeline as E30-A.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
    ):

        self.chunks = chunks

        print()
        print(
            "-" * 80
        )

        print(
            "Building E25 C7 representations..."
        )

        self.texts = [
            build_representation(
                chunk,
                FINAL_GENERIC_RULES,
            )
            for chunk in chunks
        ]

        print(
            f"Representation count: "
            f"{len(self.texts)}"
        )

        # ----------------------------------------------------
        # Embed all chunks
        # ----------------------------------------------------

        print()
        print(
            "Embedding E25 C7 representations..."
        )

        self.model = (
            BGEEmbeddingModel()
        )

        embedding_start = (
            time.perf_counter()
        )

        embeddings = []

        for start in range(
            0,
            len(self.texts),
            EMBED_BATCH_SIZE,
        ):

            batch = self.texts[
                start:
                start + EMBED_BATCH_SIZE
            ]

            batch_embeddings = (
                self.model.embed_documents(
                    batch
                )
            )

            embeddings.append(
                batch_embeddings
            )

            end = min(
                start + EMBED_BATCH_SIZE,
                len(self.texts),
            )

            print(
                f"  Embedded "
                f"{end:>5}/"
                f"{len(self.texts)}"
            )

        self.doc_embeddings = np.vstack(
            embeddings
        )

        embedding_elapsed = (
            time.perf_counter()
            - embedding_start
        )

        print()
        print(
            f"Embedding shape: "
            f"{self.doc_embeddings.shape}"
        )

        print(
            f"Embedding time: "
            f"{embedding_elapsed:.2f}s"
        )

        self.embedding_elapsed = (
            embedding_elapsed
        )

    # ========================================================
    # Retrieve
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> tuple[
        list[dict[str, Any]],
        float,
    ]:

        start = time.perf_counter()

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.model.embed_query(
                question
            )
        )

        # ----------------------------------------------------
        # Vector similarity
        #
        # EXACTLY E25
        # ----------------------------------------------------

        similarities = np.dot(
            self.doc_embeddings,
            query_embedding,
        )

        # ----------------------------------------------------
        # Top 100 candidates
        # ----------------------------------------------------

        top_indices = np.argsort(
            -similarities
        )[:CANDIDATE_LIMIT]

        candidates = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            chunk = self.chunks[
                index
            ]

            candidates.append(
                {
                    "rank": rank,
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "section": chunk.get("section"),
                    "url": normalize_url(
                        chunk.get(
                            "url",
                            "",
                        )
                    ),
                    "content": (
                        chunk.get(
                            "content",
                            "",
                        )
                    ),
                    "similarity": float(
                        similarities[index]
                    ),
                }
            )

        # ----------------------------------------------------
        # Source diversity
        #
        # EXACTLY E25
        # ----------------------------------------------------

        final_results = []

        source_counts = {}

        for result in candidates:

            source = normalize_url(
                result.get(
                    "url",
                    "",
                )
            )

            if (
                source_counts.get(
                    source,
                    0,
                )
                >= MAX_CHUNKS_PER_SOURCE
            ):
                continue

            final_results.append(
                result
            )

            source_counts[source] = (
                source_counts.get(
                    source,
                    0,
                )
                + 1
            )

            if len(final_results) >= TOP_K:
                break

        elapsed = (
            time.perf_counter()
            - start
        )

        return (
            final_results,
            elapsed * 1000,
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
# Generation
# ============================================================

def generate_answer(
    client: OpenAI,
    question: str,
    context: str,
) -> tuple[
    str,
    dict[str, Any],
]:

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

    answer = (
        response.output_text.strip()
    )

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
# Retrieval Hit
# ============================================================

def calculate_retrieval_hit(
    retrieved_details: list[dict[str, Any]],
    relevant_sources: list[str],
) -> tuple[
    bool,
    list[str],
]:

    relevant_source_set = {
        normalize_url(source)
        for source in relevant_sources
        if source
    }

    retrieved_urls = []

    for item in retrieved_details:

        url = normalize_url(
            item.get("url")
        )

        if url:
            retrieved_urls.append(
                url
            )

    retrieved_source_set = set(
        retrieved_urls
    )

    retrieval_hit = bool(
        relevant_source_set
        & retrieved_source_set
    )

    return (
        retrieval_hit,
        retrieved_urls,
    )


# ============================================================
# Single Question
# ============================================================

def evaluate_question(
    client: OpenAI,
    retriever: FinalC7Retriever,
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

    retrieved_details, retrieval_latency_ms = (
        retriever.retrieve(
            question
        )
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_start = (
        time.perf_counter()
    )

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

    generation_start = (
        time.perf_counter()
    )

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
    # Retrieval hit
    # --------------------------------------------------------

    processing_start = (
        time.perf_counter()
    )

    retrieval_hit, retrieved_urls = (
        calculate_retrieval_hit(
            retrieved_details,
            relevant_sources,
        )
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

        "relevant_sources": (
            relevant_sources
        ),

        "retrieval": {
            "mode": FINAL_COMBINATION,

            "candidate_limit": (
                CANDIDATE_LIMIT
            ),

            "top_k": TOP_K,

            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),

            "representation": (
                "title + conditional section "
                "+ content"
            ),

            "section_rules": (
                "E25 C7"
            ),

            "reranker": False,

            "hybrid": False,

            "lexical_boost": False,

            "query_expansion": False,

            "latency_ms": round(
                retrieval_latency_ms,
                2,
            ),

            "hit": retrieval_hit,

            "retrieved_urls": (
                retrieved_urls
            ),

            "retrieved_details": (
                retrieved_details
            ),
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
    embedding_time_seconds: float,
) -> dict[str, Any]:

    question_count = len(details)

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
        "experiment": (
            "E30_B_final_end_to_end"
        ),

        "status": "FINAL",

        "dataset": {
            "file": DATASET_PATH.name,

            "question_count": (
                question_count
            ),

            "excluded_domain": "spec",

            "expected_question_count": 48,
        },

        "retrieval": {
            "mode": FINAL_COMBINATION,

            "candidate_limit": (
                CANDIDATE_LIMIT
            ),

            "top_k": TOP_K,

            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),

            "representation": (
                "E25 C7: title + conditional "
                "section + content"
            ),

            "section_rules": (
                sorted(FINAL_GENERIC_RULES)
            ),

            "reranker": False,

            "hybrid": False,

            "lexical_boost": False,

            "query_expansion": False,

            "retrieval_hit_rate": round(
                retrieval_hit_rate,
                4,
            ),

            "retrieval_hits": (
                retrieval_hits
            ),

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

            "max_output_tokens": (
                MAX_OUTPUT_TOKENS
            ),
        },

        "successful_questions": (
            len(successful)
        ),

        "failed_questions": (
            len(errors)
        ),

        "errors": [
            {
                "id": item["id"],

                "type": item["error"]["type"],

                "message": item["error"]["message"],
            }
            for item in errors
        ],

        "timing": {
            "embedding_all_chunks_seconds": (
                embedding_time_seconds
            ),

            "end_to_end_total_ms": round(
                total_latency_ms,
                2,
            ),

            "end_to_end_avg_ms": round(
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

            "question_delay_seconds": (
                QUESTION_DELAY_SECONDS
            ),
        },

        "pipeline_status": {
            "retrieval": "FROZEN_E25_C7",

            "reranker": "OFF",

            "hybrid": "OFF",

            "lexical_boost": "OFF",

            "query_expansion": "OFF",

            "experiment_purpose": (
                "Final end-to-end benchmark "
                "of the frozen retrieval pipeline."
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "E30-B - FINAL END-TO-END RAG BENCHMARK"
    )
    print("=" * 100)

    print()

    print(
        "FINAL RETRIEVAL PIPELINE"
    )

    print(
        f"  Combination          : "
        f"{FINAL_COMBINATION}"
    )

    print(
        f"  Candidate limit      : "
        f"{CANDIDATE_LIMIT}"
    )

    print(
        f"  Top K                : "
        f"{TOP_K}"
    )

    print(
        f"  Max chunks/source    : "
        f"{MAX_CHUNKS_PER_SOURCE}"
    )

    print(
        "  Section rules        : E25 C7"
    )

    print(
        "  Reranker             : OFF"
    )

    print(
        "  Hybrid               : OFF"
    )

    print(
        "  Lexical boost        : OFF"
    )

    print(
        "  Query expansion      : OFF"
    )

    print()

    print(
        "GENERATION"
    )

    print(
        f"  Model                : "
        f"{MODEL}"
    )

    print(
        f"  Max output tokens    : "
        f"{MAX_OUTPUT_TOKENS}"
    )

    print(
        f"  Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Batch delay          : "
        f"{BATCH_DELAY_SECONDS}s"
    )

    print()

    # ========================================================
    # Load dataset
    # ========================================================

    dataset = load_evaluation_dataset()

    print()

    # ========================================================
    # Load chunks
    # ========================================================

    chunks = load_jsonl(
        CHUNKS_PATH
    )

    print(
        f"Loaded chunks: "
        f"{len(chunks)}"
    )

    # ========================================================
    # Build FINAL C7 retriever
    # ========================================================

    retriever = FinalC7Retriever(
        chunks
    )

    # ========================================================
    # OpenAI client
    # ========================================================

    client = OpenAI(
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    details = []

    evaluation_start = (
        time.perf_counter()
    )

    total_questions = len(dataset)

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
        print("=" * 100)

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

        print("=" * 100)

        # ----------------------------------------------------
        # Sequential questions
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
                    retriever=retriever,
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
            # Delay between questions
            # ------------------------------------------------

            if local_index < len(batch):

                time.sleep(
                    QUESTION_DELAY_SECONDS
                )

        # ----------------------------------------------------
        # Save after every batch
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

        if (
            batch_index + 1
            < total_batches
        ):

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
        embedding_time_seconds=(
            retriever.embedding_elapsed
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
    print("=" * 100)
    print(
        "E30-B FINAL END-TO-END RESULT"
    )
    print("=" * 100)

    print()

    print(
        f"Questions: "
        f"{summary['dataset']['question_count']}"
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

    print(
        "Retrieval"
    )

    print(
        f"  Pipeline: "
        f"{summary['retrieval']['mode']}"
    )

    print(
        f"  Hit rate: "
        f"{summary['retrieval']['retrieval_hit_rate']:.4f}"
    )

    print(
        f"  Hits: "
        f"{summary['retrieval']['retrieval_hits']}"
    )

    print(
        f"  Misses: "
        f"{summary['retrieval']['retrieval_misses']}"
    )

    print(
        f"  Avg latency: "
        f"{summary['retrieval']['avg_latency_ms']:.2f} ms"
    )

    print()

    print(
        "Generation"
    )

    print(
        f"  Model: "
        f"{summary['generation']['model']}"
    )

    print(
        f"  Avg latency: "
        f"{summary['generation']['avg_latency_ms']:.2f} ms"
    )

    print()

    print(
        "End-to-End"
    )

    print(
        f"  Avg latency: "
        f"{summary['timing']['end_to_end_avg_ms']:.2f} ms"
    )

    print()

    print(
        "FILES"
    )

    print(
        f"  Details: "
        f"{DETAILS_PATH}"
    )

    print(
        f"  Summary: "
        f"{SUMMARY_PATH}"
    )

    print()

    print("=" * 100)
    print(
        "E30-B COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()