"""
MCP RAG Generation Evaluation

Evaluates generated answers using LLM-as-a-Judge.

Metrics:
    1. Answer Correctness
    2. Faithfulness
    3. Citation Correctness

Input:
    - mcp_rag_eval_dataset.jsonl
    - evaluation/results/rag_generation_details.json

Output:
    - evaluation/results/generation_evaluation.json

Execution:
    - Sequential requests
    - Small batches
    - Delay between batches
    - No automatic SDK retries
    - One failed evaluation does not stop the whole run
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


DATASET_PATH = (
    BASE_DIR
    / "mcp_rag_eval_dataset.jsonl"
)

GENERATION_RESULTS_PATH = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "e2_final_comparison_generation_details.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "e2_final_generation_evaluation.json"
)


MODEL = os.getenv("OPENAI_MODEL")

if not MODEL:
    raise RuntimeError(
        "OPENAI_MODEL is not set in the environment."
    )


# ============================================================
# Judge Configuration
# ============================================================

JUDGE_MODEL = os.getenv(
    "OPENAI_EVAL_MODEL",
    MODEL,
)

REQUEST_TIMEOUT_SECONDS = 90

MAX_OUTPUT_TOKENS = 1500

BATCH_SIZE = 3

BATCH_DELAY_SECONDS = 3


# ============================================================
# Load JSONL Dataset
# ============================================================

def load_dataset(
    path: Path,
) -> dict[str, dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    records = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            records[record["id"]] = record

    return records


# ============================================================
# Load Generation Results
# ============================================================

def load_generation_results(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Generation results not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# Build Context For Judge
# ============================================================

def build_judge_context(
    generation_result: dict[str, Any],
) -> str:

    retrieved_details = (
        generation_result
        .get("retrieval", {})
        .get("retrieved_details", [])
    )

    context_parts = []

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

        context_parts.append(
            f"""
[Retrieved Source {rank}]
Title: {title}
Section: {section}
URL: {url}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# Judge Prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for a Retrieval-Augmented Generation
(RAG) system that answers questions about the MCP Python SDK.

Your task is to evaluate a generated answer using ONLY the supplied
evaluation data and retrieved documentation context.

Evaluate three dimensions independently:

1. Answer Correctness
   Does the generated answer correctly answer the question?
   Compare it with the expected answer.
   Equivalent wording is acceptable.
   Do not require exact wording.

2. Faithfulness
   Are the claims in the generated answer supported by the retrieved
   documentation context?
   Penalize unsupported claims, hallucinations, or contradictions.
   If the answer is correct but the retrieved context does not support
   it, faithfulness should be reduced.

3. Citation Correctness
   Does the generated answer cite appropriate source URLs?
   A citation is correct when the cited source is among the relevant
   sources or the retrieved documentation clearly supports the claim.
   Missing citations should reduce the score when citations are expected.
   Do not reward a citation merely because the URL appears in the
   retrieved context.

Use the following scoring scale for EACH dimension:

0 = Incorrect / unsupported
1 = Partially correct / partially supported
2 = Fully correct / fully supported

Important:

* Judge the answer itself, not the quality of the retrieval system.
* Do not give credit merely because the answer sounds plausible.
* Do not use outside knowledge.
* Be conservative when evidence is insufficient.
* Evaluate the generated answer against the expected answer and
  retrieved context actually provided.
* Do not assume that information missing from the generated answer
  was implied.
* If the question asks for multiple items and the expected answer
  explicitly lists them, check whether the generated answer covers
  the important items.
* Do not invent claims about what the generated answer contains.
* Keep each reason concise: one sentence, preferably fewer than
  30 words.

OUTPUT FORMAT:
Return ONLY one valid JSON object.
Do not use Markdown code fences.
Do not include any text before or after the JSON.

The JSON object MUST contain exactly these fields:
{
"answer_correctness": {
"score": 0,
"reason": "..."
},
"faithfulness": {
"score": 0,
"reason": "..."
},
"citation_correctness": {
"score": 0,
"reason": "..."
},
"overall_score": 0.0,
"overall_label": "FAIL"
}

Allowed overall_label values:
"PASS", "PARTIAL", "FAIL".

Do not add any additional fields.
"""

def build_judge_prompt(
    question: str,
    expected_answer: str,
    relevant_sources: list[str],
    generated_answer: str,
    retrieved_context: str,
) -> str:
    relevant_sources_text = "\n".join(
        f"- {url}"
        for url in relevant_sources
    )

    return f"""
{JUDGE_SYSTEM_PROMPT}

================ QUESTION ================

{question}

================ EXPECTED ANSWER ================

{expected_answer}

================ RELEVANT SOURCES ================

{relevant_sources_text}

================ RETRIEVED DOCUMENTATION ================

{retrieved_context}

================ GENERATED ANSWER ================

{generated_answer}

================ TASK ================

Evaluate the generated answer independently on:

1. answer_correctness
2. faithfulness
3. citation_correctness

For EACH dimension:
- score must be exactly 0, 1, or 2
- reason must be ONE concise sentence, preferably fewer than 30 words
- Do not repeat the question, expected answer, context, or generated answer
- Do not add unnecessary explanation

Scoring:

0 = Incorrect / unsupported
1 = Partially correct / partially supported
2 = Fully correct / fully supported

Also calculate:

- overall_score = arithmetic mean of the three scores
- overall_label:
    "PASS" if all three dimensions are >= 1
    "PARTIAL" if the answer is partially successful
    "FAIL" if the answer is substantially incorrect or unsupported

IMPORTANT:
- Judge only the generated answer against the supplied expected answer
  and retrieved documentation.
- Do not use outside knowledge.
- Do not assume information that is not explicitly present.
- Do not give credit for information that is merely implied.
- If the expected answer contains multiple important items, check whether
  the generated answer covers them.
- Do not claim that the generated answer contains information that it does
  not actually contain.
- Missing citations should affect citation_correctness when citations
  are expected.
- Do not confuse retrieval quality with answer quality.

OUTPUT REQUIREMENTS:

Return ONLY one valid JSON object.

Do NOT use Markdown code fences.
Do NOT include any text before or after the JSON.
Do NOT add any fields other than the required fields below.

Required JSON format:

{{
  "answer_correctness": {{
    "score": 0,
    "reason": "..."
  }},
  "faithfulness": {{
    "score": 0,
    "reason": "..."
  }},
  "citation_correctness": {{
    "score": 0,
    "reason": "..."
  }},
  "overall_score": 0.0,
  "overall_label": "PASS"
}}
"""

# ============================================================
# JSON Extraction
# ============================================================

def extract_json(
    text: str,
) -> dict[str, Any]:

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Markdown code block
    # --------------------------------------------------------

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:

        candidate = match.group(1).strip()

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # First JSON object
    # --------------------------------------------------------

    start = text.find("{")

    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse judge JSON:\n{text}"
    )


# ============================================================
# Validate Judge Result
# ============================================================

def validate_judge_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    required_metrics = [
        "answer_correctness",
        "faithfulness",
        "citation_correctness",
    ]

    for metric in required_metrics:

        if metric not in result:
            raise ValueError(
                f"Missing metric: {metric}"
            )

        score = result[metric].get(
            "score"
        )

        if score not in [0, 1, 2]:

            raise ValueError(
                f"Invalid score for "
                f"{metric}: {score}"
            )

    scores = [
        result[
            metric
        ]["score"]
        for metric in required_metrics
    ]

    calculated_average = (
        sum(scores) / len(scores)
    )

    result["overall_score"] = round(
        calculated_average,
        2,
    )

    if all(
        score >= 1
        for score in scores
    ):

        if all(
            score == 2
            for score in scores
        ):
            result["overall_label"] = (
                "PASS"
            )
        else:
            result["overall_label"] = (
                "PARTIAL"
            )

    else:

        result["overall_label"] = (
            "FAIL"
        )

    return result


# ============================================================
# Judge One Answer
# ============================================================
def judge_answer(
    client: OpenAI,
    question: str,
    expected_answer: str,
    relevant_sources: list[str],
    generated_answer: str,
    retrieved_context: str,
) -> tuple[dict[str, Any], float, dict[str, Any]]:

    prompt = build_judge_prompt(
        question=question,
        expected_answer=expected_answer,
        relevant_sources=relevant_sources,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context,
    )

    print(
        f"    Judge prompt size: "
        f"{len(prompt):,} chars"
    )

    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):

        try:

            start = time.perf_counter()

            response = client.responses.create(
                model=JUDGE_MODEL,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            raw_output = (
                response.output_text.strip()
            )

            parsed = extract_json(
                raw_output
            )

            validated = validate_judge_result(
                parsed
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

            if attempt > 1:
                print(
                    f"    Judge succeeded on retry "
                    f"{attempt}/{max_attempts}"
                )

            return (
                validated,
                latency_ms,
                usage,
            )

        except Exception as exc:

            last_error = exc

            print(
                f"    Judge attempt "
                f"{attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_attempts:
                time.sleep(
                    2 ** (attempt - 1)
                )

    raise last_error

# ============================================================
# Build Error Result
# ============================================================

def build_error_result(
    question_id: str,
    error: Exception,
    latency_ms: float,
) -> dict[str, Any]:

    return {
        "id": question_id,

        "evaluation_error": {
            "type": type(error).__name__,
            "message": str(error),
            "latency_ms": round(
                latency_ms,
                2,
            ),
        },
    }


# ============================================================
# Evaluate One Question
# ============================================================

def evaluate_question(
    client: OpenAI,
    generation_result: dict[str, Any],
    dataset_record: dict[str, Any],
) -> dict[str, Any]:

    question_id = generation_result["id"]

    question = generation_result[
        "question"
    ]

    expected_answer = dataset_record.get(
        "expected_answer",
        "",
    )

    relevant_sources = dataset_record.get(
        "relevant_sources",
        [],
    )

    generated_answer = (
        generation_result
        .get("generation", {})
        .get("answer", "")
    )

    retrieved_context = (
        build_judge_context(
            generation_result
        )
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not generated_answer:

        raise ValueError(
            "Generated answer is empty."
        )

    if not expected_answer:

        raise ValueError(
            "Expected answer is empty."
        )

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------

    evaluation, latency_ms, usage = (
        judge_answer(
            client=client,
            question=question,
            expected_answer=expected_answer,
            relevant_sources=relevant_sources,
            generated_answer=generated_answer,
            retrieved_context=retrieved_context,
        )
    )

    # --------------------------------------------------------
    # Retrieval status
    # --------------------------------------------------------

    retrieval_hit = (
        generation_result
        .get("retrieval", {})
        .get("hit")
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "id": question_id,

        "question": question,

        "expected_answer": expected_answer,

        "relevant_sources": relevant_sources,

        "generated_answer": generated_answer,

        "retrieval": {
            "hit": retrieval_hit,

            "retrieved_urls": (
                generation_result
                .get("retrieval", {})
                .get("retrieved_urls", [])
            ),
        },

        "evaluation": evaluation,

        "judge": {
            "model": JUDGE_MODEL,

            "latency_ms": round(
                latency_ms,
                2,
            ),

            "usage": usage,
        },
    }


# ============================================================
# Save Results
# ============================================================

def save_results(
    results: list[dict[str, Any]],
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Build Summary
# ============================================================

def build_summary(
    results: list[dict[str, Any]],
    total_latency_ms: float,
) -> dict[str, Any]:

    successful = [
        item
        for item in results
        if "evaluation_error" not in item
    ]

    failed = [
        item
        for item in results
        if "evaluation_error" in item
    ]

    def average_score(
        metric: str,
    ) -> float:

        scores = [
            item["evaluation"][metric]["score"]
            for item in successful
        ]

        if not scores:
            return 0.0

        return round(
            sum(scores)
            / len(scores),
            4,
        )

    correctness_scores = [
        item["evaluation"][
            "answer_correctness"
        ]["score"]
        for item in successful
    ]

    faithfulness_scores = [
        item["evaluation"][
            "faithfulness"
        ]["score"]
        for item in successful
    ]

    citation_scores = [
        item["evaluation"][
            "citation_correctness"
        ]["score"]
        for item in successful
    ]

    overall_scores = [
        item["evaluation"][
            "overall_score"
        ]
        for item in successful
    ]

    pass_count = sum(
        1
        for item in successful
        if item["evaluation"][
            "overall_label"
        ] == "PASS"
    )

    partial_count = sum(
        1
        for item in successful
        if item["evaluation"][
            "overall_label"
        ] == "PARTIAL"
    )

    fail_count = sum(
        1
        for item in successful
        if item["evaluation"][
            "overall_label"
        ] == "FAIL"
    )

    retrieval_hit_results = [
        item
        for item in successful
        if item["retrieval"]["hit"]
    ]

    retrieval_miss_results = [
        item
        for item in successful
        if not item["retrieval"]["hit"]
    ]

    def avg_overall(
        items: list[dict[str, Any]],
    ) -> float:

        if not items:
            return 0.0

        values = [
            item["evaluation"][
                "overall_score"
            ]
            for item in items
        ]

        return round(
            sum(values) / len(values),
            4,
        )

    return {
        "dataset": str(
            DATASET_PATH
        ),

        "generation_results": str(
            GENERATION_RESULTS_PATH
        ),

        "question_count": len(results),

        "successful_questions": len(
            successful
        ),

        "failed_questions": len(
            failed
        ),

        "generation_model": MODEL,

        "judge_model": JUDGE_MODEL,

        "metrics": {
            "answer_correctness": {
                "avg_score": average_score(
                    "answer_correctness"
                ),

                "max_score": 2,

                "normalized_score": round(
                    average_score(
                        "answer_correctness"
                    ) / 2,
                    4,
                ),
            },

            "faithfulness": {
                "avg_score": average_score(
                    "faithfulness"
                ),

                "max_score": 2,

                "normalized_score": round(
                    average_score(
                        "faithfulness"
                    ) / 2,
                    4,
                ),
            },

            "citation_correctness": {
                "avg_score": average_score(
                    "citation_correctness"
                ),

                "max_score": 2,

                "normalized_score": round(
                    average_score(
                        "citation_correctness"
                    ) / 2,
                    4,
                ),
            },

            "overall": {
                "avg_score": round(
                    sum(overall_scores)
                    / len(overall_scores),
                    4,
                )
                if overall_scores
                else 0.0,

                "normalized_score": round(
                    (
                        sum(overall_scores)
                        / len(overall_scores)
                    ) / 2,
                    4,
                )
                if overall_scores
                else 0.0,
            },
        },

        "labels": {
            "pass": pass_count,

            "partial": partial_count,

            "fail": fail_count,
        },

        "retrieval_vs_generation": {
            "retrieval_hits": {
                "count": len(
                    retrieval_hit_results
                ),

                "avg_overall_score": (
                    avg_overall(
                        retrieval_hit_results
                    )
                ),
            },

            "retrieval_misses": {
                "count": len(
                    retrieval_miss_results
                ),

                "avg_overall_score": (
                    avg_overall(
                        retrieval_miss_results
                    )
                ),
            },
        },

        "evaluation": {
            "avg_judge_latency_ms": round(
                sum(
                    item["judge"][
                        "latency_ms"
                    ]
                    for item in successful
                )
                / len(successful),
                2,
            )
            if successful
            else 0.0,

            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),

            "avg_latency_ms": round(
                total_latency_ms
                / len(results),
                2,
            )
            if results
            else 0.0,
        },

        "errors": [
            {
                "id": item["id"],
                "type": item[
                    "evaluation_error"
                ]["type"],
                "message": item[
                    "evaluation_error"
                ]["message"],
            }
            for item in failed
        ],
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("MCP RAG GENERATION EVALUATION")
    print("=" * 80)

    print()

    print(
        f"Dataset: "
        f"{DATASET_PATH}"
    )

    print(
        f"Generation results: "
        f"{GENERATION_RESULTS_PATH}"
    )

    print(
        f"Generation model: "
        f"{MODEL}"
    )

    print(
        f"Judge model: "
        f"{JUDGE_MODEL}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Batch delay: "
        f"{BATCH_DELAY_SECONDS}s"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataset = load_dataset(
        DATASET_PATH
    )

    generation_results = (
        load_generation_results(
            GENERATION_RESULTS_PATH
        )
    )

    print(
        f"Loaded {len(dataset)} "
        f"dataset questions."
    )

    print(
        f"Loaded {len(generation_results)} "
        f"generation results."
    )

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

    results = []

    evaluation_start = (
        time.perf_counter()
    )

    total_questions = len(
        generation_results
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

        batch = generation_results[
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

        for local_index, generation_result in enumerate(
            batch,
            start=1,
        ):

            global_index = (
                batch_start
                + local_index
            )

            question_id = (
                generation_result["id"]
            )

            question = (
                generation_result["question"]
            )

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

                dataset_record = dataset.get(
                    question_id
                )

                if dataset_record is None:
                    raise KeyError(
                        f"Question {question_id} "
                        f"not found in dataset."
                    )

                result = evaluate_question(
                    client=client,
                    generation_result=(
                        generation_result
                    ),
                    dataset_record=(
                        dataset_record
                    ),
                )

                results.append(result)

                evaluation = result[
                    "evaluation"
                ]

                print(
                    f"    Correctness: "
                    f"{evaluation['answer_correctness']['score']}/2"
                )

                print(
                    f"    Faithfulness: "
                    f"{evaluation['faithfulness']['score']}/2"
                )

                print(
                    f"    Citation: "
                    f"{evaluation['citation_correctness']['score']}/2"
                )

                print(
                    f"    Overall: "
                    f"{evaluation['overall_score']}/2 "
                    f"({evaluation['overall_label']})"
                )

                print(
                    f"    Judge latency: "
                    f"{result['judge']['latency_ms']:.2f} ms"
                )

            except Exception as exc:

                latency_ms = (
                    time.perf_counter()
                    - question_start
                ) * 1000

                print(
                    f"    ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                results.append(
                    build_error_result(
                        question_id=question_id,
                        error=exc,
                        latency_ms=latency_ms,
                    )
                )

            # ------------------------------------------------
            # Small delay between questions
            # ------------------------------------------------

            if local_index < len(batch):

                time.sleep(1)

        # ----------------------------------------------------
        # Save after every batch
        # ----------------------------------------------------

        save_results(results)

        print()

        print(
            f"Batch {batch_index + 1} "
            f"completed."
        )

        print(
            f"Saved {len(results)} "
            f"evaluation results."
        )

        # ----------------------------------------------------
        # Delay between batches
        # ----------------------------------------------------

        if batch_index + 1 < total_batches:

            print(
                f"Waiting "
                f"{BATCH_DELAY_SECONDS}s..."
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
        results=results,
        total_latency_ms=(
            total_latency_ms
        ),
    )

    # --------------------------------------------------------
    # Save final result
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # Add summary into output
    # --------------------------------------------------------

    final_output = {
        "summary": summary,

        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print Summary
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATION EVALUATION SUMMARY")
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

    print("Answer Quality")

    print(
        f"  Correctness: "
        f"{summary['metrics']['answer_correctness']['avg_score']:.4f}/2 "
        f"("
        f"{summary['metrics']['answer_correctness']['normalized_score']:.2%}"
        f")"
    )

    print(
        f"  Faithfulness: "
        f"{summary['metrics']['faithfulness']['avg_score']:.4f}/2 "
        f"("
        f"{summary['metrics']['faithfulness']['normalized_score']:.2%}"
        f")"
    )

    print(
        f"  Citation: "
        f"{summary['metrics']['citation_correctness']['avg_score']:.4f}/2 "
        f"("
        f"{summary['metrics']['citation_correctness']['normalized_score']:.2%}"
        f")"
    )

    print(
        f"  Overall: "
        f"{summary['metrics']['overall']['avg_score']:.4f}/2 "
        f"("
        f"{summary['metrics']['overall']['normalized_score']:.2%}"
        f")"
    )

    print()

    print("Labels")

    print(
        f"  PASS: "
        f"{summary['labels']['pass']}"
    )

    print(
        f"  PARTIAL: "
        f"{summary['labels']['partial']}"
    )

    print(
        f"  FAIL: "
        f"{summary['labels']['fail']}"
    )

    print()

    print(
        "Retrieval Hit vs Miss"
    )

    print(
        f"  Retrieval Hit "
        f"({summary['retrieval_vs_generation']['retrieval_hits']['count']}): "
        f"{summary['retrieval_vs_generation']['retrieval_hits']['avg_overall_score']:.4f}/2"
    )

    print(
        f"  Retrieval Miss "
        f"({summary['retrieval_vs_generation']['retrieval_misses']['count']}): "
        f"{summary['retrieval_vs_generation']['retrieval_misses']['avg_overall_score']:.4f}/2"
    )

    print()

    print(
        f"Avg Judge Latency: "
        f"{summary['evaluation']['avg_judge_latency_ms']:.2f} ms"
    )

    print()

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()