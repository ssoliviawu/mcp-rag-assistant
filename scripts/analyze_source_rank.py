import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from retrieval.search import _model


# ============================================================
# Config
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/mcp_rag",
)

DATASET_PATH = Path(
    "mcp_rag_eval_dataset.jsonl"
)


# ============================================================
# Helpers
# ============================================================

def normalize_source(url: str) -> str:
    return url.rstrip("/")


def load_questions():
    questions = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(
                json.loads(line)
            )

    return questions


def get_best_source_chunk(
    conn,
    query_embedding,
    source,
):
    """
    Find the most similar chunk belonging to
    the relevant source.
    """

    source = normalize_source(source)

    with conn.cursor() as cur:

        # ----------------------------------------------------
        # Find the best chunk inside the correct source
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT
                id,
                title,
                section,
                url,
                1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE embedding IS NOT NULL
              AND rtrim(url, '/') = %s
            ORDER BY embedding <=> %s
            LIMIT 1
            """,
            (
                query_embedding,
                source,
                query_embedding,
            ),
        )

        row = cur.fetchone()

        if row is None:
            return None

        chunk_id = row[0]
        title = row[1]
        section = row[2]
        url = row[3]
        similarity = float(row[4])

        # ----------------------------------------------------
        # Calculate global rank
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE embedding IS NOT NULL
              AND (
                    embedding <=> %s
                  ) < (
                    SELECT embedding <=> %s
                    FROM chunks
                    WHERE id = %s
                  )
            """,
            (
                query_embedding,
                query_embedding,
                chunk_id,
            ),
        )

        better_count = cur.fetchone()[0]

        global_rank = better_count + 1

        return {
            "chunk_id": chunk_id,
            "title": title,
            "section": section,
            "url": url,
            "similarity": similarity,
            "global_rank": global_rank,
        }


# ============================================================
# Main
# ============================================================

def main():

    questions = load_questions()

    print("=" * 100)
    print("Vector Source Rank Diagnostic")
    print("=" * 100)
    print(f"Questions: {len(questions)}")
    print()

    conn = psycopg.connect(
        DATABASE_URL
    )

    register_vector(conn)

    records = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        relevant_sources = [
            normalize_source(source)
            for source in item["relevant_sources"]
        ]

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = _model.embed_query(
            question
        )

        # ----------------------------------------------------
        # Multiple relevant sources:
        # keep the best one
        # ----------------------------------------------------

        source_results = []

        for source in relevant_sources:

            result = get_best_source_chunk(
                conn,
                query_embedding,
                source,
            )

            if result is not None:

                source_results.append(
                    result
                )

        # ----------------------------------------------------
        # No source found
        # ----------------------------------------------------

        if not source_results:

            print(
                f"{index:03d} | "
                f"NO MATCH | "
                f"{question}"
            )

            continue

        # ----------------------------------------------------
        # Best relevant source chunk
        # ----------------------------------------------------

        best = min(
            source_results,
            key=lambda x: x["global_rank"],
        )

        record = {
            "question_index": index,
            "question": question,
            "best_source_rank": best["global_rank"],
            "best_source_similarity": best["similarity"],
            "best_chunk_id": best["chunk_id"],
            "best_title": best["title"],
            "best_section": best["section"],
            "best_url": best["url"],
        }

        records.append(record)

        print(
            f"{index:03d} | "
            f"rank={best['global_rank']:4d} | "
            f"sim={best['similarity']:.6f} | "
            f"{best['chunk_id']} | "
            f"{question}"
        )

    conn.close()

    # ========================================================
    # Statistics
    # ========================================================

    if not records:
        print()
        print("No results.")
        return

    ranks = [
        r["best_source_rank"]
        for r in records
    ]

    total = len(ranks)

    print()
    print("=" * 100)
    print("Best Relevant Source Rank")
    print("=" * 100)

    for k in [
        1,
        3,
        5,
        10,
        20,
        50,
        100,
        200,
    ]:

        count = sum(
            rank <= k
            for rank in ranks
        )

        percentage = (
            count / total
            if total
            else 0
        )

        print(
            f"Recall@{k:<3} "
            f"{count:>2}/{total:<2} "
            f"({percentage:.4f})"
        )

    print()
    print(
        f"Average rank: "
        f"{sum(ranks) / len(ranks):.2f}"
    )

    print(
        f"Median rank: "
        f"{sorted(ranks)[len(ranks) // 2]}"
    )

    # ========================================================
    # Save JSON
    # ========================================================

    output_path = Path(
        "evaluation/source_rank_diagnostic.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()