import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from retrieval.search import _model

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/mcp_rag",
)

query = "Why would I use an MCP dependency instead of calling a helper directly?"

query_embedding = _model.embed_query(query)

conn = psycopg.connect(DATABASE_URL)
register_vector(conn)

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            id,
            title,
            1 - (embedding <=> %s) AS similarity
        FROM chunks
        WHERE url = 'https://py.sdk.modelcontextprotocol.io/handlers/dependencies/'
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        """,
        (
            query_embedding,
            query_embedding,
        ),
    )

    dependency_chunks = cur.fetchall()

    print("=" * 80)
    print("Q043 - dependencies page chunks")
    print("=" * 80)

    for row in dependency_chunks:
        print(
            f"{row[0]} | "
            f"{float(row[2]):.6f} | "
            f"{row[1]}"
        )

    # 找 dependencies 页面中 similarity 最高的 chunk
    best_chunk = dependency_chunks[0]
    best_id = best_chunk[0]
    best_similarity = best_chunk[2]

    # 计算这个 chunk 的全库 rank
    cur.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        WHERE embedding IS NOT NULL
          AND (embedding <=> %s) < (
              SELECT embedding <=> %s
              FROM chunks
              WHERE id = %s
          )
        """,
        (
            query_embedding,
            query_embedding,
            best_id,
        ),
    )

    better_count = cur.fetchone()[0]

rank = better_count + 1

print()
print("=" * 80)
print("BEST CHUNK FROM CORRECT SOURCE")
print("=" * 80)
print(f"Chunk       : {best_id}")
print(f"Similarity  : {float(best_similarity):.6f}")
print(f"Global rank : {rank}")

conn.close()