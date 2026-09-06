
from typing import Iterable

from pgvector.psycopg import register_vector

from db.connection import get_connection


def insert_documents(
    documents: Iterable[dict],
):
    with get_connection() as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            for document in documents:

                cur.execute(
                    """
                    INSERT INTO documents
                        (
                            id,
                            title,
                            url,
                            source,
                            path
                        )
                    VALUES
                        (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                    ON CONFLICT (id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        source = EXCLUDED.source,
                        path = EXCLUDED.path
                    """,
                    (
                        document["id"],
                        document["title"],
                        document["url"],
                        document["source"],
                        document["path"],
                    ),
                )


def insert_chunks(
    chunks: Iterable[dict],
):
    with get_connection() as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            for chunk in chunks:

                cur.execute(
                    """
                    INSERT INTO chunks
                        (
                            id,
                            document_id,
                            title,
                            section,
                            content,
                            url,
                            source,
                            chunk_index,
                            embedding
                        )
                    VALUES
                        (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                    ON CONFLICT (id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["title"],
                        chunk["section"],
                        chunk["content"],
                        chunk["url"],
                        chunk["source"],
                        chunk["chunk_index"],
                        chunk["embedding"],
                    ),
                )


# ============================================================
# Vector Search
# ============================================================

def search_chunks(
    query_embedding,
    limit: int = 5,
):
    """
    Search chunks using vector similarity.
    """

    with get_connection() as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    id,
                    title,
                    section,
                    content,
                    url,
                    1 - (
                        embedding <=> %s
                    ) AS similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    limit,
                ),
            )

            rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "section": row[2],
                    "content": row[3],
                    "url": row[4],
                    "similarity": float(row[5]),
                }
                for row in rows
            ]


# ============================================================
# Keyword / Full-Text Search
# ============================================================

def search_chunks_keyword(
    query: str,
    limit: int = 5,
):
    """
    Search chunks using PostgreSQL Full-Text Search.

    This searches across:
        title
        section
        content

    ts_rank() is used as the lexical relevance score.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    id,
                    title,
                    section,
                    content,
                    url,

                    ts_rank(
                        to_tsvector(
                            'english',
                            coalesce(title, '')
                            || ' '
                            || coalesce(section, '')
                            || ' '
                            || coalesce(content, '')
                        ),
                        plainto_tsquery(
                            'english',
                            %s
                        )
                    ) AS similarity

                FROM chunks

                WHERE
                    to_tsvector(
                        'english',
                        coalesce(title, '')
                        || ' '
                        || coalesce(section, '')
                        || ' '
                        || coalesce(content, '')
                    )
                    @@
                    plainto_tsquery(
                        'english',
                        %s
                    )

                ORDER BY similarity DESC

                LIMIT %s
                """,
                (
                    query,
                    query,
                    limit,
                ),
            )

            rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "section": row[2],
                    "content": row[3],
                    "url": row[4],
                    "similarity": float(row[5]),
                }
                for row in rows
            ]

