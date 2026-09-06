import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/mcp_rag",
)

conn = psycopg.connect(DATABASE_URL)

with conn.cursor() as cur:
    cur.execute(
    """
    SELECT
        id,
        title,
        section,
        chunk_index,
        content
    FROM chunks
    WHERE url = 'https://py.sdk.modelcontextprotocol.io/handlers/dependencies/'
    ORDER BY chunk_index
    """
)

    rows = cur.fetchall()

conn.close()

print(f"Found {len(rows)} chunks\n")

for chunk_id, title, section, chunk_index, content in rows:
    print("=" * 80)
    print(f"CHUNK {chunk_index}")
    print(f"ID: {chunk_id}")
    print(f"TITLE: {title}")
    print(f"SECTION: {section}")
    print("-" * 80)
    print(content[:1000])
    print()