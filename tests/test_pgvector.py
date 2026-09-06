from db.connection import get_connection
from pgvector.psycopg import register_vector


def test_pgvector():

    with get_connection() as conn:

        register_vector(conn)

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )

            result = cur.fetchone()

            assert result is not None

            print(
                "\npgvector:",
                result[0],
            )