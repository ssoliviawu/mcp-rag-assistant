from db.connection import get_connection


def test_database_connection():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                "SELECT 1"
            )

            result = cur.fetchone()

            assert result == (1,)