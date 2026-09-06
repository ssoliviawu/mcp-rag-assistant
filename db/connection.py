import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/mcp_rag",
)


def get_connection():
    return psycopg.connect(
        DATABASE_URL
    )