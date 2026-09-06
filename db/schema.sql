CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    path TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    title TEXT NOT NULL,

    section TEXT,

    content TEXT NOT NULL,

    url TEXT NOT NULL,

    source TEXT NOT NULL,

    chunk_index INTEGER NOT NULL,

    embedding vector(384)
);
