import json
from pathlib import Path

from embedding.model import BGEEmbeddingModel
from db.repository import insert_documents, insert_chunks


DOCUMENTS_FILE = Path(
    "data/documents.jsonl"
)

CHUNKS_FILE = Path(
    "data/chunks.jsonl"
)

BATCH_SIZE = 32


def load_jsonl(
    path: Path,
) -> list[dict]:

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


def main():

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print(
        f"Loading documents from "
        f"{DOCUMENTS_FILE}"
    )

    documents = load_jsonl(
        DOCUMENTS_FILE
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    if not documents:
        raise RuntimeError(
            "No documents found"
        )

    # --------------------------------------------------
    # 2. Insert documents
    # --------------------------------------------------

    insert_documents(
        documents
    )

    print(
        f"Inserted {len(documents)} documents"
    )

    # --------------------------------------------------
    # 3. Load chunks
    # --------------------------------------------------

    print(
        f"Loading chunks from "
        f"{CHUNKS_FILE}"
    )

    chunks = load_jsonl(
        CHUNKS_FILE
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )

    if not chunks:
        raise RuntimeError(
            "No chunks found"
        )

    # --------------------------------------------------
    # 4. Create embedding model
    # --------------------------------------------------

    model = BGEEmbeddingModel()

    total = len(chunks)

    # --------------------------------------------------
    # 5. Batch embedding
    # --------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = chunks[start:end]

        texts = [
            chunk["content"]
            for chunk in batch
        ]

        embeddings = (
            model.embed_documents(
                texts
            )
        )

        rows = []

        for chunk, embedding in zip(
            batch,
            embeddings,
        ):

            rows.append(
                {
                    **chunk,
                    "embedding": embedding,
                }
            )

        insert_chunks(rows)

        print(
            f"Processed {end}/{total}"
        )

    print()
    print(
        f"Finished embedding "
        f"{total} chunks"
    )


if __name__ == "__main__":
    main()