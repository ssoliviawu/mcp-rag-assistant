import json
from pathlib import Path

from bs4 import BeautifulSoup

from ingestion.chunk import chunk_markdown
from ingestion.clean import clean_markdown
from ingestion.models import Chunk, Document


RAW_DIR = Path("data/raw/mcp-python-sdk")

DOCUMENTS_OUTPUT_FILE = Path(
    "data/documents.jsonl"
)

CHUNKS_OUTPUT_FILE = Path(
    "data/chunks.jsonl"
)

BASE_URL = "https://py.sdk.modelcontextprotocol.io"


# ============================================================
# URL
# ============================================================

def path_to_url(path: Path) -> str:
    """
    Convert a raw document path to its canonical documentation URL.
    """

    relative = path.relative_to(RAW_DIR)

    relative_str = relative.as_posix()

    # Markdown index pages
    if relative_str == "index.md":
        return f"{BASE_URL}/"

    if relative_str.endswith("/index.md"):
        relative_str = relative_str[
            :-len("index.md")
        ]

        return f"{BASE_URL}/{relative_str}"

    # Markdown pages
    if relative_str.endswith(".md"):
        relative_str = relative_str.removesuffix(".md")

        return f"{BASE_URL}/{relative_str}"

    # API reference pages without extensions
    # e.g.
    # api/mcp
    # api/mcp_types
    return f"{BASE_URL}/{relative_str}/"


# ============================================================
# Document ID
# ============================================================

def document_id_from_path(
    path: Path,
) -> str:

    relative = path.relative_to(RAW_DIR)

    parts = list(relative.parts)

    if parts[-1] == "index.md":
        parts = parts[:-1]

    else:
        # Works for both:
        #
        # foo.md
        # api/mcp
        #
        parts[-1] = Path(parts[-1]).stem

    return "mcp-python-sdk-" + "-".join(parts)


# ============================================================
# Markdown title
# ============================================================

def title_from_markdown(
    text: str,
) -> str:

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("# "):
            return line[2:].strip()

    return "Untitled"


# ============================================================
# HTML -> Markdown-like text
# ============================================================

def html_to_markdown(
    html: str,
) -> str:
    """
    Convert MkDocs / mkdocstrings HTML API reference pages
    into clean text that can be processed by the existing
    Markdown cleaning + chunking pipeline.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove elements that are not useful for RAG.
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "header",
            "footer",
        ]
    ):
        tag.decompose()

    # Prefer the main documentation content.
    main = soup.find("main")

    if main is None:
        main = soup.body

    if main is None:
        return soup.get_text(
            "\n",
            strip=True,
        )

    # Convert important HTML structures into Markdown-like
    # representations before extracting text.

    for tag in main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        level = int(tag.name[1])

        text = tag.get_text(
            " ",
            strip=True,
        )

        tag.replace_with(
            f"\n{'#' * level} {text}\n"
        )

    for tag in main.find_all("li"):
        text = tag.get_text(
            " ",
            strip=True,
        )

        tag.replace_with(
            f"\n- {text}\n"
        )

    for tag in main.find_all("pre"):
        text = tag.get_text(
            "\n",
            strip=False,
        )

        tag.replace_with(
            f"\n```\n{text}\n```\n"
        )

    text = main.get_text(
        "\n",
        strip=True,
    )

    return text


# ============================================================
# Load document
# ============================================================

def load_document(
    path: Path,
) -> Document:

    raw_text = path.read_text(
        encoding="utf-8"
    )

    # Markdown
    if path.suffix.lower() == ".md":

        content = clean_markdown(
            raw_text
        )

    # API HTML pages without extensions
    else:

        markdown_like = html_to_markdown(
            raw_text
        )

        content = clean_markdown(
            markdown_like
        )

    return Document(
        id=document_id_from_path(path),
        title=title_from_markdown(content),
        url=path_to_url(path),
        source="mcp-python-sdk",
        path=str(path),
        content=content,
    )


# ============================================================
# Build chunks
# ============================================================

def build_chunks(
    document: Document,
) -> list[Chunk]:

    raw_chunks = chunk_markdown(
        document.content
    )

    chunks = []

    for index, raw_chunk in enumerate(
        raw_chunks
    ):

        chunk = Chunk(
            id=f"{document.id}-{index}",
            document_id=document.id,
            title=raw_chunk["title"],
            section=raw_chunk["section"],
            content=raw_chunk["content"],
            url=document.url,
            source=document.source,
            chunk_index=index,
        )

        chunks.append(chunk)

    return chunks


# ============================================================
# Save documents
# ============================================================

def save_documents_jsonl(
    documents: list[Document],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for document in documents:

            data = document.model_dump(
                exclude={"content"}
            )

            f.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Save chunks
# ============================================================

def save_jsonl(
    chunks: list[Chunk],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Pipeline
# ============================================================

def run_pipeline() -> None:

    markdown_files = sorted(
        RAW_DIR.rglob("*.md")
    )

    # API reference pages are HTML but have no extension.
    api_files = sorted(
        path
        for path in (
            RAW_DIR / "api"
        ).rglob("*")
        if path.is_file()
        and path.suffix == ""
    )

    files = sorted(
        markdown_files + api_files
    )

    print(
        f"Found {len(markdown_files)} Markdown files"
    )

    print(
        f"Found {len(api_files)} API HTML files"
    )

    print(
        f"Total files: {len(files)}"
    )

    documents = []
    all_chunks = []

    for i, path in enumerate(
        files,
        start=1,
    ):

        document = load_document(
            path
        )

        documents.append(
            document
        )

        chunks = build_chunks(
            document
        )

        all_chunks.extend(
            chunks
        )

        print(
            f"[{i}/{len(files)}] "
            f"{document.id}: "
            f"{len(chunks)} chunks"
        )

    save_documents_jsonl(
        documents,
        DOCUMENTS_OUTPUT_FILE,
    )

    save_jsonl(
        all_chunks,
        CHUNKS_OUTPUT_FILE,
    )

    print()

    print(
        f"Created {len(documents)} documents"
    )

    print(
        f"Created {len(all_chunks)} chunks"
    )

    print(
        f"Documents saved to "
        f"{DOCUMENTS_OUTPUT_FILE}"
    )

    print(
        f"Chunks saved to "
        f"{CHUNKS_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_pipeline()