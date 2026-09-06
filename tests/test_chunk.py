from ingestion.chunk import chunk_markdown


def test_chunk_markdown():
    text = """
# Resources

Resources provide context.

## URI

Resources have URIs.
"""

    chunks = chunk_markdown(text)

    assert len(chunks) == 2

    assert chunks[0]["title"] == "Resources"
    assert chunks[1]["title"] == "URI"

    assert "Resources provide" in chunks[0]["content"]
    assert "Resources have URIs" in chunks[1]["content"]