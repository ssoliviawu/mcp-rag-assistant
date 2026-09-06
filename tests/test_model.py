from ingestion.models import Chunk


def test_chunk_model():
    chunk = Chunk(
        id="test-1",
        document_id="doc-1",
        title="Resources",
        section="Resources",
        content="Resources provide context.",
        url="https://example.com",
        source="test",
        chunk_index=0,
    )

    assert chunk.id == "test-1"
    assert chunk.document_id == "doc-1"
    assert chunk.chunk_index == 0