from rag.retriever import retrieve
from rag.context import build_context


def test_rag_context():

    query = "What are MCP resources?"

    results = retrieve(
        query,
        limit=5,
    )

    assert len(results) > 0

    context = build_context(results)

    assert "resource" in context.lower()

    print("\n")
    print(context)