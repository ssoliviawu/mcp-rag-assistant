from mcp_server.tools import search_mcp_docs


def test_search_mcp_docs():

    results = search_mcp_docs(
        "How do I create an MCP tool in Python?",
        limit=3,
    )

    assert len(results) == 3

    for result in results:
        assert "content" in result
        assert "title" in result
        assert "url" in result
        assert "similarity" in result