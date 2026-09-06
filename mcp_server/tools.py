from rag.retriever import retrieve


def search_mcp_docs(
    query: str,
    limit: int = 5,
):
    """
    Search the MCP Python SDK documentation.

    Args:
        query: Natural language question.
        limit: Number of results to return.

    Returns:
        Relevant MCP documentation chunks.
    """

    return retrieve(
        query=query,
        limit=limit,
    )