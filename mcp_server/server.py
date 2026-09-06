from mcp.server import MCPServer

from retrieval.search import search
from rag.pipeline import answer


mcp = MCPServer("MCP RAG Assistant")


@mcp.tool()
def search_mcp_docs(
    query: str,
    limit: int = 5,
) -> str:
    """Search the MCP Python SDK documentation."""

    results = search(
        query=query,
        limit=limit,
    )

    if not results:
        return "No relevant documents found."

    output = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        output.append(
            f"""
Rank: {rank}
Similarity: {result['similarity']:.4f}
Title: {result['title']}
Section: {result['section']}
URL: {result['url']}

{result['content']}
""".strip()
        )

    return "\n\n" + (
        "\n" + "=" * 80 + "\n"
    ).join(output)


@mcp.tool()
def ask_mcp_docs(
    query: str,
    limit: int = 5,
) -> str:
    """Answer a question about the MCP Python SDK using retrieved documentation."""

    return answer(
        query=query,
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run()