from retrieval.search import search


def retrieve(
    query: str,
    limit: int = 5,
):
    return search(
        query=query,
        limit=limit,
    )