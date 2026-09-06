from rag.retriever import retrieve
from rag.context import build_context
from rag.prompt import SYSTEM_PROMPT, build_prompt
from rag.generator import generate


def answer(
    query: str,
    limit: int = 5,
) -> str:

    # 1. Retrieve relevant chunks
    results = retrieve(
        query,
        limit=limit,
    )

    # 2. Build context
    context = build_context(
        results
    )

    # 3. Build prompt
    user_prompt = build_prompt(
        query,
        context,
    )

    # 4. Generate answer
    response = generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return response