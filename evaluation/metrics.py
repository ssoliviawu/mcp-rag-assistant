def calculate_hit_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:

    retrieved_at_k = retrieved[:k]

    relevant_set = set(relevant)

    return float(
        any(
            source in relevant_set
            for source in retrieved_at_k
        )
    )


def calculate_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:

    if not relevant:
        return 0.0

    retrieved_at_k = set(
        retrieved[:k]
    )

    relevant_set = set(
        relevant
    )

    found = (
        retrieved_at_k
        & relevant_set
    )

    return len(found) / len(
        relevant_set
    )


def calculate_mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float:

    relevant_set = set(
        relevant
    )

    for rank, source in enumerate(
        retrieved,
        start=1,
    ):

        if source in relevant_set:
            return 1.0 / rank

    return 0.0