import numpy as np

from embedding.model import BGEEmbeddingModel


def test_embedding():

    model = BGEEmbeddingModel()

    embedding = model.embed_query(
        "What are MCP Resources?"
    )

    print(
        "\nEmbedding shape:",
        embedding.shape,
    )

    print(
        "Embedding norm:",
        np.linalg.norm(embedding),
    )

    print(
        "First 10 values:",
        embedding[:10],
    )

    assert embedding.shape == (384,)

    assert np.isclose(
        np.linalg.norm(embedding),
        1.0,
        atol=1e-5,
    )