
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ============================================================
# Model paths
# ============================================================

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "reranker"
    / "ms-marco-MiniLM-L-6-v2"
)

ONNX_MODEL = (
    MODEL_DIR
    / "onnx"
    / "model.onnx"
)


# ============================================================
# Load tokenizer
# ============================================================

_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ============================================================
# Load ONNX model
# ============================================================

_session = ort.InferenceSession(
    str(ONNX_MODEL),
    providers=[
        "CPUExecutionProvider"
    ],
)


# ============================================================
# Rerank
# ============================================================

def rerank(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Rerank retrieval results using
    ms-marco-MiniLM-L-6-v2 ONNX model.

    Parameters
    ----------
    query:
        User query.

    results:
        Candidate retrieval results.

    Returns
    -------
    list[dict]:
        Results sorted by reranker_score
        in descending order.
    """

    if not results:
        return []

    
    # --------------------------------------------------------
    # Build query-document pairs
    # --------------------------------------------------------

    pairs = [
        (
            query,
            result.get(
                "content",
                "",
            ),
        )
        for result in results
    ]
    
    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    encoded = _tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    # --------------------------------------------------------
    # ONNX inputs
    # --------------------------------------------------------

    inputs = {
        "input_ids": (
            encoded["input_ids"]
            .astype(np.int64)
        ),
        "attention_mask": (
            encoded["attention_mask"]
            .astype(np.int64)
        ),
    }

    if "token_type_ids" in encoded:

        inputs["token_type_ids"] = (
            encoded["token_type_ids"]
            .astype(np.int64)
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    outputs = _session.run(
        None,
        inputs,
    )
    
    scores = np.asarray(
            outputs[0]
        ).reshape(-1)
    
    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    reranked = []

    for original_index, (result, score) in enumerate(
    zip(results, scores)
    ):
        reranked.append(
        {
            **result,
            "_original_index": original_index,
            "reranker_score": float(score),
        }
        )

    # --------------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------------

    reranked.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked

