from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = Path("models/bge-small-en-v1.5")

QUERY_PREFIX = (
    "Represent this sentence for searching "
    "relevant passages: "
)


class BGEEmbeddingModel:

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
    ):
        self.model_dir = Path(model_dir)

        self.onnx_model = (
            self.model_dir
            / "onnx"
            / "model.onnx"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_dir
            )
        )

        self.session = (
            ort.InferenceSession(
                str(self.onnx_model),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.input_names = {
            item.name
            for item in self.session.get_inputs()
        }

    def _tokenize(
        self,
        texts: list[str],
    ):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

    @staticmethod
    def _mean_pooling(
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:

        mask = attention_mask[
            :, :, None
        ].astype(np.float32)

        masked_embeddings = (
            token_embeddings * mask
        )

        summed = masked_embeddings.sum(
            axis=1
        )

        counts = np.clip(
            mask.sum(axis=1),
            a_min=1e-9,
            a_max=None,
        )

        return summed / counts

    @staticmethod
    def _normalize(
        embeddings: np.ndarray,
    ) -> np.ndarray:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        return embeddings / np.clip(
            norms,
            a_min=1e-12,
            a_max=None,
        )

    def embed_documents(
    self,
    texts: list[str],
    batch_size: int = 16,
) -> np.ndarray:

        if not texts:
            return np.empty(
            (0, 384),
            dtype=np.float32,
        )

        all_embeddings = []

        for start in range(
        0,
        len(texts),
        batch_size,
        ):
            batch_texts = texts[
            start:start + batch_size
            ]

            encoded = self._tokenize(
            batch_texts
            )

            inputs = {
            "input_ids":
                encoded["input_ids"].astype(
                    np.int64
                ),

            "attention_mask":
                encoded["attention_mask"].astype(
                    np.int64
                ),

            "token_type_ids":
                encoded["token_type_ids"].astype(
                    np.int64
                ),
            }

            outputs = self.session.run(
            None,
            inputs,
            )

            token_embeddings = outputs[0]

            embeddings = self._mean_pooling(
            token_embeddings,
            encoded["attention_mask"],
            )

            embeddings = self._normalize(
            embeddings
            )

            all_embeddings.append(
            embeddings.astype(
                np.float32
                )
            )

        return np.vstack(
        all_embeddings
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        query = (
            QUERY_PREFIX + query
        )

        return self.embed_documents(
            [query]
        )[0]