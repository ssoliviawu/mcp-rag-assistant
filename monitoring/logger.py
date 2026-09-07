import json
from datetime import datetime, timezone
from pathlib import Path


MONITORING_DIR = Path(__file__).resolve().parent
METRICS_FILE = MONITORING_DIR / "metrics.jsonl"


def log_request(
    question,
    success,
    result_count=None,
    model=None,
    generation_latency_ms=None,
    total_latency_ms=None,
    error=None,
):
    """
    Log one RAG request as a JSONL record.
    """

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "success": success,
        "result_count": result_count,
        "model": model,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "error": error,
    }

    with open(
        METRICS_FILE,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )