import json
from pathlib import Path


INPUT = Path("evaluation/error_analysis.json")
OUTPUT = Path("evaluation/complete_misses.json")


with INPUT.open(
    "r",
    encoding="utf-8",
) as f:
    report = json.load(f)


misses = [
    result
    for result in report["results"]
    if result["error_type"] == "COMPLETE_MISS"
]


output = {
    "mode": report["mode"],
    "retrieval_limit": report["retrieval_limit"],
    "count": len(misses),
    "complete_misses": misses,
}


with OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    f"Found {len(misses)} COMPLETE_MISS queries."
)

print(
    f"Output: {OUTPUT.resolve()}"
)