from typing import Dict, Any


def format_metrics(
    result: Dict[str, Any]
) -> str:

    return f"""
### 📊 Performance

| Metric | Value |
|---|---:|
| Retrieval Time | {result["retrieval_time"]:.3f} sec |
| Generation Time | {result["generation_time"]:.3f} sec |
| Total Time | {result["total_time"]:.3f} sec |
| Retrieved Chunks | {result["num_chunks"]} |
"""