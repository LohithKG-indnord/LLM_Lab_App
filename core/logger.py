"""Append-only JSONL logging for measured LLM Lab runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_RUN_LOG = _ROOT / "runs.jsonl"


def log_run(
    *,
    module: str,
    provider: str,
    model: str,
    prompt_id: str,
    parameters: dict[str, Any],
    input_tokens: int,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    output_text: str = "",
    scores: dict[str, Any] | None = None,
    path: str | Path = _RUN_LOG,
) -> dict[str, Any]:

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "provider": provider,
        "model": model,
        "prompt_id": prompt_id,
        "parameters": parameters,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms": round(float(latency_ms), 3),
        "cost_usd": round(float(cost_usd), 10),
        "output_text": output_text,
    }
    if scores is not None:
        record["scores"] = scores

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
