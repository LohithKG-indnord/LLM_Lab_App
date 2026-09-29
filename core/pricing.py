"""Pricing loader used by the LLM Lab modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_PRICING_FILE = _ROOT / "pricing.yaml"


def load_pricing(path: str | Path = _PRICING_FILE) -> dict[str, dict[str, Any]]:
    """Load and normalize model prices from pricing.yaml."""
    pricing_path = Path(path)
    if not pricing_path.exists():
        raise FileNotFoundError(f"Pricing file not found: {pricing_path}")

    raw = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("pricing.yaml must contain a mapping")

    models: dict[str, dict[str, Any]] = {}
    for model_id, entry in raw.items():
        if model_id == "models" or not isinstance(entry, dict):
            continue
        input_price = next(
            (
                entry.get(key)
                for key in ("input_per_million", "input_per_mtok", "input_price", "input")
                if entry.get(key) is not None
            ),
            None,
        )
        output_price = next(
            (
                entry.get(key)
                for key in ("output_per_million", "output_per_mtok", "output_price", "output")
                if entry.get(key) is not None
            ),
            None,
        )
        if input_price is None or output_price is None:
            continue
        normalized = dict(entry)
        normalized["input_per_million"] = float(input_price)
        normalized["output_per_million"] = float(output_price)
        models[str(model_id)] = normalized
    return models


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    prices: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Calculate USD cost from per-million-token prices."""
    prices = load_pricing() if prices is None else prices
    if model not in prices:
        raise KeyError(f"No pricing entry for model: {model}")
    entry = prices[model]
    return (
        input_tokens * float(entry["input_per_million"])
        + output_tokens * float(entry["output_per_million"])
    ) / 1_000_000
