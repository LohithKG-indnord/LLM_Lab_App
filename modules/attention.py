"""A small, inspectable implementation of single-head causal self-attention.

This module intentionally uses only NumPy for the attention calculation.  It is
an educational implementation: every intermediate matrix is returned so the
Streamlit UI can show how the result was produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import unicodedata
from typing import Any
import zlib

import numpy as np
try:
    import streamlit as st
except ModuleNotFoundError:  # Keep the pure math functions usable in test-only environments.
    st = None  # type: ignore[assignment]


MASK_VALUE = -1e9
SUPPORTED_DIMS = (8, 16)
EMOJI_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2300, 0x23FF))


@dataclass(frozen=True)
class AttentionResult:
    """All values produced by one deterministic attention run."""

    tokens: list[str]
    embeddings: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    raw_scores: np.ndarray
    scaled_scores: np.ndarray
    mask: np.ndarray
    masked_scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


def tokenize_words(text: str) -> list[str]:
    """Split on whitespace and separate adjacent standalone emoji characters.

    This is intentionally not a full language tokenizer.  It keeps ordinary
    words and punctuation together, but makes ``😀😃😄`` three tokens and keeps
    common joined emoji sequences (for example skin-tone or ZWJ sequences)
    together.
    """
    def is_emoji_base(character: str) -> bool:
        codepoint = ord(character)
        return any(start <= codepoint <= end for start, end in EMOJI_RANGES)

    def consume_emoji(value: str, start: int) -> tuple[str, int]:
        end = start + 1
        while end < len(value):
            codepoint = ord(value[end])
            if (unicodedata.combining(value[end]) or 0xFE00 <= codepoint <= 0xFE0F
                    or 0x1F3FB <= codepoint <= 0x1F3FF):
                end += 1
            elif value[end] == "\u200d" and end + 1 < len(value) and is_emoji_base(value[end + 1]):
                end += 2
            else:
                break
        return value[start:end], end

    tokens: list[str] = []
    for chunk in text.split():
        current: list[str] = []
        index = 0
        while index < len(chunk):
            if is_emoji_base(chunk[index]):
                if current:
                    tokens.append("".join(current))
                    current = []
                emoji, index = consume_emoji(chunk, index)
                tokens.append(emoji)
            else:
                current.append(chunk[index])
                index += 1
        if current:
            tokens.append("".join(current))
    return tokens


def causal_mask(length: int, mask_value: float = MASK_VALUE) -> np.ndarray:
    """Return a lower-triangular additive causal mask."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return np.where(np.triu(np.ones((length, length), dtype=bool), k=1), mask_value, 0.0)


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax, including rows containing large negatives."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values.copy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def word_embedding(word: str, dimension: int) -> np.ndarray:
    """Create a process-stable deterministic embedding for one word."""
    if dimension not in SUPPORTED_DIMS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMS}")
    seed = zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed).normal(0.0, 1.0, dimension)


def embeddings_for_tokens(tokens: list[str], dimension: int) -> np.ndarray:
    if not tokens:
        return np.empty((0, dimension), dtype=float)
    return np.vstack([word_embedding(token, dimension) for token in tokens])


def _projection_matrices(dimension: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(dimension)
    matrices = [rng.normal(0.0, scale, (dimension, dimension)) for _ in range(3)]
    return matrices[0], matrices[1], matrices[2]


def run_attention(
    tokens: list[str] | str,
    dimension: int = 8,
    seed: int = 7,
    causal: bool = True,
    scale: bool = True,
) -> AttentionResult:
    """Run one deterministic head of scaled attention; causal by default."""
    if isinstance(tokens, str):
        tokens = tokenize_words(tokens)
    if dimension not in SUPPORTED_DIMS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMS}")
    x = embeddings_for_tokens(tokens, dimension)
    w_q, w_k, w_v = _projection_matrices(dimension, seed)
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    raw = q @ k.T
    scaled = raw / np.sqrt(dimension) if scale else raw.copy()
    mask = causal_mask(len(tokens)) if causal else np.zeros((len(tokens), len(tokens)))
    masked = scaled + mask
    weights = softmax(masked, axis=1)
    output = weights @ v
    return AttentionResult(list(tokens), x, q, k, v, raw, scaled, mask, masked, weights, output)


def _matrix_rows(matrix: np.ndarray, labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Convert a matrix into readable rows for Streamlit."""
    matrix = np.asarray(matrix)
    rows = []
    for index, row in enumerate(matrix):
        item: dict[str, Any] = {"position": index, "token": labels[index] if labels and index < len(labels) else ""}
        item.update({f"d{column}": float(value) for column, value in enumerate(row)})
        rows.append(item)
    return rows


def _show_matrix(title: str, matrix: np.ndarray, tokens: list[str] | None = None) -> None:
    st.markdown(f"**{title}**")
    matrix = np.asarray(matrix)
    if matrix.ndim == 2 and matrix.shape[0] <= 32:
        st.dataframe(_matrix_rows(matrix, tokens), hide_index=True, use_container_width=True)
    else:
        st.write(matrix)


def _weights_heatmap(
    weights: np.ndarray,
    tokens: list[str],
    title: str,
    colour_scale: str,
) -> None:
    import plotly.express as px

    labels = [f"{i}: {token}" for i, token in enumerate(tokens)]
    figure = px.imshow(
        weights,
        x=labels,
        y=labels,
        text_auto=".2f",
        color_continuous_scale=colour_scale,
        aspect="auto",
        labels={"x": "Key token", "y": "Query token", "color": "weight"},
    )
    figure.update_layout(title=title, height=max(350, 45 * len(tokens)), margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(figure, use_container_width=True)


def render() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to render Attention Explorer. Install requirements.txt first.")
    st.header("Attention Explorer")
    st.write("Follow one head of causal self-attention from words to attention weights and output vectors.")

    text = st.text_area("Text (split into tokens on whitespace)", "The cat sat on the mat", height=90)
    dimension = st.selectbox("Embedding dimension", SUPPORTED_DIMS, key="attention_dimension")
    mask_enabled = st.checkbox(
        "Apply causal mask",
        value=True,
        key="attention_mask",
        help="When selected, tokens can attend only to themselves and earlier tokens.",
    )
    scale_enabled = st.checkbox(
        "Scale scores by √dₖ",
        value=True,
        key="attention_scale",
        help="When selected, use the standard QKᵀ / √dₖ scaling before softmax.",
    )
    tokens = tokenize_words(text)
    if not tokens:
        st.warning("Enter at least one token to explore attention.")
        return
    if len(tokens) > 32:
        st.warning("Please use 32 tokens or fewer so the matrices remain readable.")
        return

    result = run_attention(tokens, int(dimension), causal=mask_enabled, scale=scale_enabled)
    before_mask = run_attention(tokens, int(dimension), causal=False, scale=scale_enabled)
    st.subheader("Tokens")
    st.markdown(" ".join(f'<span style="background:#E8F1FF;padding:4px 7px;margin:2px;border-radius:4px;display:inline-block">{i}: {escape(token)}</span>' for i, token in enumerate(tokens)), unsafe_allow_html=True)

    st.subheader("Attention weights")
    st.caption("The blue heatmap is the unmasked reference. The orange heatmap follows the mask button. The scale button affects both heatmaps.")
    before_column, after_column = st.columns(2)
    with before_column:
        _weights_heatmap(before_mask.weights, result.tokens, "Before causal mask", "Blues")
    with after_column:
        _weights_heatmap(result.weights, result.tokens, "After causal mask", "Oranges")
    st.dataframe(
        [{"query": token, "row sum": float(row.sum()), "attends to": ", ".join(f"{t}: {w:.3f}" for t, w in zip(tokens, row) if w > 0)} for token, row in zip(tokens, result.weights)],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Show detailed calculations", expanded=False):
        st.latex(r"Q=XW_Q,\quad K=XW_K,\quad V=XW_V")
        st.latex(r"S=QK^T/\sqrt{d_k},\quad A=\operatorname{softmax}(S+M),\quad O=AV")
        _show_matrix("Input embeddings (X)", result.embeddings, result.tokens)
        for title, matrix in (("Queries (Q)", result.q), ("Keys (K)", result.k), ("Values (V)", result.v), ("Raw scores (QKᵀ)", result.raw_scores), ("Scaled scores", result.scaled_scores), ("Causal mask (0 / -1e9)", result.mask), ("Masked scores", result.masked_scores), ("Output (A V)", result.output)):
            _show_matrix(title, matrix, result.tokens)

    st.caption(f"Deterministic run: dimension={dimension}. The same input produces the same attention weights across app restarts.")
