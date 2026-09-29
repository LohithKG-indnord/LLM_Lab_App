# Changelog

All notable changes to this project will be documented in this file.

## 29-09-2026

**commit:** `feat: Created working attention explorer tab`

### Added

- Added a pure NumPy implementation of single-head scaled causal self-attention.
- Added deterministic CRC32-based word embeddings and seeded projection matrices.
- Added simple token handling that separates adjacent emojis while preserving joined emoji sequences.
- Added Q, K, V, raw scores, scaled scores, causal mask, masked scores, attention weights, and `A × V` output.
- Added mask and `QKᵀ / √dₖ` scaling controls.
- Added side-by-side comparison heatmaps before and after the selected causal mask, with distinct colors.
- Added tests for softmax rows, causal masking, determinism, dimensions, emoji tokenization, and scaling.

### Updated

- Updated `app.py` to render the Attention Explorer tab.
- Updated the Attention Explorer UI to keep detailed matrices in a collapsed section.
- Updated `HANDOFF.md` with the completed Attention Explorer workstream and validation status.

## 29-09-2026

**commit:** `feat: Created working tokenizer_lens tab`

### Added

- Completed Tokenizer Lens UI.
- Added tiktoken `o200k_base` tokenization.
- Added Hugging Face GPT-2 tokenization.
- Added preset examples for English, code, non-English text, emoji, and numbers.
- Added color-coded token display.
- Added token counts, token IDs, and characters per token.
- Added special-token comparison.
- Added GPT-2 BPE merge-step visualizer.
- Added Anthropic API token counting for Sonnet and Haiku.
- Added local estimate versus API count comparison.
- Added API latency and cost display.

### Updated

- Updated `app.py` to load `.env` and render the Tokenizer Lens safely.
- Added pricing loading and cost calculation in `core/pricing.py`.
- Added JSONL logging in `core/logger.py`.
- Updated `pricing.yaml` with Anthropic model IDs and prices.

## 28-09-2026

**commit:** `chore: Initialize project structure`

### Added

- Created Initial project structure.
- Created Initial required docs files.

## 28-09-2026

**commit:** `docs : add architecture`

### Added

- Architecture documentation for project structure
- Folder organization guide for modules and utils
