# Changelog

All notable changes to this project will be documented in this file.

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