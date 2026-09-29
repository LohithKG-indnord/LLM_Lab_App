# LLM Lab App Handoff

## Current stage

The Tokenizer Lens tab is the current completed workstream. It is implemented
against the Week 5 assessment requirements for the tokenizer module. The other
three dashboard tabs remain placeholders and are not part of this handoff.

## Completed Tokenizer Lens capabilities

- Preset examples for English, code, non-English text, emoji, long numbers, and custom text.
- Local comparison using tiktoken o200k_base and Hugging Face GPT2Tokenizer with gpt2.
- Color-coded token display.
- Token counts, token IDs, and character counts per token.
- Input-cost estimates loaded from pricing.yaml.
- Special-token inspection showing literal and special-aware behavior.
- GPT-2 BPE merge replay using merges.txt, including rank, pair, and symbol states.
- Anthropic token-count comparison for both claude-sonnet-5 and claude-haiku-4-5-20251001.
- Per-model API token count, local estimate, difference, latency, and estimated cost.
- Append-only tokenizer records in runs.jsonl with a shared comparison ID.
- .env loading and clear error messages for missing keys or unavailable dependencies.

## Files completed or updated for this stage

- app.py: loads .env, renders the four tabs, and protects the Tokenizer tab with setup errors.
- modules/tokenizer_lens.py: tokenizer UI, local tokenization, cost display, API comparison, BPE replay, and summary views.
- core/pricing.py: loads and normalizes pricing data and calculates USD cost.
- core/logger.py: appends complete JSON records to runs.jsonl.
- pricing.yaml: contains Anthropic model IDs, per-million-token prices, dates, and source URLs.
- requirements.txt: adjusted for Python 3.12-compatible Windows wheels.

## Environment setup

The machine uses the python command rather than the Windows py launcher.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --only-binary=:all: -r requirements.txt
    streamlit run app.py

The project root must be the current directory when running Streamlit.

## Environment variables

Create a .env file in the project root:

    ANTHROPIC_API_KEY=your_key_here

Never commit .env or expose the key in runs.jsonl.

## Tokenizer API behavior

The API button performs two Anthropic token-count requests, one per model. It
does not generate a response, so output_tokens is recorded as zero. Each
successful request is logged with:

- UTC timestamp
- module and provider
- model ID
- shared comparison prompt_id
- tokenizer parameters
- input and output token counts
- latency in milliseconds
- estimated USD cost

The local tiktoken and GPT-2 values are estimates and are not expected to match
Anthropic exactly because the vocabularies and tokenization rules differ.

## Validation completed

- app.py, modules/tokenizer_lens.py, core/pricing.py, and core/logger.py pass Python syntax compilation.
- core.pricing.estimate_cost successfully calculates cost using normalized pricing entries.
- git diff --check passes for the current edits.
- GPT-2 BPE replay was checked with changed; ranks and pairs progress through expected merges.

## Known limitations

- The Anthropic API check requires a recent Anthropic SDK and a valid API key.
- The first GPT-2 run needs network access to download tokenizer files and merges.txt into the Hugging Face cache.
- runs.jsonl contains prompts and outputs when other modules are implemented; only safe, non-confidential test data should be used.
- core/clients.py is not yet implemented; the Tokenizer module currently falls back to the Anthropic SDK directly when no client wrapper exists.
- Attention Explorer, Sampling Playground, and Model Arena are still marked Coming soon in app.py.

