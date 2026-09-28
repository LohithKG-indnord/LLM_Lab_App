# Proposal — LLM Lab App

Week 5 Assessment · Streamlit application · Provider: Anthropic

## 1. Problem statement

Working with large language models is usually a black box: text goes in,
text comes out, and the intermediate steps (tokenization, attention,
sampling) and the practical trade-offs (cost, latency, quality between
model sizes) stay invisible. This makes it hard to answer everyday
questions such as:

- Why does the same sentence cost different amounts on different models?
- What does a causal mask actually do to the attention pattern?
- What do temperature, top-k and top-p change about an answer?
- Is a smaller, cheaper model good enough for a given task?

## 2. Objective

Build a single Streamlit app, **LLM Lab**, with four tabs that follow one
request through an LLM, and log every real API call so conclusions in the
final report can be traced to measured data.

| Module | Question it answers |
|---|---|
| A. Tokenizer Lens | How does text become tokens, and how does that affect cost? |
| B. Attention Explorer | How does one head of causal self-attention weigh earlier words? |
| C. Sampling Playground | How do temperature, top-k and top-p change output diversity? |
| D. Model Arena | Is the smaller model good enough at lower cost and latency? |

## 3. Scope

**In scope**

- Two local tokenizers (tiktoken `o200k_base`, Hugging Face `gpt2`) plus an
  API token-count check against Claude.
- Single-head causal self-attention in pure NumPy, with unit tests.
- A free toy-distribution view of temperature/top-k/top-p, and real API
  repeated runs scored with distinct-n diversity.
- Side-by-side comparison of Claude Sonnet 5 and Claude Haiku 4.5 with
  latency, token usage, cost and blind human scoring.
- A shared append-only run log (`runs.jsonl`) and a sourced `pricing.yaml`.

**Out of scope**

- Multi-head attention and positional encoding (covered in theory only).
- Running Claude's tokenizer locally (it is not public).
- A second provider, streaming and time-to-first-token (stretch goals).
- Model training or fine-tuning.

## 4. Approach

1. Build each module's logic in its own file, free of UI code where
   possible, so it can be tested and explained independently.
2. Route every API call through one client wrapper and one logger so all
   modules produce comparable records.
3. Read prices only from `pricing.yaml`; compute cost from logged token
   counts.
4. Verify with unit tests (attention), manual acceptance checks per module,
   and a hand-calculated cost spot-check.

## 5. Model selection summary

| Role | Model | Input $/MTok | Output $/MTok |
|---|---|---|---|
| Larger | `claude-sonnet-5` | 2.00 | 10.00 |
| Smaller | `claude-haiku-4-5-20251001` | 1.00 | 5.00 |

Prices from Anthropic's pricing page, recorded in `pricing.yaml` on
2026-09-27. Opus was not chosen because it costs roughly twice as much and
its strengths (long agentic work) are not exercised by short single-turn
prompts. Two models of the same tier would leave nothing to compare.

## 6. Deliverables

- Working Streamlit app (`app.py`, `modules/`, `core/`).
- `runs.jsonl` with real logged runs and an Arena history CSV export.
- `tests/test_attention.py` with passing tests.
- `architecture/proposal.md` and `architecture/design.md`.
- README with setup steps, screenshots and known issues.
- Comparison report built from `runs.jsonl`.

## 7. Success criteria

- Same text always yields the same tokens; same input always yields the
  same attention weights (including after an app restart).
- Every softmax row sums to 1; masked positions have weight 0.
- Toy sampling distribution always sums to 1.
- Both Arena models receive identical prompt, temperature and max_tokens.
- Logged `cost_usd` matches a hand calculation from `pricing.yaml`.
- Missing key, rate limit and timeout show a clear message instead of a
  crash.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Library APIs change between versions (e.g. Anthropic SDK 1.0 removed `temperature` as a direct argument; `transformers` removed tokenizer internals) | Isolate SDK calls in `core/clients.py`; read GPT-2 `merges.txt` directly; pin `anthropic>=1.0,<2` |
| API spend grows with repeats and grid runs | Small default N, short `max_tokens`, running-cost sidebar |
| Non-deterministic behaviour breaks reproducibility | Seeded RNG; stable `crc32` word seeding instead of Python `hash()` |
| Scores not traceable to runs | Shared `prompt_id` across both model runs and their scores |
| Leaked API key | `.env` git-ignored, `.env.example` committed |

## 9. Timeline

| Phase | Work |
|---|---|
| 1 | Repo setup, pricing loader, logger, client wrapper |
| 2 | Tokenizer Lens and Attention Explorer with tests |
| 3 | Sampling Playground (toy view, then real runs and grid mode) |
| 4 | Model Arena, blind scoring, error handling |
| 5 | Real data collection, report, screenshots, README polish |
