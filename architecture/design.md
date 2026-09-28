# Design — LLM Lab App

## 1. Architecture overview

```
                        ┌──────────────────────────┐
                        │        app.py            │
                        │  Streamlit, 4 tabs +     │
                        │  sidebar running cost    │
                        └─────────────┬────────────┘
      ┌───────────────┬───────────────┼────────────────┬────────────────┐
      ▼               ▼               ▼                ▼
┌───────────┐  ┌────────────┐  ┌──────────────┐  ┌────────────┐
│ tokenizer │  │ attention  │  │  sampling    │  │   arena    │
│ _lens.py  │  │ .py (NumPy)│  │  .py         │  │   .py      │
└─────┬─────┘  └────────────┘  └──────┬───────┘  └─────┬──────┘
      │  (no API)                     │                │
      │                               └───────┬────────┘
      ▼                                       ▼
┌───────────┐                       ┌───────────────────┐
│ pricing.py│◄──────────────────────│   clients.py      │──► Anthropic API
│ + yaml    │                       │ send_chat,        │
└───────────┘                       │ count_tokens      │
      ▲                             └─────────┬─────────┘
      │                                       ▼
      └────────────────────────────── logger.py ──► runs.jsonl
```

Design rule: modules contain logic, `app.py` contains only widgets, and
anything that touches the network or the log goes through `core/`.

## 2. Repository layout

```
llm-lab/
  app.py
  modules/  tokenizer_lens.py  attention.py  sampling.py  arena.py
  core/     clients.py  pricing.py  logger.py
  tests/    test_attention.py
  architecture/  proposal.md  design.md
  docs/     design_decisions.md
  pricing.yaml  runs.jsonl  requirements.txt  .env.example  README.md
```

## 3. Component design

### 3.1 Tokenizer Lens (`modules/tokenizer_lens.py`)

- `tokenize_with_tiktoken` (`o200k_base`) and `tokenize_with_hf`
  (`GPT2Tokenizer`, the pure-Python version, avoiding the Rust `tokenizers`
  install issues seen on Windows).
- Each returns a `TokenizationResult` (tokens, ids, chars per token).
- `estimate_costs_for_models` prices the token count for both Claude models.
- `bpe_merge_steps` reads GPT-2 `merges.txt` from the Hugging Face Hub and
  replays merges by rank, using a local `_bytes_to_unicode()` mapping.
  It deliberately avoids `tok.byte_encoder` and `tok.bpe_ranks`, which newer
  `transformers` versions no longer expose.
- The "Check against the API" button calls `count_tokens`. Claude's tokenizer
  is not public, so local counts are estimates, not ground truth.

### 3.2 Attention Explorer (`modules/attention.py`)

Single head, pure NumPy, no framework attention layers.

```
X (n×d_model) → Q = XW_Q, K = XW_K, V = XW_V
scores = QKᵀ / √d_k          (scaling optional)
scores += causal_mask         (0 on/below diagonal, −1e9 above)
weights = softmax(scores)     (row-wise, max-subtracted)
output = weights · V
```

- `d_model` ∈ {8, 16}; weights scaled by 1/√d_model; RNG via
  `np.random.default_rng(seed)`.
- Mask uses −1e9 rather than −inf to avoid NaN in fully masked rows.
- Word embeddings are seeded per word with `zlib.crc32(word)`, not Python's
  `hash()`, which is randomised per process and would break determinism
  across restarts.
- Every intermediate (Q, K, V, raw, scaled, masked, weights, output) is
  returned so the UI can display it.

### 3.3 Sampling Playground (`modules/sampling.py`)

- **Toy view** (no cost): fixed 10-token logits → temperature → softmax →
  top-k → top-p, each with renormalisation. Temperature is clamped at 1e-6
  to avoid division by zero.
- **Real view**: `run_repeated` calls the API N times at one setting;
  `run_grid` sweeps exactly one parameter while others stay at defaults.
- **Scoring**: `distinct_n` (unique n-grams / total n-grams) for n = 1, 2,
  plus unique-output count.

### 3.4 Model Arena (`modules/arena.py`)

- `run_arena` sends the same prompt, temperature and max_tokens to Sonnet 5
  and Haiku 4.5 concurrently using `ThreadPoolExecutor` (calls are I/O
  bound).
- `_safe_call` catches failures per model and maps them to friendly
  messages (missing key, rate limit, timeout) so one failure does not crash
  the other model's result or the app.
- Both runs share one `prompt_id`, returned to the UI so blind scores can
  be logged against the same ID.
- `record_blind_scores` appends `module="arena_score"` rows instead of
  editing earlier rows, keeping the log append-only.
- `run_batch` runs a prompt set for aggregation (stretch goal).

### 3.5 Core services

**`core/clients.py`** — one place that builds the client from
`ANTHROPIC_API_KEY`, times calls with `perf_counter`, and extracts text and
usage. Since `anthropic` SDK 1.0, `temperature`, `top_p` and `top_k` are no
longer direct arguments to `messages.create()`; they are sent through
`extra_body`.

**`core/pricing.py`** — loads `pricing.yaml`;
`cost = in_tokens × in_price/1e6 + out_tokens × out_price/1e6`.

**`core/logger.py`** — appends one JSON object per call to `runs.jsonl`.

## 4. Data design

### 4.1 `runs.jsonl` record

| Field | Description |
|---|---|
| timestamp | UTC ISO-8601 |
| module | `arena`, `arena_score`, … |
| provider, model | e.g. `anthropic`, `claude-haiku-4-5-20251001` |
| prompt_id | shared across both models and their scores |
| parameters | temperature, top_p, top_k, max_tokens as used |
| input_tokens, output_tokens | from `response.usage` |
| latency_ms | end-to-end wall time |
| cost_usd | computed from `pricing.yaml` |
| output_text | model response |
| scores | accuracy, instruction_following, clarity_style (1–5) |

### 4.2 `pricing.yaml`

Per-million-token input/output prices per `model_id`, with source URL and
date copied. Prices are never hard-coded in Python.

## 5. Request flow (Model Arena)

1. User enters prompt, temperature, max_tokens and clicks **Send**.
2. `run_arena` submits two concurrent `_safe_call`s.
3. Each call → `send_chat` → Anthropic API → text + usage + latency.
4. `estimate_cost` prices each result from `pricing.yaml`.
5. Successful runs are logged with a shared `prompt_id`.
6. UI shows results side by side, names hidden under blind mode.
7. User scores 1–5 on three criteria; **Save** logs `arena_score` rows with
   the same `prompt_id`.

## 6. Error handling

| Condition | Behaviour |
|---|---|
| No `ANTHROPIC_API_KEY` | Clear message telling the user to fill `.env` |
| Rate limit | Message shown for that model; the other model still displays |
| Timeout | Message suggesting retry or a shorter prompt |
| Other exception | Generic message with exception class; details in terminal |
| Missing pricing entry | `KeyError` naming the model id |

## 7. Key design decisions

| Decision | Reason |
|---|---|
| Sonnet 5 + Haiku 4.5 | Opposite ends of the cost/quality curve, which is what the Arena tests |
| tiktoken + GPT-2 tokenizers | Two different families; GPT-2 merges are readable |
| Pure-Python `GPT2Tokenizer` | Avoids Rust `tokenizers` install problems |
| −1e9 mask | Avoids NaN from −inf |
| `crc32` word seeding | Determinism across processes |
| Same temperature and max_tokens for both models | Fair comparison |
| Threads for concurrency | Network-bound work |
| Append-only log | Auditable, traceable report numbers |
| `extra_body` for sampling params | Required by `anthropic` ≥ 1.0 |

## 8. Testing strategy

| Level | What |
|---|---|
| Unit (`tests/test_attention.py`) | Softmax rows sum to 1; output shape; future weights are 0; masked rows still sum to 1; determinism; hand-worked 2-token example; mask matrix values |
| Manual acceptance | Token counts stable; toy distribution sums to 1; grid changes one parameter; both Arena models get identical settings |
| Cost audit | Recompute one `cost_usd` by hand from `pricing.yaml` |
| Failure paths | Blank the API key and confirm a clean message |

## 9. Security and configuration

- Key lives only in `.env` (git-ignored); `.env.example` is committed.
- Dependencies pinned in `requirements.txt` (`anthropic>=1.0,<2`).
- `runs.jsonl` stores prompts and outputs, so avoid entering sensitive data.

## 10. Known limitations and future work

- Single head, no positional encoding, by scope.
- No automatic retry or backoff on rate limits.
- Latency is total round trip, not time to first token.
- Possible extensions: second provider column, streaming, batch mode
  aggregation, embedding-based diversity scoring.
