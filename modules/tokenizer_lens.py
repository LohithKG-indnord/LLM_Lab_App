from __future__ import annotations
from dataclasses import dataclass
from html import escape
from pathlib import Path
import time
from typing import Any, Iterable
from uuid import uuid4

import streamlit as st

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_MODELS = (
    ("Sonnet", "claude-sonnet-5"),
    ("Haiku", _ANTHROPIC_MODEL),
)


@dataclass(frozen=True)
class TokenizationResult:
    tokens: list[str]
    ids: list[int]
    chars_per_token: list[int]

    @property
    def token_count(self) -> int:
        return len(self.ids)


@st.cache_resource(show_spinner=False)
def _get_tiktoken_encoding(name: str = "o200k_base") -> Any:
    import tiktoken
    return tiktoken.get_encoding(name)


@st.cache_resource(show_spinner=False)
def _get_hf_tokenizer(name: str = "gpt2") -> Any:
    from transformers import GPT2Tokenizer
    return GPT2Tokenizer.from_pretrained(name)


def tokenize_with_tiktoken(text: str, encoding_name: str = "o200k_base") -> TokenizationResult:
    encoding = _get_tiktoken_encoding(encoding_name)
    ids = list(encoding.encode(text, disallowed_special=()))
    tokens = []
    for token_id in ids:
        try:
            token = encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        except (AttributeError, ValueError):
            token = encoding.decode([token_id])
        tokens.append(token)
    return TokenizationResult(tokens, ids, [len(token) for token in tokens])


def tokenize_with_hf(text: str, tokenizer_name: str = "gpt2") -> TokenizationResult:
    tokenizer = _get_hf_tokenizer(tokenizer_name)
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = []
    for token_id, raw_token in zip(ids, raw_tokens):
        decoded = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False
        )
        tokens.append(decoded if decoded else str(raw_token))
    return TokenizationResult(tokens, ids, [len(token) for token in tokens])


def _load_prices() -> dict[str, dict[str, Any]]:
    try:
        from core.pricing import load_pricing
        value = load_pricing()
        return value if isinstance(value, dict) else {}
    except (ImportError, AttributeError, FileNotFoundError, TypeError, ValueError):
        return {}


def estimate_costs_for_models(
    token_count: int, prices: dict[str, dict[str, Any]] | None = None
) -> dict[str, float]:
    """Estimate input cost using only prices loaded from pricing.yaml."""
    prices = _load_prices() if prices is None else prices
    estimates: dict[str, float] = {}
    for model, entry in prices.items():
        if not isinstance(entry, dict):
            continue
        raw = next(
            (entry.get(key) for key in ("input_per_million", "input_price", "input")
             if entry.get(key) is not None),
            None,
        )
        if raw is not None:
            estimates[model] = token_count * float(raw) / 1_000_000
    return estimates


def _bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values.copy()
    used = set(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in used:
            used.add(byte)
            byte_values.append(byte)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


def bpe_merge_steps(word: str, merges_path: str | Path | None = None) -> list[dict[str, Any]]:
    if not word:
        return []
    if merges_path is None:
        tokenizer = None
        try:
            tokenizer = _get_hf_tokenizer("gpt2")
            merges_path = getattr(tokenizer, "merges_file", None)
            if not merges_path:
                merges_path = getattr(tokenizer, "init_kwargs", {}).get("merges_file")
        except (AttributeError, OSError, RuntimeError):
            tokenizer = None
        if not merges_path or not Path(str(merges_path)).exists():
            try:
                from huggingface_hub import hf_hub_download
                merges_path = hf_hub_download(repo_id="gpt2", filename="merges.txt")
            except Exception:
                return []
    path = Path(merges_path)
    if not path.exists():
        return []
    ranks: dict[tuple[str, str], int] = {}
    if tokenizer is not None:
        tokenizer_ranks = getattr(tokenizer, "bpe_ranks", None)
        if isinstance(tokenizer_ranks, dict):
            ranks = {
                (str(pair[0]), str(pair[1])): int(rank)
                for pair, rank in tokenizer_ranks.items()
                if isinstance(pair, tuple) and len(pair) == 2
            }
    if not ranks:
        for rank, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            fields = line.strip().split()
            if len(fields) == 2 and not fields[0].startswith("#"):
                ranks[(fields[0], fields[1])] = rank
    byte_encoder = getattr(tokenizer, "byte_encoder", None) if tokenizer is not None else None
    if isinstance(byte_encoder, dict):
        symbols = [byte_encoder[value] for value in word.encode("utf-8")]
    else:
        byte_map = _bytes_to_unicode()
        symbols = [byte_map[value] for value in word.encode("utf-8")]
    steps = [{
        "rank": "initial",
        "pair": "before merges",
        "symbols": symbols.copy(),
    }]
    while len(symbols) > 1:
        candidates = [
            (ranks[(symbols[i], symbols[i + 1])], (symbols[i], symbols[i + 1]), i)
            for i in range(len(symbols) - 1)
            if (symbols[i], symbols[i + 1]) in ranks
        ]
        if not candidates:
            break
        rank, pair, index = min(candidates, key=lambda item: item[0])
        symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
        steps.append({
            "rank": rank,
            "pair": pair,
            "symbols": symbols.copy(),
        })
    return steps


def _colour_tokens(tokens: Iterable[str]) -> str:
    colours = ("#E8F1FF", "#FFF0D9", "#E9F7EF", "#F5E8FF", "#FFE8EE", "#E7F7F7")
    spans = []
    for index, token in enumerate(tokens):
        visible = escape(token).replace(" ", "·").replace("\n", "↵")
        spans.append(
            f'<span style="background:{colours[index % len(colours)]}; padding:3px 5px; '
            f'margin:2px; border-radius:4px; display:inline-block">{visible or "∅"}</span>'
        )
    return "".join(spans) or '<span style="color:#777">(no tokens)</span>'


def _result_table(result: TokenizationResult) -> list[dict[str, Any]]:
    return [
        {"#": index, "token": token, "id": token_id, "characters": chars}
        for index, (token, token_id, chars) in enumerate(
            zip(result.tokens, result.ids, result.chars_per_token), start=1
        )
    ]


def _anthropic_count_tokens(text: str, model: str) -> int:
    try:
        from core import clients
        counter = getattr(clients, "count_tokens", None)
        if counter is not None:
            try:
                result = counter(text, model)
            except TypeError:
                try:
                    result = counter(model, text)
                except TypeError:
                    result = counter(text)
            if isinstance(result, dict):
                for key in ("input_tokens", "token_count", "count"):
                    if key in result:
                        return int(result[key])
                raise RuntimeError("count_tokens returned no token count field")
            return int(getattr(result, "input_tokens", result))
    except (ImportError, AttributeError):
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import anthropic
    client = anthropic.Anthropic()
    if not hasattr(client.messages, "count_tokens"):
        raise RuntimeError("Installed Anthropic SDK has no count_tokens endpoint.")
    response = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
    return int(response.input_tokens)


def render() -> None:
    st.header("Tokenizer Lens")
    st.write("See how the same text becomes different tokens, and how that changes estimated input cost.")
    presets = {
        "Plain English": "The quick brown fox jumps over the lazy dog.",
        "Code": "def greet(name):\n    return f'Hello, {name}!'",
        "Non-English": "Bonjour, comment allez-vous aujourd'hui ?",
        "Emoji": "Learning LLMs is fun 🚀🤗✨",
        "Long number": "The reference number is 123456789012345678901234567890.",
        "Custom": "",
    }
    choice = st.selectbox("Preset example", list(presets))
    text = st.text_area("Text to tokenize", value=presets[choice], height=120)
    if not text:
        st.info("Enter text to begin the comparison.")
        return
    try:
        tik = tokenize_with_tiktoken(text)
        hf = tokenize_with_hf(text)
    except Exception as exc:
        st.error(f"Tokenizer setup failed: {type(exc).__name__}: {exc}")
        st.caption("Install the pinned requirements and ensure Hugging Face files are available.")
        return

    left, right = st.columns(2)
    for column, name, result in (
        (left, "tiktoken · o200k_base", tik),
        (right, "Hugging Face · gpt2", hf),
    ):
        with column:
            st.subheader(name)
            st.metric("Token count", result.token_count)
            st.markdown(_colour_tokens(result.tokens), unsafe_allow_html=True)
            st.dataframe(_result_table(result), hide_index=True, use_container_width=True)

    st.subheader("Estimated input cost")
    estimates = estimate_costs_for_models(tik.token_count)
    if estimates:
        st.dataframe(
            [{"model": model, "estimated input cost (USD)": f"USD {cost:.8f}"}
             for model, cost in estimates.items()],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No model prices loaded. Add input prices to pricing.yaml; prices are not hard-coded here.")

    with st.expander("summary across preset examples"):
        if st.button("Build preset summary", key="tokenizer_preset_summary"):
            summary = []
            for preset_name, preset_text in presets.items():
                if not preset_text:
                    continue
                preset_tik = tokenize_with_tiktoken(preset_text)
                preset_hf = tokenize_with_hf(preset_text)
                preset_costs = estimate_costs_for_models(preset_tik.token_count)
                summary.append({
                    "example": preset_name,
                    "characters": len(preset_text),
                    "tiktoken tokens": preset_tik.token_count,
                    "GPT-2 tokens": preset_hf.token_count,
                    "Sonnet estimate (USD)": f"USD {preset_costs.get('claude-sonnet-5', 0.0):.8f}",
                    "Haiku estimate (USD)": f"USD {preset_costs.get(_ANTHROPIC_MODEL, 0.0):.8f}",
                })
            st.dataframe(summary, hide_index=True, use_container_width=True)

    with st.expander("Special tokens and tokenizer behaviour"):
        st.write("Special tokens are reserved markers such as BOS/EOS, padding, role markers, or <|endoftext|>.")
        st.write("This comparison encodes ordinary text without automatically adding BOS/EOS. Each tokenizer has its own vocabulary and reserved-token rules.")
        special_text = st.text_input("Try a special-token string", "<|endoftext|>")
        if special_text:
            try:
                encoding = _get_tiktoken_encoding()
                try:
                    tiktoken_special_ids = encoding.encode(
                        special_text, allowed_special={special_text}
                    )
                except (ValueError, KeyError):
                    tiktoken_special_ids = "Not registered by o200k_base"
                st.write({
                    "tiktoken literal-text IDs": encoding.encode(
                        special_text, disallowed_special=()
                    ),
                    "tiktoken special-aware IDs": tiktoken_special_ids,
                    "gpt2 IDs": _get_hf_tokenizer().encode(special_text, add_special_tokens=False),
                })
            except Exception as exc:
                st.caption(f"Could not inspect special-token text: {exc}")

    st.subheader("Check against the Anthropic API")
    st.caption("Claude's tokenizer is not public, so local counts are estimates. The API count is the comparison value for each model.")
    if st.button("Check API token count", type="secondary"):
        comparison_id = str(uuid4())
        rows = []
        for label, model in _ANTHROPIC_MODELS:
            started = time.perf_counter()
            try:
                api_count = _anthropic_count_tokens(text, model)
                latency_ms = (time.perf_counter() - started) * 1000
                from core.pricing import estimate_cost
                cost_usd = estimate_cost(model, api_count)
                rows.append({
                    "model": label,
                    "model ID": model,
                    "API tokens": api_count,
                    "tiktoken estimate": tik.token_count,
                    "difference": tik.token_count - api_count,
                    "latency (ms)": round(latency_ms, 2),
                    "estimated cost (USD)": f"USD {cost_usd:.8f}",
                })
                from core.logger import log_run
                log_run(
                    module="tokenizer",
                    provider="anthropic",
                    model=model,
                    prompt_id=comparison_id,
                    parameters={
                        "tiktoken_encoding": "o200k_base",
                        "hf_tokenizer": "gpt2",
                        "operation": "count_tokens",
                    },
                    input_tokens=api_count,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    output_text="",
                )
            except Exception as exc:
                st.warning(f"{label} API count unavailable: {exc}")
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.info("Differences are expected because Anthropic uses a different tokenizer and may account for message overhead.")
            st.caption(f"{len(rows)} API comparison(s) logged to runs.jsonl with shared prompt ID {comparison_id}.")
        else:
            st.caption("Confirm ANTHROPIC_API_KEY is loaded and that the installed SDK supports token counting.")

    with st.expander("GPT-2 BPE merge steps"):
        word = st.text_input("Word to replay", "tokenization")
        if word:
            steps = bpe_merge_steps(word)
            if steps:
                st.caption(f"{max(0, len(steps) - 1)} BPE merge(s) applied.")
                st.dataframe(steps, hide_index=True, use_container_width=True)
            else:
                st.info("GPT-2 merges.txt is not available in the local cache yet.")
