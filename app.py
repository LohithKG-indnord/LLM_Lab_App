import os

import streamlit as st
from dotenv import load_dotenv

from modules import attention, tokenizer_lens

load_dotenv()

st.set_page_config(page_title="LLM Lab", layout="wide")
st.title("LLM Lab App")
st.caption("Tokenizer Lens compares local tokenizers and checks Anthropic's token count.")

tab1, tab2, tab3, tab4 = st.tabs(["Tokenizer Lens", "Attention Explorer", "Sampling Playground", "Model Arena"])

with tab1:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning(
            "ANTHROPIC_API_KEY is not loaded. Local tokenization still works, "
            "but the Anthropic API count check will remain unavailable."
        )
    try:
        tokenizer_lens.render()
    except Exception as exc:
        st.error(f"Tokenizer Lens could not start")
        st.info("Install the requirements, then restart Streamlit.")
with tab2:
    try:
        attention.render()
    except Exception as exc:
        st.error("Attention Explorer could not start")
        st.exception(exc)
with tab3:
    st.write("Sampling Playground - Coming soon")
with tab4:
    st.write("Model Arena - Coming soon")
