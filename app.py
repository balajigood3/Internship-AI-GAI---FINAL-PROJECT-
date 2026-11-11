import os
import streamlit as st
from dotenv import load_dotenv
from grounded_assistant.core import (
    GroundedAssistant, OpenAIGenerator, SentenceTransformerEmbedder,
    VectorRetriever, load_text_documents,
)

load_dotenv()
st.set_page_config(page_title="Grounded Maintenance Assistant", page_icon="🛠️")
st.title("Grounded Industrial Maintenance Assistant")
st.caption("Answers only from your indexed documents, with guardrails and visible evidence.")

folder = st.text_input("Document folder", value="data")
if st.button("Index documents"):
    with st.spinner("Loading and embedding documents..."):
        chunks = load_text_documents(folder)
        retriever = VectorRetriever(SentenceTransformerEmbedder())
        retriever.index(chunks)
        st.session_state["retriever"] = retriever
    st.success(f"Indexed {len(chunks)} document chunks.")

if "retriever" in st.session_state:
    question = st.text_input("Ask a maintenance question")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Set OPENAI_API_KEY in your .env file before asking questions.")
        else:
            assistant = GroundedAssistant(st.session_state["retriever"], OpenAIGenerator())
            response = assistant.ask(question)
            st.subheader("Response")
            st.write(response.answer)
            st.caption(f"Status: {response.status}")
            if response.retrieved_passages:
                st.subheader("Retrieved evidence")
                for p in response.retrieved_passages:
                    with st.expander(f"{p.source} · page {p.page} · similarity {p.score:.3f}"):
                        st.write(p.text)
            st.subheader("Usage")
            st.write({
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost_usd": response.estimated_cost_usd,
            })
