
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.rag import answer_question


st.set_page_config(
    page_title="MCP RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("MCP RAG Assistant")

st.write(
    "Ask questions about the MCP Python SDK documentation."
)

question = st.text_input(
    "Your question",
    placeholder="How do I create an MCP tool in Python?",
)

if st.button("Ask", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner(
            "Searching the MCP documentation..."
        ):

            try:
                result = answer_question(
                    question.strip()
                )

            except Exception as e:
                st.error(f"Error: {e}")

            else:

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Latency",
                        f"{result['latency_ms']:.0f} ms",
                    )

                with col2:
                    st.metric(
                        "Retrieved documents",
                        len(
                            result["retrieval"]["results"]
                        ),
                    )

                st.subheader("Sources")

                for doc in result["retrieval"]["results"]:

                    title = doc["title"]
                    url = doc["url"]

                    if url:
                        st.markdown(
                            f"**{doc['rank']}. {title}**  \n"
                            f"{url}"
                        )
                    else:
                        st.write(
                            f"{doc['rank']}. {title}"
                        )