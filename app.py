"""Simple page for asking the local document assistant one question."""

import streamlit as st

import curator


SAMPLE_QUESTIONS = [
    "What is the fixed fee in the Harborline Markets engagement letter?",
    "When is the Cedar & Pine inventory observation report due?",
    "How long does Northwind keep final engagement files?",
    "What late charge applies to unpaid invoices?",
    "Who must be told about a suspected confidentiality breach, and how soon?",
    "What was Harborline Markets' revenue in 2025?",
]


st.set_page_config(page_title="Document assistant demo", layout="centered")
st.title("Document assistant")
st.write(
    "Ask one question about the fictional Northwind Advisory files. "
    "The answer cites the file name and page it came from. "
    "If those files do not say it, the answer is “I don't know.”"
)

with st.sidebar:
    st.header("Files")
    docs_dir = st.text_input("PDF folder", curator.DEFAULT_DOCS_DIR)
    reload_documents = st.button("Reload documents")
    st.caption(
        "The sample files are invented for this demo. "
        "Qwen and the embedding model run locally through Ollama. "
        "No paid API and no API key are used."
    )


@st.cache_resource(show_spinner="Indexing the PDF folder on this computer...")
def cached_index(folder):
    return curator.build_index(folder)


if reload_documents:
    cached_index.clear()
    st.rerun()

try:
    index = cached_index(docs_dir)
except (
    curator.LocalModelError,
    FileNotFoundError,
    RuntimeError,
    OSError,
) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.write(f"{len(index.files)} PDF file(s) indexed:")
    for name in index.files:
        st.write(f"- {name}")

if "question" not in st.session_state:
    st.session_state.question = SAMPLE_QUESTIONS[0]


def use_sample():
    st.session_state.question = st.session_state.sample_choice


st.selectbox(
    "Sample questions",
    SAMPLE_QUESTIONS,
    key="sample_choice",
    on_change=use_sample,
)
st.text_area("Your question", key="question", height=100)

if st.button("Ask", type="primary"):
    try:
        with st.spinner("Asking the local Qwen model..."):
            result = curator.answer_question(st.session_state.question, index)
    except (curator.LocalModelError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
    else:
        st.subheader("Answer")
        if result["citations_valid"] and result["citations"]:
            st.write(result["delivered_answer"])
            st.subheader("Where this came from")
            for citation in result["citations"]:
                st.markdown(
                    f"**{citation['filename']}, page {citation['page_number']}**"
                )
                st.write(citation["quote"])
        elif result["citations_valid"]:
            st.info(result["delivered_answer"])
            st.caption(
                "Nothing was cited. The assistant says it does not know when "
                "the files do not support an answer."
            )
        else:
            st.warning(result["delivered_answer"])
            for problem in result["citation_problems"]:
                st.caption(problem)
        st.caption(
            f"Local model: {result['model']}. "
            f"Answer time: {result['generation_seconds']:.1f}s."
        )
