import uuid
from dotenv import load_dotenv
import streamlit as st

from app.config import Settings
from app.rag.pipeline import IngestionPipeline, QueryPipeline


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = []


def sidebar_uploader(ingestion: IngestionPipeline):
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.sidebar.button("Ingest Selected PDFs"):
        for f in uploaded_files:
            doc_id = str(uuid.uuid4())
            with st.spinner(f"Uploading and ingesting {f.name}..."):
                source_uri = ingestion.upload_to_cos(doc_id, f.name, f)
                count = ingestion.ingest_pdf(doc_id, f.name, source_uri)
            st.session_state["ingested_docs"].append((doc_id, f.name, source_uri, count))
        st.sidebar.success("Ingestion complete.")

    if st.session_state["ingested_docs"]:
        st.sidebar.subheader("Ingested Documents")
        for doc_id, name, uri, count in st.session_state["ingested_docs"]:
            st.sidebar.write(f"- {name} ({count} chunks)")


def chat_ui(query_pipeline: QueryPipeline):
    st.title("IBM Cloud RAG MVP")
    st.caption("Granite-13B-Instruct + Milvus + COS + Streamlit")

    for role, content in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask a question about your PDFs…")
    if user_input:
        st.session_state["messages"].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, sources = query_pipeline.answer(user_input)
                citations = "\n\n" + "\n".join([f"- {s}" for s in sources]) if sources else ""
                st.markdown(answer + citations)
                st.session_state["messages"].append(("assistant", answer + citations))


def main():
    # Load local .env for development
    load_dotenv()
    settings = Settings.from_env()
    init_state()

    ingestion = IngestionPipeline(settings)
    query_pipeline = QueryPipeline(settings)

    sidebar_uploader(ingestion)
    chat_ui(query_pipeline)


if __name__ == "__main__":
    main()


