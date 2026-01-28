import streamlit as st
from pathlib import Path
import uuid

from core.pdf_loader import load_pdf
from core.chunker import smart_chunk
from core.embeddings import embed_texts
from core.vectorstore import create_faiss_index
from core.retriever import retrieve_context
from core.generator import generate_answer
from utils.confidence import calculate_confidence
from utils.citations import format_citations

# --------------------------------
# 🔥 TOKEN UTILS (ADDED)
# --------------------------------
def estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    if not text:
        return 0
    return max(1, len(text) // 4)

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="DocuKnow AI", page_icon="🧠", layout="wide")

# --------------------------------
# Session State
# --------------------------------
if "index_name" not in st.session_state:
    st.session_state.index_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:
    st.markdown("## 🧠 DocuKnow AI")
    st.markdown("##### Intelligent Document Assistant")
    st.divider()

    mode = st.radio("📂 Document Mode", ["Single PDF", "Multiple PDFs"])

    uploaded_files = None
    if mode == "Single PDF":
        uploaded_files = st.file_uploader(
            "Upload a PDF", type=["pdf"], accept_multiple_files=False
        )
        if uploaded_files:
            uploaded_files = [uploaded_files]
    else:
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )

    process_btn = st.button("🚀 Process Documents", use_container_width=True)
    clear_chat = st.button("🧹 Clear Chat", use_container_width=True)

# --------------------------------
# Header
# --------------------------------
st.markdown(
    """
    <div style="text-align:center">
        <h1>📄 DocuKnow AI</h1>
        <p>Ask intelligent questions from your documents with speed & accuracy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------
# Clear Chat
# --------------------------------
if clear_chat:
    st.session_state.chat_history = []
    st.session_state.index_name = None
    st.rerun()

# --------------------------------
# Process Documents
# --------------------------------
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            all_chunks = []

            for file in uploaded_files:
                temp_path = Path(f"data/uploads/{file.name}")
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                pages = load_pdf(str(temp_path))
                chunks = smart_chunk(pages)

                for c in chunks:
                    c["source"] = file.name

                all_chunks.extend(chunks)

            texts = [c["text"] for c in all_chunks]
            embeddings = embed_texts(texts)

            index_name = str(uuid.uuid4())
            create_faiss_index(
                embeddings=embeddings, metadatas=all_chunks, index_name=index_name
            )

            st.session_state.index_name = index_name

        st.success("Documents processed successfully! You can now ask questions.")

# --------------------------------
# Chat Interface
# --------------------------------
if st.session_state.index_name:
    st.markdown("### 💬 Ask Your Documents")

    query = st.text_input(
        "Type your question", placeholder="e.g. Explain deadlock in simple terms"
    )

    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            contexts = retrieve_context(
                query=query, index_name=st.session_state.index_name, top_k=4
            )

            # 🔥 TOKEN COUNT (INPUT)
            context_text = "\n".join(c["text"] for c in contexts)
            input_tokens = estimate_tokens(query + context_text)

            answer = generate_answer(query, contexts)

            # 🔥 TOKEN COUNT (OUTPUT)
            output_tokens = estimate_tokens(answer)

            confidence = calculate_confidence(contexts)
            citations = format_citations(contexts)

            st.session_state.chat_history.append(
                {
                    "question": query,
                    "answer": answer,
                    "confidence": confidence,
                    "citations": citations,
                    # 🔥 TOKEN DATA STORED PER MESSAGE
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                    }
                }
            )

    # --------------------------------
    # Display Chat History
    # --------------------------------
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {chat['question']}")
        st.markdown(f"**🤖 DocuKnow AI:** {chat['answer']}")

        conf = chat["confidence"]
        if conf["level"] == "High":
            st.success(f"🟢 Confidence: {conf['level']} ({conf['score']})")
        elif conf["level"] == "Medium":
            st.warning(f"🟡 Confidence: {conf['level']} ({conf['score']})")
        else:
            st.error(f"🔴 Confidence: {conf['level']} ({conf['score']})")

        with st.expander("📄 Sources"):
            for src in chat["citations"]:
                st.markdown(f"- {src}")

        # 🔥 TOKEN DISPLAY (PREMIUM FEATURE)
        tokens = chat["tokens"]
        st.caption(
            f"🧮 Tokens — Input: {tokens['input']} | "
            f"Output: {tokens['output']} | "
            f"Total: {tokens['total']}"
        )

        st.divider()

else:
    st.info("⬅ Upload and process documents to start asking questions.")
