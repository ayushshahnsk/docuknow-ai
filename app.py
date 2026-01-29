import streamlit as st
from pathlib import Path
import uuid
from utils.web_search import web_search

from core.pdf_loader import load_pdf
from core.chunker import smart_chunk
from core.embeddings import embed_texts
from core.vectorstore import create_faiss_index
from core.retriever import retrieve_context
from core.generator import generate_answer
from utils.confidence import calculate_confidence
from utils.citations import format_citations

from analytics.token_tracker import TokenTracker
from chat.chat_manager import ChatManager

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="DocuKnow AI", page_icon="🧠", layout="wide")

# --------------------------------
# Init Chat Manager (persistent)
# --------------------------------
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

chat_manager = st.session_state.chat_manager

# --------------------------------
# Init rename state (IMPORTANT)
# --------------------------------
if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = None

# --------------------------------
# Init token trackers (per chat)
# --------------------------------
if "token_trackers" not in st.session_state:
    st.session_state.token_trackers = {}

# --------------------------------
# Sidebar – ChatGPT style
# --------------------------------
with st.sidebar:
    st.markdown("## 🧠 DocuKnow AI")
    st.caption("History style Document Assistant")
    st.divider()

    st.markdown("### 💬 Chats")

    # ➕ New Chat
    if st.button("➕ New Chat", use_container_width=True):
        chat = chat_manager.create_chat("New Chat")
        st.session_state.token_trackers[chat.chat_id] = TokenTracker()
        st.rerun()

    # List chats
    for chat in chat_manager.list_chats():
        cid = chat["chat_id"]

        cols = st.columns([6, 1, 1])

        # Open chat
        if cols[0].button(chat["chat_name"], key=f"open_{cid}"):
            chat_manager.switch_chat(cid)
            st.rerun()

        # Rename (toggle rename mode)
        if cols[1].button("✏️", key=f"edit_{cid}"):
            st.session_state.renaming_chat = cid
            st.rerun()

        # Delete
        if cols[2].button("🗑️", key=f"del_{cid}"):
            chat_manager.delete_chat(cid)
            st.session_state.token_trackers.pop(cid, None)
            if st.session_state.renaming_chat == cid:
                st.session_state.renaming_chat = None
            st.rerun()

        # Rename input (persistent)
        if st.session_state.renaming_chat == cid:
            new_name = st.text_input(
                "Rename chat", chat["chat_name"], key=f"rename_input_{cid}"
            )
            if st.button("✅ Save", key=f"save_{cid}"):
                chat_manager.rename_chat(cid, new_name)
                st.session_state.renaming_chat = None
                st.rerun()

    st.divider()

    # Show active chat PDFs
    active_chat = chat_manager.get_active_chat()
    if active_chat and active_chat.pdf_names:
        st.markdown("### 📄 Active PDFs")
        for pdf in active_chat.pdf_names:
            st.caption(pdf)

    st.divider()

    # Upload section
    mode = st.radio("📂 Document Mode", ["Single PDF", "Multiple PDFs"])

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=(mode == "Multiple PDFs")
    )

    process_btn = st.button("🚀 Process Documents", use_container_width=True)
    clear_chat = st.button("🧹 Clear Messages", use_container_width=True)

    # --------------------------------
    # Normalize uploaded files (IMPORTANT)
    # --------------------------------
    if uploaded_files and not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

# --------------------------------
# Header
# --------------------------------
st.markdown(
    """
    <div style="text-align:center">
        <h1>📄 DocuKnow AI</h1>
        <p>Ask intelligent questions from your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------
# Active Chat
# --------------------------------
active_chat = chat_manager.get_active_chat()

if not active_chat:
    st.info("👈 Create or select a chat to begin")
    st.stop()

# Ensure token tracker exists
if active_chat.chat_id not in st.session_state.token_trackers:
    st.session_state.token_trackers[active_chat.chat_id] = TokenTracker()

tracker = st.session_state.token_trackers[active_chat.chat_id]

# --------------------------------
# Clear messages only (not PDFs)
# --------------------------------
if clear_chat:
    chat_manager.clear_chat_messages(active_chat.chat_id)
    tracker.reset()
    st.rerun()

# --------------------------------
# Process Documents (PER CHAT)
# --------------------------------
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            all_chunks = []
            pdf_names = []

            for file in uploaded_files:
                # Strict safety: only real UploadedFile
                if not hasattr(file, "getbuffer") or not hasattr(file, "name"):
                    st.warning("Invalid file detected. Please re-upload.")
                    continue

                filename = file.name
                pdf_names.append(filename)

                temp_path = Path(f"data/uploads/{active_chat.chat_id}_{filename}")
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                pages = load_pdf(str(temp_path))
                chunks = smart_chunk(pages)

                for c in chunks:
                    c["source"] = filename

                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No valid PDF content found.")
                st.stop()

            texts = [c["text"] for c in all_chunks]
            embeddings = embed_texts(texts)

            # FAISS index isolated per chat
            index_name = active_chat.chat_id
            create_faiss_index(
                embeddings=embeddings, metadatas=all_chunks, index_name=index_name
            )

            chat_manager.set_index_for_chat(active_chat.chat_id, index_name)
            chat_manager.set_pdfs_for_chat(active_chat.chat_id, pdf_names)

        st.success("Documents processed for this chat!")

# --------------------------------
# Chat Interface
# --------------------------------
if active_chat.index_name:
    st.markdown("### 💬 Conversation")

    # Display history
    for msg in active_chat.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 DocuKnow AI:** {msg['content']}")

    # 🔥 ChatGPT-style input
    query = st.chat_input("Message DocuKnow AI…")

    if query:
        chat_manager.add_user_message(query)

        with st.spinner("Thinking..."):
            # 🔹 STEP 1: Retrieve from PDF
            contexts = retrieve_context(
                query=query, index_name=active_chat.index_name, top_k=4
            )

            context_text = "\n".join(c["text"][:500] for c in contexts)
            input_tokens = tracker.count_input(query, context_text)

            pdf_answer = generate_answer(query, contexts)
            output_tokens = tracker.count_output(pdf_answer)

            confidence = calculate_confidence(contexts)
            citations = format_citations(contexts)

            # --------------------------------
            # 🔥 PDF FIRST → WEB SEARCH FALLBACK (ADDED)
            # --------------------------------
            if confidence["level"] == "Low":
                web_answer = web_search(query)

                if web_answer:
                    answer = "🌐 From internet (not found in PDF):\n\n" + web_answer
                else:
                    answer = (
                        "🌐 From internet (not found in PDF):\n\n"
                        "No reliable information found."
                    )
            else:
                answer = "📄 From document:\n\n" + pdf_answer

            chat_manager.add_assistant_message(answer)

        # --------------------------------
        # Display Answer
        # --------------------------------
        st.markdown(f"**🤖 DocuKnow AI:** {answer}")

        if confidence["level"] == "High":
            st.success(f"🟢 Confidence: {confidence['score']}")
        elif confidence["level"] == "Medium":
            st.warning(f"🟡 Confidence: {confidence['score']}")
        else:
            st.error(f"🔴 Confidence: {confidence['score']}")

        with st.expander("📄 Sources"):
            for src in citations:
                st.markdown(f"- {src}")

        st.caption(
            f"🧮 Tokens — Input: {input_tokens} | "
            f"Output: {output_tokens} | "
            f"Total: {input_tokens + output_tokens}"
        )

else:
    st.info("⬅ Upload and process documents to start asking questions.")
