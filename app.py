import streamlit as st
from pathlib import Path

from core.pdf_loader import load_pdf
from core.chunker import smart_chunk
from core.embeddings import embed_texts
from core.vectorstore import create_faiss_index
from core.retriever import retrieve_context
from core.generator import generate_answer

from utils.confidence import calculate_confidence
from utils.citations import format_citations
from utils.web_search import web_search

from analytics.token_tracker import TokenTracker
from chat.chat_manager import ChatManager

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="DocuKnow AI", page_icon="🧠", layout="wide")

# ======================================================
# SESSION STATE INIT
# ======================================================
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

if "token_trackers" not in st.session_state:
    st.session_state.token_trackers = {}

if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = None

# ---- Settings / PDF Manager state ----
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

if "settings_tab" not in st.session_state:
    st.session_state.settings_tab = "pdf"

if "settings_uploaded_pdfs" not in st.session_state:
    st.session_state.settings_uploaded_pdfs = []

chat_manager = st.session_state.chat_manager

# ======================================================
# SIDEBAR (CHAT LIST + ACTIVE PDF + SETTINGS)
# ======================================================
with st.sidebar:
    st.markdown("## 🧠 DocuKnow AI")
    st.caption("History-style Document Assistant")
    st.divider()

    # ---------------- Chats ----------------
    st.markdown("### 💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        chat = chat_manager.create_chat("New Chat")
        st.session_state.token_trackers[chat.chat_id] = TokenTracker()
        st.rerun()

    for chat in chat_manager.list_chats():
        cid = chat["chat_id"]
        cols = st.columns([6, 1, 1])

        if cols[0].button(chat["chat_name"], key=f"open_{cid}"):
            chat_manager.switch_chat(cid)
            st.rerun()

        if cols[1].button("✏️", key=f"edit_{cid}"):
            st.session_state.renaming_chat = cid
            st.rerun()

        if cols[2].button("🗑️", key=f"del_{cid}"):
            chat_manager.delete_chat(cid)
            st.session_state.token_trackers.pop(cid, None)
            st.session_state.renaming_chat = None
            st.rerun()

        if st.session_state.renaming_chat == cid:
            new_name = st.text_input(
                "Rename chat", chat["chat_name"], key=f"rename_{cid}"
            )
            if st.button("✅ Save", key=f"save_{cid}"):
                chat_manager.rename_chat(cid, new_name)
                st.session_state.renaming_chat = None
                st.rerun()

    st.divider()

    # ---------------- Active PDFs ----------------
    active_chat = chat_manager.get_active_chat()
    st.markdown("### 📄 Active PDFs")

    if active_chat and active_chat.pdf_names:
        for pdf in active_chat.pdf_names:
            st.caption(pdf)
    else:
        st.caption("No active PDF currently")

    st.divider()

    # ---------------- Upload Disabled ----------------
    st.markdown("### 📂 Document Upload")
    st.info("PDF upload moved to ⚙️ Settings → PDF Manager")
    st.button("🚫 Upload Disabled", disabled=True, use_container_width=True)

    st.divider()

    # ---------------- Settings ----------------
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()

# ======================================================
# HEADER
# ======================================================
st.markdown(
    """
    <div style="text-align:center">
        <h1>📄 DocuKnow AI</h1>
        <p>Ask intelligent questions from your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# SETTINGS PANEL
# ======================================================
if st.session_state.show_settings:
    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    left, right = st.columns([1, 3])

    # -------- Left menu --------
    with left:
        if st.button("📂 PDF Manager", use_container_width=True):
            st.session_state.settings_tab = "pdf"

        if st.button("🧠 Model / API", use_container_width=True):
            st.session_state.settings_tab = "model"

        st.divider()

        if st.button("❌ Close Settings", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()

    # -------- Right panel --------
    with right:
        # ================= PDF MANAGER =================
        if st.session_state.settings_tab == "pdf":
            st.markdown("### 📂 PDF Manager")
            st.caption("Upload and activate PDFs for this chat")

            uploaded = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
            )

            if uploaded:
                st.session_state.settings_uploaded_pdfs = uploaded

            st.divider()
            st.markdown("#### 📄 Uploaded PDFs")

            selected_indices = []

            if not st.session_state.settings_uploaded_pdfs:
                st.info("No PDFs uploaded yet.")
            else:
                for i, pdf in enumerate(st.session_state.settings_uploaded_pdfs):
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        checked = st.checkbox(
                            "",
                            value=True,
                            key=f"pdf_active_{i}",
                        )
                        if checked:
                            selected_indices.append(i)

                    with c2:
                        st.markdown(f"📄 **{pdf.name}**")

            # -------- SAVE & PROCESS PDFs --------
            if st.button("💾 Save & Process PDFs", use_container_width=True):
                active_chat = chat_manager.get_active_chat()

                selected_files = [
                    st.session_state.settings_uploaded_pdfs[i]
                    for i in selected_indices
                ]

                if not selected_files:
                    st.error("Please select at least one PDF")
                    st.stop()

                all_chunks = []
                pdf_names = []

                with st.spinner("Processing PDFs..."):
                    for file in selected_files:
                        pdf_names.append(file.name)

                        temp_path = Path(
                            f"data/uploads/{active_chat.chat_id}_{file.name}"
                        )
                        temp_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())

                        pages = load_pdf(str(temp_path))
                        chunks = smart_chunk(pages)

                        for c in chunks:
                            c["source"] = file.name

                        all_chunks.extend(chunks)

                    embeddings = embed_texts([c["text"] for c in all_chunks])
                    index_name = active_chat.chat_id

                    create_faiss_index(
                        embeddings=embeddings,
                        metadatas=all_chunks,
                        index_name=index_name,
                    )

                    chat_manager.set_index_for_chat(active_chat.chat_id, index_name)
                    chat_manager.set_pdfs_for_chat(active_chat.chat_id, pdf_names)

                st.success("PDFs processed and activated!")
                st.session_state.show_settings = False
                st.rerun()

        # ================= MODEL / API =================
        elif st.session_state.settings_tab == "model":
            st.markdown("### 🧠 Model / API Settings")
            st.caption("Custom API will safely fallback to Ollama for now")

            active_chat = chat_manager.get_active_chat()
            pref = active_chat.model_pref

            choice = st.radio(
                "Select model",
                ["Ollama (Local)", "Use my API Key"],
                index=0 if pref["type"] == "ollama" else 1,
            )

            api_key_input = None
            if choice == "Use my API Key":
                api_key_input = st.text_input(
                    "Enter your API key",
                    type="password",
                    value=pref.get("api_key") or "",
                )

            if st.button("💾 Save Model Settings", use_container_width=True):
                if choice == "Ollama (Local)":
                    chat_manager.set_model_pref(active_chat.chat_id, "ollama")
                else:
                    chat_manager.set_model_pref(
                        active_chat.chat_id, "api", api_key_input
                    )

                st.success("Model preference saved")
                st.session_state.show_settings = False
                st.rerun()

    st.markdown("---")

# ======================================================
# CHAT AREA
# ======================================================
active_chat = chat_manager.get_active_chat()

if not active_chat:
    st.info("👈 Create or select a chat to begin")
    st.stop()

if active_chat.chat_id not in st.session_state.token_trackers:
    st.session_state.token_trackers[active_chat.chat_id] = TokenTracker()

tracker = st.session_state.token_trackers[active_chat.chat_id]

if active_chat.index_name:
    st.markdown("### 💬 Conversation")

    for msg in active_chat.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='font-size:16px; margin-bottom:6px;'><b>🧑 You:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:16px; line-height:1.6; margin-bottom:12px;'><b>🤖 DocuKnow AI:</b><br>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    query = st.chat_input("Message DocuKnow AI…")

    if query:
        chat_manager.add_user_message(query)

        with st.spinner("Thinking..."):
            contexts = retrieve_context(query, active_chat.index_name)
            context_text = "\n".join(c["text"][:500] for c in contexts)
            tracker.count_input(query, context_text)

            pdf_answer = generate_answer(query, contexts)

            confidence = calculate_confidence(contexts)

            if confidence["level"] == "Low":
                api_key = None
                if active_chat.model_pref["type"] == "api":
                    api_key = active_chat.model_pref.get("api_key")
                raw_web = web_search(query, api_key=api_key)
                answer = raw_web[:800].strip() if raw_web else "No reliable information found."
                source_type = "internet"
            else:
                answer = pdf_answer
                source_type = "pdf"

            tracker.count_output(answer)
            chat_manager.add_assistant_message(answer)
            st.rerun()

        if source_type == "pdf":
            st.divider()
            st.success(f"Confidence Score: {confidence['score']}")
            citations = format_citations(contexts)
            with st.expander("📄 Sources"):
                for src in citations:
                    st.markdown(f"- {src}")

else:
    st.info("⬅ Upload PDFs from Settings to start chatting.")
