"""
DocuKnow AI - Version 1.0 (Multilingual Edition)

A RAG-based document Q&A system with:
- Multilingual support (20+ languages)
- ChatGPT-style chat interface
- PDF processing with OCR
- Text-to-Speech output
- Free web search integration
- Advanced debugging tools
"""

import streamlit as st
from pathlib import Path
import logging
import time
import json
import base64
from typing import List, Dict, Optional, Tuple

# Import core modules
from core.pdf_loader import load_pdf
from core.chunker import smart_chunk
from core.embeddings import embed_texts
from core.vectorstore import create_faiss_index, load_faiss_index
from core.retriever import retrieve_context
from core.generator import generate_answer, generate_answer_with_fallback
from core.ocr_processor import OCRProcessor, EASYOCR_AVAILABLE
from core.pdf_detector import detect_pdf_type
from core.tts_processor import TTSProcessor

# Import utilities
from utils.confidence import calculate_confidence, calculate_confidence_detailed, diagnose_confidence_issue
from utils.citations import format_citations
from utils.web_search import web_search, advanced_web_search, get_search_stats
from utils.language_detector import detect_query_language, get_language_name, get_supported_languages
from utils.multilingual_search import MultilingualSearch
from utils.translator import get_translator, translate_text, detect_language

# Import analytics and chat management
from analytics.token_tracker import TokenTracker
from chat.chat_manager import ChatManager, ChatSession
from chat.chat_store import save_chat, load_all_chats, delete_chat

# Import config
from config import settings
from config.prompts import build_rag_prompt, build_followup_prompt

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="DocuKnow AI - Multilingual",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOGGING SETUP
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================================================
# SESSION STATE INITIALIZATION
# ======================================================

# Chat management
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

if "token_trackers" not in st.session_state:
    st.session_state.token_trackers = {}

if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = None

# Settings state
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

if "settings_tab" not in st.session_state:
    st.session_state.settings_tab = "pdf"

if "settings_uploaded_pdfs" not in st.session_state:
    st.session_state.settings_uploaded_pdfs = []

# Context storage for citations
if "message_contexts" not in st.session_state:
    st.session_state.message_contexts = {}

# OCR processing state
if "ocr_test_file" not in st.session_state:
    st.session_state.ocr_test_file = None

if "ocr_test_result" not in st.session_state:
    st.session_state.ocr_test_result = None

if "ocr_languages" not in st.session_state:
    st.session_state.ocr_languages = ["en"]

if "ocr_use_gpu" not in st.session_state:
    st.session_state.ocr_use_gpu = False

# TTS State
if "tts_processor" not in st.session_state:
    st.session_state.tts_processor = TTSProcessor()

if "current_playing_audio" not in st.session_state:
    st.session_state.current_playing_audio = None

if "current_playing_index" not in st.session_state:
    st.session_state.current_playing_index = None

if "tts_paused" not in st.session_state:
    st.session_state.tts_paused = False

if "tts_language" not in st.session_state:
    st.session_state.tts_language = "en"

# Multilingual settings
if "user_language" not in st.session_state:
    st.session_state.user_language = "auto"  # "auto" or specific language code

if "detected_languages" not in st.session_state:
    st.session_state.detected_languages = {}

# ↓ ADD THIS RIGHT AFTER ABOVE LINE ↓
if "pdf_languages" not in st.session_state:
    st.session_state.pdf_languages = {}

# Debug state for source selection
if "show_source_debug" not in st.session_state:
    st.session_state.show_source_debug = False

if "last_fallback_reason" not in st.session_state:
    st.session_state.last_fallback_reason = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

if "last_context_count" not in st.session_state:
    st.session_state.last_context_count = 0

if "force_pdf_only" not in st.session_state:
    st.session_state.force_pdf_only = False

if "low_confidence_threshold" not in st.session_state:
    st.session_state.low_confidence_threshold = 0.3

# Search settings
if "search_provider" not in st.session_state:
    st.session_state.search_provider = "multilingual"  # "multilingual", "tavily", "duckduckgo"

if "search_api_key" not in st.session_state:
    st.session_state.search_api_key = ""

# Language model settings
if "llm_model_override" not in st.session_state:
    st.session_state.llm_model_override = None

# ======================================================
# TTS HELPER FUNCTIONS
# ======================================================

def play_tts(text: str, message_index: int, lang: str = None):
    """Generate and play TTS audio for given text."""
    try:
        audio_base64 = st.session_state.tts_processor.text_to_audio(
            text, 
            lang or st.session_state.tts_language
        )
        
        if audio_base64:
            st.session_state.current_playing_audio = audio_base64
            st.session_state.current_playing_index = message_index
            st.session_state.tts_paused = False
        else:
            st.error("Failed to generate audio")
            
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")

def pause_tts():
    """Pause current TTS playback."""
    st.session_state.tts_paused = True

def resume_tts():
    """Resume paused TTS playback."""
    st.session_state.tts_paused = False

def stop_tts():
    """Stop TTS playback."""
    st.session_state.current_playing_audio = None
    st.session_state.current_playing_index = None
    st.session_state.tts_paused = False

def is_currently_playing(message_index: int) -> bool:
    """Check if a specific message is currently playing."""
    return (
        st.session_state.current_playing_index == message_index and
        st.session_state.current_playing_audio is not None and
        not st.session_state.tts_paused
    )

def cleanup_tts_on_chat_switch():
    """Stop TTS when switching chats."""
    if st.session_state.current_playing_audio:
        stop_tts()
        st.info("Audio stopped due to chat switch")

# ======================================================
# LANGUAGE HELPER FUNCTIONS
# ======================================================

def detect_and_set_language(query: str) -> str:
    """
    Detect language of query and set user language preference.
    
    Args:
        query: User query
        
    Returns:
        Detected language code
    """
    if st.session_state.user_language != "auto":
        return st.session_state.user_language
    
    try:
        lang_info = detect_query_language(query)
        detected_lang = lang_info.get("language", "en")
        
        # Store detection history
        if detected_lang not in st.session_state.detected_languages:
            st.session_state.detected_languages[detected_lang] = 0
        st.session_state.detected_languages[detected_lang] += 1
        
        # If we've detected this language multiple times, set it as preference
        if st.session_state.detected_languages.get(detected_lang, 0) >= 3:
            st.session_state.user_language = detected_lang
            st.toast(f"Language preference set to {get_language_name(detected_lang)}", icon="🌍")
        
        return detected_lang
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "en"

def get_response_language(query_lang: str, pdf_lang: str = None) -> str:
    """
    Determine which language to use for response.
    
    Args:
        query_lang: Language of the query
        pdf_lang: Language of PDF content (if known)
        
    Returns:
        Language code for response
    """
    # If user has set a specific language, use it
    if st.session_state.user_language != "auto":
        return st.session_state.user_language
    
    # If PDF language is known and matches query, use it
    if pdf_lang and pdf_lang == query_lang:
        return query_lang
    
    # Otherwise use query language
    return query_lang

# ======================================================
# SEARCH HELPER FUNCTIONS
# ======================================================

def perform_web_search(query: str, lang: str) -> Optional[str]:
    """
    Perform web search with appropriate provider and language.
    
    Args:
        query: Search query
        lang: Language for search results
        
    Returns:
        Search results or None
    """
    try:
        api_key = None
        if st.session_state.search_provider == "tavily" and st.session_state.search_api_key:
            api_key = st.session_state.search_api_key
        
        result = web_search(
            query=query,
            api_key=api_key,
            lang=lang,
            provider=st.session_state.search_provider
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return None

# ======================================================
# SIDEBAR (CHAT LIST + ACTIVE PDF + SETTINGS)
# ======================================================
with st.sidebar:
    st.markdown("## 🌍 DocuKnow AI")
    st.caption("Multilingual Document Assistant")
    st.divider()

    # ---------------- Language Selector ----------------
    st.markdown("### 🌐 Language")
    
    # Get supported languages
    supported_langs = get_supported_languages()
    lang_options = ["auto"] + list(supported_langs.keys())
    lang_names = ["Auto-detect"] + [supported_langs[code] for code in supported_langs]
    
    selected_lang = st.selectbox(
        "Response Language",
        options=lang_options,
        format_func=lambda x: "Auto-detect" if x == "auto" else f"{supported_langs.get(x, x)} ({x})",
        index=lang_options.index(st.session_state.user_language) if st.session_state.user_language in lang_options else 0,
        help="Choose language for responses. Auto-detect will use the language of your query."
    )
    
    if selected_lang != st.session_state.user_language:
        st.session_state.user_language = selected_lang
        if selected_lang != "auto":
            st.toast(f"Language set to {supported_langs.get(selected_lang, selected_lang)}", icon="✅")
    
    st.divider()
    
    # ---------------- Chats ----------------
    st.markdown("### 💬 Chats")
    
    chat_manager = st.session_state.chat_manager
    
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"):
        chat = chat_manager.create_chat("New Chat")
        st.session_state.token_trackers[chat.chat_id] = TokenTracker()
        cleanup_tts_on_chat_switch()
        st.rerun()
    
    for chat in chat_manager.list_chats():
        cid = chat["chat_id"]
        cols = st.columns([6, 1, 1])
        
        if cols[0].button(chat["chat_name"], key=f"open_{cid}", use_container_width=True):
            cleanup_tts_on_chat_switch()
            chat_manager.switch_chat(cid)
            st.rerun()
        
        if cols[1].button("✏️", key=f"edit_{cid}", help="Rename chat"):
            st.session_state.renaming_chat = cid
            st.rerun()
        
        if cols[2].button("🗑️", key=f"del_{cid}", help="Delete chat"):
            chat_manager.delete_chat(cid)
            st.session_state.token_trackers.pop(cid, None)
            st.session_state.renaming_chat = None
            cleanup_tts_on_chat_switch()
            st.rerun()
        
        if st.session_state.renaming_chat == cid:
            new_name = st.text_input(
                "Rename chat", 
                chat["chat_name"], 
                key=f"rename_{cid}"
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
            st.caption(f"📄 {pdf}")
        
        # Show language distribution if available
        if active_chat.chat_id in st.session_state.get("pdf_languages", {}):
            pdf_langs = st.session_state.pdf_languages[active_chat.chat_id]
            lang_summary = ", ".join([f"{get_language_name(lang)}" for lang in pdf_langs[:3]])
            if len(pdf_langs) > 3:
                lang_summary += f" +{len(pdf_langs)-3} more"
            # st.caption(f"🌐 Languages: {lang_summary}")
    else:
        st.caption("No active PDFs")
    
    st.divider()
    
    # ---------------- Upload Disabled ----------------
    st.markdown("### 📂 Document Upload")
    st.info("PDF upload moved to ⚙️ Settings → PDF Manager")
    st.button("🚫 Upload Disabled", disabled=True, use_container_width=True)
    
    st.divider()
    
    # ---------------- Settings ----------------
    if st.button("⚙️ Settings", use_container_width=True, key="settings_sidebar"):
        st.session_state.show_settings = True
        st.rerun()
    
    # ---------------- Stats ----------------
    with st.expander("📊 Quick Stats", expanded=False):
        if active_chat:
            tracker = st.session_state.token_trackers.get(active_chat.chat_id)
            if tracker:
                totals = tracker.get_totals()
                st.metric("📝 Input Tokens", f"{totals['input']:,}")
                st.metric("💬 Output Tokens", f"{totals['output']:,}")
                st.metric("🎯 Total Tokens", f"{totals['total']:,}")
        
        # Language stats
        if st.session_state.detected_languages:
            st.markdown("**🌍 Detected Languages:**")
            for lang, count in list(st.session_state.detected_languages.items())[:5]:
                lang_name = get_language_name(lang)
                st.caption(f"{lang_name}: {count} queries")

# ======================================================
# HEADER
# ======================================================
st.markdown(
    """
    <div style="text-align:center">
        <h1>🌍 DocuKnow AI - Multilingual</h1>
        <p>Ask questions in any language. Get answers from your documents or the web.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Language indicator
if st.session_state.user_language != "auto":
    lang_name = get_language_name(st.session_state.user_language)
    st.info(f"**Response Language:** {lang_name} ({st.session_state.user_language})")
else:
    st.info("**Response Language:** Auto-detected from your query")

# ======================================================
# SETTINGS PANEL
# ======================================================
if st.session_state.show_settings:
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    
    left, right = st.columns([1, 3])
    
    with left:
        # Navigation buttons for settings tabs
        tabs = {
            "📂 PDF Manager": "pdf",
            "🧠 Model / API": "model",
            "🌐 Web Search": "web_search",
            "🔍 OCR Techniques": "ocr_tech",
            "🔊 TTS Settings": "tts_settings",
            "🌍 Language Settings": "language_settings",
            "🐛 Debug Settings": "debug_settings"
        }
        
        for tab_name, tab_key in tabs.items():
            if st.button(tab_name, use_container_width=True, key=f"tab_{tab_key}"):
                st.session_state.settings_tab = tab_key
        
        st.divider()
        
        if st.button("❌ Close Settings", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()
    
    with right:
        # ----------------- PDF Manager Tab -----------------
        if st.session_state.settings_tab == "pdf":
            st.markdown("### 📂 PDF Manager")
            st.caption("Upload and process PDFs in multiple languages")
            
            uploaded = st.file_uploader(
                "Upload PDFs (Multiple languages supported)",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader_settings"
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
                        checked = st.checkbox("", value=True, key=f"pdf_active_{i}")
                        if checked:
                            selected_indices.append(i)
                    with c2:
                        st.markdown(f"📄 **{pdf.name}**")
                        
                        # Detect PDF language
                        if st.button("🌍 Detect Language", key=f"detect_lang_{i}"):
                            with st.spinner("Detecting language..."):
                                try:
                                    # Save temp file
                                    temp_path = Path(f"data/temp/{pdf.name}")
                                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    with open(temp_path, "wb") as f:
                                        f.write(pdf.getbuffer())
                                    
                                    # Load first page to detect language
                                    import fitz
                                    doc = fitz.open(temp_path)
                                    if len(doc) > 0:
                                        text = doc[0].get_text().strip()[:1000]
                                        if text:
                                            lang_info = detect_query_language(text)
                                            lang_name = get_language_name(lang_info["language"])
                                            st.success(f"Detected: {lang_name} (confidence: {lang_info.get('confidence', 0):.2f})")
                                    doc.close()
                                    
                                except Exception as e:
                                    st.error(f"Language detection failed: {e}")
            
            if st.button("💾 Save & Process PDFs", use_container_width=True, key="process_pdfs"):
                active_chat = chat_manager.get_active_chat()
                
                if not active_chat:
                    st.error("Please create or select a chat first")
                    st.stop()
                
                selected_files = [
                    st.session_state.settings_uploaded_pdfs[i]
                    for i in selected_indices
                ]
                
                if not selected_files:
                    st.error("Please select at least one PDF")
                    st.stop()
                
                all_chunks = []
                pdf_names = []
                processing_info = []
                pdf_languages = []
                
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                for file_idx, file in enumerate(selected_files):
                    pdf_names.append(file.name)
                    progress = (file_idx) / len(selected_files)
                    progress_bar.progress(progress)
                    
                    status_container.info(f"Processing {file.name} ({file_idx + 1}/{len(selected_files)})...")
                    
                    temp_path = Path(f"data/uploads/{active_chat.chat_id}_{file.name}")
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process PDF
                    pages = load_pdf(str(temp_path))
                    
                    if pages:
                        processing_method = pages[0].get("processed_with", "unknown")
                        processing_info.append({
                            "filename": file.name,
                            "method": processing_method,
                            "pages": len(pages)
                        })
                        
                        # Detect language from first few pages
                        sample_text = ""
                        for page in pages[:3]:
                            sample_text += page.get("text", "")[:500]
                        
                        if sample_text:
                            lang_info = detect_query_language(sample_text)
                            detected_lang = lang_info.get("language", "en")
                            pdf_languages.append(detected_lang)
                        
                        chunks = smart_chunk(pages)
                        
                        for c in chunks:
                            c["source"] = file.name
                            c["language"] = detected_lang if sample_text else "en"
                        
                        all_chunks.extend(chunks)
                
                # Create embeddings and index
                if all_chunks:
                    progress_bar.progress(0.9)
                    status_container.info("Creating embeddings and vector index...")
                    
                    try:
                        embeddings = embed_texts([c["text"] for c in all_chunks])
                        index_name = active_chat.chat_id
                        
                        create_faiss_index(
                            embeddings=embeddings,
                            metadatas=all_chunks,
                            index_name=index_name,
                        )
                        
                        chat_manager.set_index_for_chat(active_chat.chat_id, index_name)
                        chat_manager.set_pdfs_for_chat(active_chat.chat_id, pdf_names)
                        
                        # Store PDF languages
                        if active_chat.chat_id not in st.session_state.pdf_languages:
                            st.session_state.pdf_languages = {}
                        st.session_state.pdf_languages[active_chat.chat_id] = list(set(pdf_languages))
                        
                        progress_bar.progress(1.0)
                        status_container.success("PDFs processed successfully!")
                        
                        # Show processing details
                        with st.expander("📊 Processing Details", expanded=True):
                            for info in processing_info:
                                method_icon = "🔤" if info["method"] == "text_extraction" else "🖼️"
                                method_text = "Text Extraction" if info["method"] == "text_extraction" else "OCR Processing"
                                st.write(f"{method_icon} **{info['filename']}**: {method_text} ({info['pages']} pages)")
                        
                        # Show language distribution
                        if pdf_languages:
                            lang_counts = {}
                            for lang in pdf_languages:
                                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                            
                            st.info("**📚 Document Language Distribution:**")
                            for lang, count in lang_counts.items():
                                lang_name = get_language_name(lang)
                                st.write(f"- {lang_name}: {count} document(s)")
                        
                        # Clear settings and rerun
                        time.sleep(2)
                        st.session_state.show_settings = False
                        st.session_state.settings_uploaded_pdfs = []
                        st.rerun()
                        
                    except Exception as e:
                        status_container.error(f"Error creating index: {e}")
                else:
                    st.error("No text could be extracted from the PDFs")
        
        # ----------------- Model/API Tab -----------------
        elif st.session_state.settings_tab == "model":
            st.markdown("### 🧠 Model & API Settings")
            
            active_chat = chat_manager.get_active_chat()
            pref = active_chat.model_pref if active_chat else {"type": "ollama", "api_key": None}
            
            st.markdown("#### LLM Configuration")
            choice = st.radio(
                "Select LLM Provider",
                ["Ollama (Local)", "Use Custom API"],
                index=0 if pref["type"] == "ollama" else 1,
                key="llm_provider"
            )
            
            api_key_input = None
            if choice == "Use Custom API":
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    value=pref.get("api_key") or "",
                    help="Enter your API key for custom LLM service"
                )
                
                model_name = st.text_input(
                    "Model Name",
                    value=st.session_state.llm_model_override or settings.LLM_MODEL_NAME,
                    help="Override default model name"
                )
                
                if model_name != settings.LLM_MODEL_NAME:
                    st.session_state.llm_model_override = model_name
            
            st.divider()
            
            st.markdown("#### 🌐 Language Model Selection")
            
            # Show current language model mapping
            st.info("**Current Language-Model Mapping:**")
            for lang, model in settings.LANGUAGE_MODELS.items():
                if lang != "default":
                    lang_name = get_language_name(lang)
                    st.caption(f"{lang_name} ({lang}): {model}")
            
            st.divider()
            
            st.markdown("#### 📄 PDF Answer Behavior")
            
            force_pdf_only = st.checkbox(
                "Force PDF answers only (disable web search)",
                value=st.session_state.force_pdf_only,
                help="Always use PDF content even with low confidence. Useful when PDF has answer but system shows internet answer."
            )
            st.session_state.force_pdf_only = force_pdf_only
            
            if force_pdf_only:
                st.success("✅ Web search disabled. All answers will come from your PDFs only.")
            
            if st.button("💾 Save Model Settings", use_container_width=True, key="save_model"):
                if active_chat:
                    if choice == "Ollama (Local)":
                        chat_manager.set_model_pref(active_chat.chat_id, "ollama")
                    else:
                        chat_manager.set_model_pref(
                            active_chat.chat_id, "api", api_key_input
                        )
                    
                    st.success("Model preference saved")
                    time.sleep(1)
                    st.session_state.show_settings = False
                    st.rerun()
        
        # ----------------- Web Search Tab -----------------
        elif st.session_state.settings_tab == "web_search":
            st.markdown("### 🌐 Web Search Configuration")
            
            st.markdown("#### 🔍 Search Provider")
            provider = st.radio(
                "Select Search Provider",
                options=["Multilingual (Free)", "Tavily API", "DuckDuckGo (Fallback)"],
                index=["Multilingual (Free)", "Tavily API", "DuckDuckGo (Fallback)"].index(
                    "Multilingual (Free)" if st.session_state.search_provider == "multilingual" else
                    "Tavily API" if st.session_state.search_provider == "tavily" else
                    "DuckDuckGo (Fallback)"
                ),
                help="""
                - **Multilingual**: Free, real-time search in 20+ languages
                - **Tavily API**: Premium search with API key (better results)
                - **DuckDuckGo**: Basic fallback search
                """
            )
            
            # Map selection to provider code
            provider_map = {
                "Multilingual (Free)": "multilingual",
                "Tavily API": "tavily",
                "DuckDuckGo (Fallback)": "duckduckgo"
            }
            
            new_provider = provider_map[provider]
            if new_provider != st.session_state.search_provider:
                st.session_state.search_provider = new_provider
            
            # Show API key input for Tavily
            if st.session_state.search_provider == "tavily":
                st.divider()
                st.markdown("#### 🔑 Tavily API Key")
                
                api_key = st.text_input(
                    "Enter Tavily API Key",
                    type="password",
                    value=st.session_state.search_api_key,
                    help="Get free API key from https://tavily.com"
                )
                
                if api_key != st.session_state.search_api_key:
                    st.session_state.search_api_key = api_key
                
                if api_key:
                    # Test API key
                    if st.button("🧪 Test API Key", key="test_tavily"):
                        with st.spinner("Testing API connection..."):
                            try:
                                from tavily import TavilyClient
                                client = TavilyClient(api_key=api_key)
                                response = client.search(query="test", max_results=1)
                                
                                if response and "results" in response:
                                    st.success("✅ API key is valid!")
                                    st.info(f"Test query returned {len(response['results'])} result(s)")
                                else:
                                    st.error("❌ API key test failed - no results returned")
                            except Exception as e:
                                st.error(f"❌ API key test failed: {str(e)}")
            
            st.divider()
            
            st.markdown("#### 📊 Search Statistics")
            try:
                stats = get_search_stats()
                st.json(stats, expanded=False)
            except:
                st.info("Search statistics not available")
            
            st.divider()
            
            st.markdown("#### 🧪 Test Search")
            test_query = st.text_input("Test Query", placeholder="Enter a query to test search")
            test_lang = st.selectbox("Language", options=list(get_supported_languages().keys()), index=0)
            
            if test_query and st.button("🔍 Test Search", use_container_width=True):
                with st.spinner("Searching..."):
                    result = perform_web_search(test_query, test_lang)
                    
                    if result:
                        st.success("✅ Search successful!")
                        with st.expander("View Results", expanded=True):
                            st.markdown(result[:1000])
                    else:
                        st.error("❌ No results found")
        
        # ----------------- OCR Techniques Tab -----------------
        elif st.session_state.settings_tab == "ocr_tech":
            st.markdown("### 🔍 OCR Techniques & Models")
            
            with st.expander("📚 About OCR", expanded=True):
                st.markdown("""
                OCR automatically processes scanned/image PDFs.
                - **Text-based PDFs**: Standard extraction (fast)
                - **Image-based PDFs**: OCR processing (slower)
                """)
            
            st.divider()
            st.markdown("#### ⚙️ OCR Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                languages = st.multiselect(
                    "OCR Languages",
                    options=["en", "hi", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa", "ur", "ar", "fr", "de", "es", "zh", "ja", "ko"],
                    default=st.session_state.ocr_languages,
                    help="Select languages for OCR recognition"
                )
                if languages:
                    st.session_state.ocr_languages = languages
            
            with col2:
                use_gpu = st.checkbox(
                    "Use GPU acceleration",
                    value=st.session_state.ocr_use_gpu,
                    disabled=not EASYOCR_AVAILABLE,
                    help="Requires CUDA compatible GPU"
                )
                st.session_state.ocr_use_gpu = use_gpu
            
            st.divider()
            st.markdown("#### 🧪 Test OCR")
            
            ocr_test_file = st.file_uploader(
                "Test PDF/Image",
                type=["pdf", "png", "jpg", "jpeg"],
                key="ocr_test"
            )
            
            if ocr_test_file:
                st.session_state.ocr_test_file = ocr_test_file
            
            col_test1, col_test2 = st.columns(2)
            with col_test1:
                if st.button("🔍 Analyze PDF Type", use_container_width=True) and st.session_state.ocr_test_file:
                    with st.spinner("Analyzing..."):
                        temp_path = Path(f"data/temp/ocr_test_{st.session_state.ocr_test_file.name}")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(st.session_state.ocr_test_file.getbuffer())
                        
                        pdf_type, text_ratio = detect_pdf_type(str(temp_path))
                        st.session_state.ocr_test_result = {
                            "type": pdf_type,
                            "text_ratio": text_ratio,
                            "needs_ocr": pdf_type == "image"
                        }
            
            with col_test2:
                if st.button("🚀 Run OCR", use_container_width=True) and st.session_state.ocr_test_file:
                    with st.spinner("Running OCR..."):
                        try:
                            ocr_processor = OCRProcessor(
                                languages=st.session_state.ocr_languages,
                                gpu=st.session_state.ocr_use_gpu
                            )
                            
                            temp_path = Path(f"data/temp/ocr_test_{st.session_state.ocr_test_file.name}")
                            pages = ocr_processor.process_pdf(str(temp_path))
                            
                            if pages:
                                st.session_state.ocr_test_result = {
                                    "pages": pages,
                                    "total_pages": len(pages),
                                    "sample_text": pages[0]["text"][:500]
                                }
                                st.success(f"Extracted {len(pages)} pages")

                        except Exception as e:  # ← ADD THIS EXCEPT BLOCK
                            st.error(f"OCR processing failed: {e}")
                            st.session_state.ocr_test_result = None
            
            # Show OCR results
            if st.session_state.ocr_test_result:
                st.divider()
                with st.expander("📊 OCR Results", expanded=True):
                    result = st.session_state.ocr_test_result
                    
                    if "pages" in result:
                        st.metric("Pages Extracted", result["total_pages"])
                        st.text_area("Sample Text", value=result["sample_text"], height=200)
                    elif "type" in result:
                        st.info(f"**PDF Type:** {result['type'].upper()}")
                        st.info(f"**Text Ratio:** {result['text_ratio']:.2%}")
        
        # ----------------- TTS Settings Tab -----------------
        elif st.session_state.settings_tab == "tts_settings":
            st.markdown("### 🔊 Text-to-Speech Settings")
            
            st.markdown("#### ⚙️ Configuration")
            tts_lang = st.selectbox(
                "Voice Language",
                options=[("English", "en"), ("Hindi", "hi"), ("French", "fr"), ("Spanish", "es"), 
                        ("German", "de"), ("Chinese", "zh"), ("Japanese", "ja")],
                format_func=lambda x: x[0],
                index=0
            )
            
            if tts_lang[1] != st.session_state.tts_language:
                st.session_state.tts_language = tts_lang[1]
                st.session_state.tts_processor.clear_cache()
            
            st.divider()
            st.markdown("#### 🧪 Test TTS")
            test_text = st.text_area("Test Text", value="Hello, this is a test of text-to-speech.")
            
            if st.button("🎵 Generate & Play", use_container_width=True) and test_text:
                with st.spinner("Generating audio..."):
                    audio_base64 = st.session_state.tts_processor.text_to_audio(test_text, st.session_state.tts_language)
                    
                    if audio_base64:
                        audio_html = f"""
                        <audio controls autoplay>
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.success("Audio generated!")
            
            st.divider()
            if st.button("🧹 Clear Audio Cache", use_container_width=True):
                st.session_state.tts_processor.clear_cache()
                st.success("Cache cleared!")
        
        # ----------------- Language Settings Tab -----------------
        elif st.session_state.settings_tab == "language_settings":
            st.markdown("### 🌍 Language Settings")
            
            st.markdown("#### 📚 Supported Languages")
            supported = get_supported_languages()
            
            # Show language grid
            cols = st.columns(4)
            for idx, (code, name) in enumerate(supported.items()):
                with cols[idx % 4]:
                    st.info(f"**{code}**\n{name}")
            
            st.divider()
            
            st.markdown("#### 🎯 Language Detection")
            st.caption("Test language detection on sample text")
            
            test_text = st.text_area("Enter text for language detection", height=100)
            
            if test_text and st.button("🌍 Detect Language", use_container_width=True):
                with st.spinner("Detecting..."):
                    lang_info = detect_query_language(test_text)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Language", get_language_name(lang_info["language"]))
                    with col2:
                        st.metric("Code", lang_info["language"])
                    with col3:
                        st.metric("Confidence", f"{lang_info.get('confidence', 0):.2%}")
                    
                    if "all_possibilities" in lang_info:
                        with st.expander("All Possibilities", expanded=False):
                            for poss in lang_info["all_possibilities"][:5]:
                                st.caption(f"{get_language_name(poss['lang'])}: {poss['prob']:.2%}")
            
            st.divider()
            
            st.markdown("#### 🔄 Translation Test")
            col_trans1, col_trans2 = st.columns(2)
            with col_trans1:
                trans_text = st.text_input("Text to translate", "Hello world")
                source_lang = st.selectbox("From", options=list(supported.keys()), index=0)
            
            with col_trans2:
                target_lang = st.selectbox("To", options=list(supported.keys()), index=1)
                
                if st.button("🔄 Translate", use_container_width=True) and trans_text:
                    with st.spinner("Translating..."):
                        translated = translate_text(trans_text, target_lang, source_lang)
                        
                        if translated:
                            st.success("Translation successful!")
                            st.info(f"**Result:** {translated}")
        
        # ----------------- Debug Settings Tab -----------------
        elif st.session_state.settings_tab == "debug_settings":
            st.markdown("### 🐛 Debug Settings")
            
            st.warning("Use these tools to diagnose issues")
            
            st.divider()
            st.markdown("#### 📊 Last Query Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.last_confidence:
                    conf_value = st.session_state.last_confidence
                    conf_color = "🟢" if conf_value >= 0.75 else "🟡" if conf_value >= 0.55 else "🔴"
                    st.metric("Confidence", f"{conf_value:.3f} {conf_color}")
            
            with col2:
                if st.session_state.last_context_count:
                    st.metric("Contexts Found", st.session_state.last_context_count)
            
            if st.session_state.last_fallback_reason:
                st.info(f"**Last Fallback:** {st.session_state.last_fallback_reason}")
            
            st.divider()
            st.markdown("#### ⚙️ Configuration")
            
            new_threshold = st.slider(
                "Low Confidence Threshold",
                0.0, 1.0, st.session_state.low_confidence_threshold, 0.05,
                help="PDF answers with confidence BELOW this trigger web search"
            )
            st.session_state.low_confidence_threshold = new_threshold
            
            debug_mode = st.checkbox(
                "Enable Debug Mode",
                value=st.session_state.show_source_debug,
                help="Show detailed debug information"
            )
            st.session_state.show_source_debug = debug_mode
            
            st.divider()
            
            if st.button("🔄 Reset Debug Data", use_container_width=True):
                st.session_state.last_fallback_reason = None
                st.session_state.last_confidence = None
                st.session_state.last_context_count = 0
                st.success("Debug data cleared!")
                st.rerun()
    
    st.markdown("---")

# ======================================================
# CHAT AREA
# ======================================================
active_chat = chat_manager.get_active_chat()

if not active_chat:
    st.info("👈 Create or select a chat to begin")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Create a new chat** using the ➕ button in the sidebar
        2. **Go to Settings → PDF Manager** to upload your documents
        3. **Ask questions** in any language
        4. **Get answers** from your documents or the web
        
        **🌍 Multilingual Features:**
        - Upload PDFs in English, Hindi, Marathi, Gujarati, French, etc.
        - Ask questions in any supported language
        - Get responses in the same language as your question
        - Web search in multiple languages
        
        **📚 Supported Languages:** 20+ languages including Hindi, Marathi, Gujarati, Bengali, French, German, Spanish, Chinese, Japanese, etc.
        """)
    
    st.stop()

# Initialize token tracker for this chat
if active_chat.chat_id not in st.session_state.token_trackers:
    st.session_state.token_trackers[active_chat.chat_id] = TokenTracker()

tracker = st.session_state.token_trackers[active_chat.chat_id]

# Show active PDFs and language info
if active_chat and active_chat.pdf_names:
    # PDF info with language badges
    pdf_display = []
    for pdf in active_chat.pdf_names:
        # Check if we have language info for this PDF
        pdf_lang = "🌐"
        if active_chat.chat_id in st.session_state.get("pdf_languages", {}):
            pdf_idx = active_chat.pdf_names.index(pdf)
            if pdf_idx < len(st.session_state.pdf_languages[active_chat.chat_id]):
                lang_code = st.session_state.pdf_languages[active_chat.chat_id][pdf_idx]
                lang_name = get_language_name(lang_code)
                # pdf_lang = f"🌍 {lang_name}"
        
        pdf_display.append(f"{pdf_lang} **{pdf}**")
    
    st.markdown(f"**📚 Active Documents:** {' | '.join(pdf_display)}")
    
    # Show processing method if available
    with st.expander("🔧 Document Info", expanded=False):
        st.write("**Processing Methods:**")
        st.write("- 🔤 Text extraction for searchable PDFs")
        st.write("- 🖼️ OCR for scanned/image PDFs")
        st.write("**Note:** Processing method is automatically determined")
    
    # Force PDF mode indicator
    if st.session_state.force_pdf_only:
        st.warning("📄 **PDF-Only Mode:** Web search is disabled. All answers will come from your documents.")
    
    st.divider()

# Chat history display
if active_chat.index_name:
    st.markdown("### 💬 Conversation")
    
    # Display each message in chat history
    for idx, msg in enumerate(active_chat.chat_history):
        if msg["role"] == "user":
            # User message
            st.markdown(
                f"""
                <div class="chat-container">
                    <div class="chat-user">
                        <b>🧑 You:</b> {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Assistant message
            source = msg.get("source", "pdf")
            message_content = msg["content"]
            lang_code = msg.get("language", "en")
            lang_name = get_language_name(lang_code)
            
            # Create columns for message and controls
            col1, col2 = st.columns([6, 1])
            
            with col1:
                # Language badge
                lang_badge = f"🌍 {lang_name}" if lang_code != "en" else ""
                
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-ai">
                            <b>🤖 DocuKnow AI:</b> {lang_badge}<br>
                            {message_content}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                # TTS button
                if len(message_content.strip()) > 10:
                    if is_currently_playing(idx):
                        if st.button("⏸️", key=f"pause_{idx}", help="Pause audio"):
                            pause_tts()
                            st.rerun()
                    else:
                        if st.button("🔊", key=f"play_{idx}", help=f"Listen in {lang_name}"):
                            play_tts(message_content, idx, lang_code)
                            st.rerun()
            
            # Source and confidence information
            if source == "pdf":
                st.caption(f"✅ **Answer sourced from your documents** ({lang_name})")
                
                # Show confidence if available
                if f"contexts_{idx}" in st.session_state.message_contexts:
                    contexts = st.session_state.message_contexts[f"contexts_{idx}"]
                    if contexts:
                        confidence = calculate_confidence(contexts)
                        
                        # Color-coded confidence badge
                        conf_color = {
                            "High": "🟢",
                            "Medium": "🟡", 
                            "Low": "🔴"
                        }.get(confidence["level"], "⚪")
                        
                        st.caption(f"{conf_color} **Confidence:** {confidence['level']} ({confidence['score']:.3f})")
                        
                        # Show sources
                        citations = format_citations(contexts)
                        if citations:
                            with st.expander("📄 Sources & Citations", expanded=False):
                                for src in citations:
                                    st.markdown(f"- {src}")
            else:
                st.caption(f"🌐 **Answer sourced from web search** ({lang_name})")
                
                # Show debug info if enabled
                if st.session_state.show_source_debug and st.session_state.last_fallback_reason:
                    with st.expander("ℹ️ Why web search was used", expanded=False):
                        st.write(st.session_state.last_fallback_reason)
                        if st.session_state.last_confidence:
                            st.write(f"Document confidence was too low: {st.session_state.last_confidence:.3f}")
            
            # Audio player if this message is playing
            if st.session_state.current_playing_index == idx and st.session_state.current_playing_audio:
                st.markdown("---")
                
                # Audio player HTML
                audio_html = f"""
                <audio id="audioPlayer_{idx}" controls autoplay>
                    <source src="data:audio/mp3;base64,{st.session_state.current_playing_audio}" type="audio/mp3">
                    Your browser does not support audio playback.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Audio controls
                col_audio1, col_audio2, col_audio3 = st.columns([1, 1, 8])
                
                with col_audio1:
                    if st.session_state.tts_paused:
                        if st.button("▶️", key=f"resume_{idx}", help="Resume audio"):
                            resume_tts()
                            st.rerun()
                    else:
                        if st.button("⏸️", key=f"pause2_{idx}", help="Pause audio"):
                            pause_tts()
                            st.rerun()
                
                with col_audio2:
                    if st.button("⏹️", key=f"stop_{idx}", help="Stop audio"):
                        stop_tts()
                        st.rerun()
                
                st.markdown("---")
    
    # Chat input for new message
    query = st.chat_input(f"Message in any language... (当前语言: {get_language_name(st.session_state.user_language) if st.session_state.user_language != 'auto' else 'Auto-detect'})")
    
    if query:
        # Detect language of query
        query_lang = detect_and_set_language(query)
        query_lang_name = get_language_name(query_lang)
        
        # Add user message to chat with language info
        chat_manager.add_user_message(query)
        
        with st.spinner(f"🌍 Processing in {query_lang_name}..."):
            # Step 1: Retrieve contexts from PDF
            contexts = retrieve_context(query, active_chat.index_name)
            context_text = "\n".join(c["text"][:500] for c in contexts) if contexts else ""
            
            # Count input tokens
            tracker.count_input(query, context_text)
            
            # Step 2: Generate answer from PDF context
            pdf_answer = generate_answer(
                query=query,
                contexts=contexts,
                lang=query_lang
            )
            
            # Calculate confidence
            confidence = calculate_confidence(contexts) if contexts else {"score": 0.0, "level": "Low"}
            
            # Store debug information
            st.session_state.last_confidence = confidence["score"]
            st.session_state.last_context_count = len(contexts)
            
            # Step 3: Decide source (PDF vs Web)
            use_pdf_answer = True
            fallback_reason = None
            
            # Factor 1: Force PDF only mode
            if st.session_state.force_pdf_only:
                use_pdf_answer = True
                fallback_reason = "Force PDF-only mode enabled"
            
            # Factor 2: Check if PDF answer is valid
            elif not pdf_answer or "I am not confident" in pdf_answer or "not confident" in pdf_answer.lower():
                use_pdf_answer = False
                fallback_reason = "PDF returned 'not confident' answer"
            
            # Factor 3: Check confidence threshold
            elif confidence["score"] < st.session_state.low_confidence_threshold:
                use_pdf_answer = False
                fallback_reason = f"Low confidence ({confidence['score']:.3f} < {st.session_state.low_confidence_threshold})"
            
            # Factor 4: Check answer quality
            elif len(pdf_answer.strip()) < 20:
                use_pdf_answer = False
                fallback_reason = "PDF answer too short/insufficient"
            
            # Step 4: Execute decision
            if not use_pdf_answer:
                # Try web search
                raw_web = perform_web_search(query, query_lang)
                
                # Check web search quality
                if raw_web and len(raw_web.strip()) > 50:
                    answer = raw_web[:1500].strip()
                    source_type = "internet"
                    stored_contexts = []
                    fallback_reason = f"{fallback_reason} → Web search provided better answer"
                else:
                    # Web search failed, use PDF anyway
                    answer = pdf_answer
                    source_type = "pdf"
                    stored_contexts = contexts
                    fallback_reason = f"{fallback_reason} → Web search failed, using PDF answer"
            else:
                # Use PDF answer
                answer = pdf_answer
                source_type = "pdf"
                stored_contexts = contexts
                fallback_reason = None
            
            # Store debug information
            st.session_state.last_fallback_reason = fallback_reason
            
            # Count output tokens
            tracker.count_output(answer)
            
            # Add assistant message with language and source info
            chat_manager.add_assistant_message(answer, source_type)
            
            # Update last message with language info
            last_msg_idx = len(active_chat.chat_history) - 1
            active_chat.chat_history[last_msg_idx]["language"] = query_lang
            
            # Store contexts for citations
            if "message_contexts" not in st.session_state:
                st.session_state.message_contexts = {}
            
            st.session_state.message_contexts[f"contexts_{last_msg_idx}"] = stored_contexts
            
            # Rerun to update UI
            st.rerun()

else:
    # No documents uploaded yet
    st.info("📁 **No documents uploaded yet**")
    
    with st.expander("📚 How to get started", expanded=True):
        st.markdown("""
        1. **Click '⚙️ Settings'** in the sidebar
        2. **Go to '📂 PDF Manager'** tab
        3. **Upload your PDFs** (multiple languages supported)
        4. **Click 'Save & Process PDFs'** to index them
        5. **Start asking questions** in any language!
        
        **🌍 Multilingual Tips:**
        - Upload PDFs in different languages
        - Ask questions in the same language as your PDFs
        - Or ask in one language about PDFs in another language
        - The system will automatically detect and handle languages
        
        **📄 Supported Document Types:**
        - Text-based PDFs (fast processing)
        - Scanned/image PDFs (OCR processing)
        - Multi-language PDFs
        """)
    
    # Quick upload button
    if st.button("⚙️ Go to PDF Manager", type="primary", use_container_width=True):
        st.session_state.show_settings = True
        st.session_state.settings_tab = "pdf"
        st.rerun()

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
footer_cols = st.columns(3)

with footer_cols[0]:
    st.caption(f"**Chat:** {active_chat.chat_name}")
    if tracker:
        totals = tracker.get_totals()
        st.caption(f"**Tokens:** {totals['total']:,}")

with footer_cols[1]:
    if st.session_state.user_language != "auto":
        st.caption(f"**Language:** {get_language_name(st.session_state.user_language)}")
    else:
        st.caption("**Language:** Auto-detect")
    
    if st.session_state.detected_languages:
        top_lang = max(st.session_state.detected_languages.items(), key=lambda x: x[1])[0]
        # st.caption(f"**Most used:** {get_language_name(top_lang)}")

with footer_cols[2]:
    st.caption("**DocuKnow AI v1.0**")
    st.caption("🌍 Multilingual Edition")

# ======================================================
# LOAD EXTERNAL CSS
# ======================================================

# Load Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# The actual CSS is in assets/styles.css - Streamlit loads it automatically