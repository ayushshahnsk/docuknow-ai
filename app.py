import streamlit as st
from pathlib import Path
import logging

from core.pdf_loader import load_pdf
from core.chunker import smart_chunk
from core.embeddings import embed_texts
from core.vectorstore import create_faiss_index
from core.retriever import retrieve_context
from core.generator import generate_answer
from core.ocr_processor import OCRProcessor
from core.pdf_detector import detect_pdf_type
from core.tts_processor import TTSProcessor  # NEW: TTS processor

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

# Store contexts for each assistant message to show citations
if "message_contexts" not in st.session_state:
    st.session_state.message_contexts = {}  # Format: {message_index: contexts}

# OCR processing state
if "ocr_test_file" not in st.session_state:
    st.session_state.ocr_test_file = None

if "ocr_test_result" not in st.session_state:
    st.session_state.ocr_test_result = None

if "ocr_languages" not in st.session_state:
    st.session_state.ocr_languages = ["en"]

if "ocr_use_gpu" not in st.session_state:
    st.session_state.ocr_use_gpu = False

# ======================================================
# TTS STATE (NEW)
# ======================================================
if "tts_processor" not in st.session_state:
    st.session_state.tts_processor = TTSProcessor()

if "current_playing_audio" not in st.session_state:
    st.session_state.current_playing_audio = None  # Currently playing audio data

if "current_playing_index" not in st.session_state:
    st.session_state.current_playing_index = None  # Index of message being played

if "tts_paused" not in st.session_state:
    st.session_state.tts_paused = False

if "tts_language" not in st.session_state:
    st.session_state.tts_language = "en"

chat_manager = st.session_state.chat_manager

# ======================================================
# TTS HELPER FUNCTIONS (NEW)
# ======================================================

def play_tts(text: str, message_index: int):
    """Generate and play TTS audio for given text."""
    try:
        # Generate audio
        audio_base64 = st.session_state.tts_processor.text_to_audio(
            text, 
            st.session_state.tts_language
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
            cleanup_tts_on_chat_switch()  # NEW: Stop audio on chat switch
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
    
    with left:
        # Navigation buttons for settings tabs
        if st.button("📂 PDF Manager", use_container_width=True):
            st.session_state.settings_tab = "pdf"
        
        if st.button("🧠 Model / API", use_container_width=True):
            st.session_state.settings_tab = "model"
        
        if st.button("🌐 API Help", use_container_width=True):
            st.session_state.settings_tab = "api_help"
        
        if st.button("🔍 OCR Techniques", use_container_width=True):
            st.session_state.settings_tab = "ocr_tech"
        
        # NEW: TTS Settings tab
        if st.button("🔊 TTS Settings", use_container_width=True):
            st.session_state.settings_tab = "tts_settings"
        
        st.divider()
        
        if st.button("❌ Close Settings", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()
    
    with right:
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
                        checked = st.checkbox("", value=True, key=f"pdf_active_{i}")
                        if checked:
                            selected_indices.append(i)
                    with c2:
                        st.markdown(f"📄 **{pdf.name}**")

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
                processing_info = []  # Store processing info for each PDF

                # Create processing status container
                status_container = st.empty()
                
                with st.spinner("Processing PDFs..."):
                    for file_idx, file in enumerate(selected_files):
                        pdf_names.append(file.name)
                        
                        # Update status
                        status_container.info(f"Processing {file.name} ({file_idx + 1}/{len(selected_files)})...")

                        temp_path = Path(
                            f"data/uploads/{active_chat.chat_id}_{file.name}"
                        )
                        temp_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())

                        # Load PDF with OCR support
                        pages = load_pdf(str(temp_path))
                        
                        # Collect processing info
                        if pages:
                            processing_method = pages[0].get("processed_with", "unknown")
                            processing_info.append({
                                "filename": file.name,
                                "method": processing_method,
                                "pages": len(pages)
                            })
                        
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
                
                # Show processing summary
                st.success("PDFs processed and activated!")
                
                # Display processing details
                with st.expander("📊 Processing Details", expanded=False):
                    for info in processing_info:
                        method_icon = "🔤" if info["method"] == "text_extraction" else "🖼️"
                        method_text = "Text Extraction" if info["method"] == "text_extraction" else "OCR Processing"
                        st.write(f"{method_icon} **{info['filename']}**: {method_text} ({info['pages']} pages)")
                
                st.session_state.show_settings = False
                st.rerun()

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
        
        elif st.session_state.settings_tab == "api_help":
            st.markdown("### 🌐 Tavily API Setup Guide")
            st.caption("How to get a free API key for web search functionality")
            
            # Step-by-step guide
            st.markdown("""
            #### 📋 Step 1: Sign up for Tavily
            1. Go to [Tavily's website](https://tavily.com)
            2. Click on **"Get Started"** or **"Sign Up"**
            3. Create an account using your email or Google account
            
            #### 🔑 Step 2: Get Your API Key
            1. After signing up, navigate to your **Dashboard**
            2. Look for the **"API Keys"** section
            3. Click **"Create New API Key"**
            4. Copy the generated API key (it will look like `tvly-xxxxxxxxxxxx`)
            
            #### ⚙️ Step 3: Configure in DocuKnow AI
            1. Go to **Settings → Model / API**
            2. Select **"Use my API Key"**
            3. Paste your Tavily API key in the input field
            4. Click **"Save Model Settings"**
            
            #### 🆓 Free Tier Information
            - **Free tier includes**: 1,000 API calls per month
            - **Rate limits**: Up to 10 requests per minute
            - **No credit card required** for free tier
            - **Features included**: Basic search, up to 5 results per query
            
            #### ❓ Troubleshooting
            - **"Invalid API Key"**: Make sure you copied the entire key without spaces
            - **"Rate Limit Exceeded"**: Wait a minute before making more requests
            - **"No internet results"**: Check if your query is clear and specific
            
            #### 🔒 Security Note
            - Your API key is stored **locally** in this chat session
            - It is **not** sent to any server except Tavily's
            - You can revoke your API key anytime from Tavily's dashboard
            """)
            
            # Quick actions
            st.divider()
            st.markdown("#### 🚀 Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy Signup Link", use_container_width=True):
                    st.write("https://tavily.com")
                    st.success("Link copied to clipboard (manually copy)")
            
            with col2:
                if st.button("🔑 Go to API Settings", use_container_width=True):
                    st.session_state.settings_tab = "model"
                    st.rerun()
            
            # Test API Key section (optional)
            st.divider()
            st.markdown("#### 🧪 Test Your API Key")
            
            test_api_key = st.text_input(
                "Enter API key to test (optional)",
                type="password",
                help="This will make a test call to verify your API key works"
            )
            
            if test_api_key and st.button("🧪 Test Connection", use_container_width=True):
                import os
                from tavily import TavilyClient
                
                with st.spinner("Testing API connection..."):
                    try:
                        # Test the API key
                        client = TavilyClient(api_key=test_api_key)
                        response = client.search(query="test", max_results=1)
                        
                        if response and "results" in response:
                            st.success("✅ API key is valid and working!")
                            st.info(f"Test query returned {len(response['results'])} result(s)")
                        else:
                            st.error("❌ API key test failed - no results returned")
                    except Exception as e:
                        st.error(f"❌ API key test failed: {str(e)}")
            
            # Footer note
            st.divider()
            st.caption("💡 Need help? Visit [Tavily Documentation](https://docs.tavily.com)")
        
        elif st.session_state.settings_tab == "ocr_tech":
            st.markdown("### 🔍 OCR Techniques & Models")
            st.caption("Configure and test OCR processing for scanned/image PDFs")
            
            # OCR Information Section
            with st.expander("📚 About OCR in DocuKnow AI", expanded=True):
                st.markdown("""
                #### What is OCR?
                Optical Character Recognition (OCR) converts images of text into machine-readable text.
                
                #### When is OCR used?
                - **Scanned PDFs** (image-based documents)
                - **PDFs with embedded images**
                - **Documents without selectable text**
                
                #### Automatic Detection
                DocuKnow AI automatically detects if a PDF needs OCR:
                - **Text-based PDFs**: Uses standard text extraction (fast)
                - **Image-based PDFs**: Uses OCR processing (slower but necessary)
                
                #### OCR Engine: EasyOCR
                - **Accuracy**: High accuracy for general documents
                - **Languages**: Supports 80+ languages
                - **Speed**: Moderate, with GPU acceleration option
                - **License**: Free and open-source
                """)
            
            st.divider()
            
            # OCR Configuration Section
            st.markdown("#### ⚙️ OCR Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Language selection
                languages = st.multiselect(
                    "OCR Languages",
                    options=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
                    default=st.session_state.ocr_languages,
                    help="Select languages for OCR recognition"
                )
                if languages:
                    st.session_state.ocr_languages = languages
            
            with col2:
                # GPU option
                use_gpu = st.checkbox(
                    "Use GPU acceleration",
                    value=st.session_state.ocr_use_gpu,
                    help="Requires CUDA compatible GPU and torch with CUDA support"
                )
                st.session_state.ocr_use_gpu = use_gpu
                
                if use_gpu:
                    st.info("GPU acceleration enabled (if available)")
                else:
                    st.info("Using CPU (slower but works everywhere)")
            
            st.divider()
            
            # OCR Test Section
            st.markdown("#### 🧪 Test OCR Processing")
            st.caption("Upload a scanned/image PDF to test OCR extraction")
            
            ocr_test_file = st.file_uploader(
                "Choose a test PDF",
                type=["pdf", "png", "jpg", "jpeg"],
                key="ocr_test_uploader"
            )
            
            if ocr_test_file:
                st.session_state.ocr_test_file = ocr_test_file
                st.info(f"File ready: {ocr_test_file.name}")
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                if st.button("🔍 Analyze PDF Type", use_container_width=True):
                    if st.session_state.ocr_test_file:
                        # Save uploaded file temporarily
                        temp_path = Path(f"data/temp/ocr_test_{st.session_state.ocr_test_file.name}")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(st.session_state.ocr_test_file.getbuffer())
                        
                        # Analyze PDF
                        pdf_type, text_ratio = detect_pdf_type(str(temp_path))
                        
                        st.session_state.ocr_test_result = {
                            "type": pdf_type,
                            "text_ratio": text_ratio,
                            "needs_ocr": pdf_type == "image",
                            "file_path": str(temp_path)
                        }
                        
                        st.success(f"PDF Type: **{pdf_type.upper()}**")
                        st.info(f"Text Ratio: {text_ratio:.2%}")
                        
                        if pdf_type == "image":
                            st.warning("This PDF needs OCR processing")
                        else:
                            st.success("This PDF can use standard text extraction")
            
            with col_test2:
                if st.button("🚀 Run OCR Test", use_container_width=True):
                    if st.session_state.ocr_test_file:
                        # Save uploaded file temporarily
                        temp_path = Path(f"data/temp/ocr_test_{st.session_state.ocr_test_file.name}")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(st.session_state.ocr_test_file.getbuffer())
                        
                        # Run OCR
                        with st.spinner("Running OCR (this may take a minute)..."):
                            try:
                                ocr_processor = OCRProcessor(
                                    languages=st.session_state.ocr_languages,
                                    gpu=st.session_state.ocr_use_gpu
                                )
                                
                                # Process with OCR
                                pages = ocr_processor.process_pdf(str(temp_path))
                                
                                if pages:
                                    st.session_state.ocr_test_result = {
                                        "pages": pages,
                                        "total_pages": len(pages),
                                        "sample_text": pages[0]["text"][:500] + "..." if len(pages[0]["text"]) > 500 else pages[0]["text"],
                                        "file_path": str(temp_path)
                                    }
                                    
                                    st.success(f"✅ OCR successful! Extracted {len(pages)} pages")
                                else:
                                    st.error("❌ OCR failed to extract any text")
                            except Exception as e:
                                st.error(f"OCR Error: {str(e)}")
            
            # Display OCR Test Results
            if st.session_state.ocr_test_result:
                st.divider()
                st.markdown("#### 📊 OCR Test Results")
                
                if "pages" in st.session_state.ocr_test_result:
                    result = st.session_state.ocr_test_result
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.metric("Pages Extracted", result["total_pages"])
                    
                    with col_res2:
                        st.metric("Processing Method", "OCR")
                    
                    with st.expander("📝 Sample Extracted Text", expanded=False):
                        st.text_area(
                            "First page text (sample)",
                            value=result["sample_text"],
                            height=200,
                            disabled=True
                        )
                    
                    with st.expander("🔧 Advanced Details", expanded=False):
                        st.json({
                            "languages_used": st.session_state.ocr_languages,
                            "gpu_acceleration": st.session_state.ocr_use_gpu,
                            "total_characters": sum(len(p["text"]) for p in result["pages"]),
                            "average_chars_per_page": sum(len(p["text"]) for p in result["pages"]) / len(result["pages"]) if result["pages"] else 0
                        })
                elif "type" in st.session_state.ocr_test_result:
                    result = st.session_state.ocr_test_result
                    
                    st.info(f"""
                    **Analysis Complete:**
                    - **PDF Type**: {result['type'].upper()}
                    - **Text Ratio**: {result['text_ratio']:.2%}
                    - **Needs OCR**: {'Yes' if result['needs_ocr'] else 'No'}
                    """)
            
            st.divider()
            
            # Performance Tips
            with st.expander("💡 OCR Performance Tips", expanded=False):
                st.markdown("""
                #### For Better OCR Results:
                1. **Use high-quality scans** (300 DPI or higher)
                2. **Ensure good contrast** between text and background
                3. **Avoid skewed/scanned pages** (straighten images)
                4. **Use clear fonts** (avoid handwriting when possible)
                
                #### Performance Considerations:
                - **CPU processing**: Slower but works on all systems
                - **GPU acceleration**: 5-10x faster (requires CUDA GPU)
                - **Memory usage**: Large PDFs may require more RAM
                - **Processing time**: ~2-10 seconds per page depending on complexity
                
                #### Supported File Formats:
                - **PDF** (scanned/image-based)
                - **PNG**, **JPG**, **JPEG** (image files)
                - **Multi-page TIFF** (coming soon)
                """)
            
            # Reset button
            if st.button("🔄 Reset Test", use_container_width=True):
                st.session_state.ocr_test_file = None
                st.session_state.ocr_test_result = None
                st.rerun()
        
        # NEW: TTS Settings Tab
        elif st.session_state.settings_tab == "tts_settings":
            st.markdown("### 🔊 Text-to-Speech Settings")
            st.caption("Configure audio playback for AI answers")
            
            # TTS Information
            with st.expander("📚 About TTS", expanded=True):
                st.markdown("""
                #### 🎧 Text-to-Speech Features
                - **Listen to AI answers** with one click
                - **Play/Pause/Stop** controls
                - **Natural voice** synthesis
                - **Multi-language** support
                
                #### 🔈 How it Works
                1. Click the **🔊 button** next to any AI answer
                2. Audio generates automatically (takes 2-5 seconds)
                3. Use controls to **pause, resume, or stop**
                4. Audio plays directly in your browser
                
                #### 📝 Note
                - Only **AI assistant answers** are converted to speech
                - User questions and citations are **not** read aloud
                - Internet connection required for audio generation
                """)
            
            st.divider()
            
            # TTS Configuration
            st.markdown("#### ⚙️ TTS Configuration")
            
            # Language selection
            tts_language = st.selectbox(
                "Voice Language",
                options=[
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Italian", "it"),
                    ("Portuguese", "pt"),
                    ("Hindi", "hi"),
                    ("Chinese", "zh"),
                    ("Japanese", "ja"),
                    ("Korean", "ko")
                ],
                format_func=lambda x: x[0],
                index=0,
                help="Select language for the TTS voice"
            )
            
            # Update language in session state
            if tts_language[1] != st.session_state.tts_language:
                st.session_state.tts_language = tts_language[1]
                st.session_state.tts_processor.clear_cache()  # Clear cache when language changes
            
            # Voice speed (optional)
            voice_speed = st.select_slider(
                "Voice Speed",
                options=["Slow", "Normal", "Fast"],
                value="Normal",
                help="Adjust speaking speed"
            )
            
            st.divider()
            
            # TTS Test Section
            st.markdown("#### 🧪 Test TTS Voice")
            st.caption("Enter text to test the current TTS settings")
            
            test_text = st.text_area(
                "Test Text",
                value="Hello! This is DocuKnow AI. I can read answers aloud for you.",
                height=100,
                max_chars=500
            )
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                if st.button("🎵 Generate & Play", use_container_width=True):
                    if test_text.strip():
                        with st.spinner("Generating audio..."):
                            audio_base64 = st.session_state.tts_processor.text_to_audio(
                                test_text,
                                st.session_state.tts_language
                            )
                            
                            if audio_base64:
                                # Display audio player
                                audio_html = f"""
                                <audio controls autoplay>
                                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                                """
                                st.markdown(audio_html, unsafe_allow_html=True)
                                st.success("✅ Audio generated successfully!")
                            else:
                                st.error("❌ Failed to generate audio")
                    else:
                        st.warning("Please enter some text to test")
            
            with col_test2:
                if st.button("🔄 Reset Test", use_container_width=True):
                    st.rerun()
            
            st.divider()
            
            # TTS Management
            st.markdown("#### 🗂️ Audio Management")
            
            col_mgmt1, col_mgmt2 = st.columns(2)
            
            with col_mgmt1:
                if st.button("🧹 Clear Audio Cache", use_container_width=True):
                    st.session_state.tts_processor.clear_cache()
                    st.success("Audio cache cleared!")
            
            with col_mgmt2:
                if st.button("⏹️ Stop All Audio", use_container_width=True):
                    stop_tts()
                    st.success("All audio stopped!")
            
            # Current Status
            with st.expander("📊 Current TTS Status", expanded=False):
                st.write(f"**Current Language**: {tts_language[0]}")
                st.write(f"**Voice Speed**: {voice_speed}")
                st.write(f"**Audio Cache Size**: {len(st.session_state.tts_processor.audio_cache)} items")
                
                if st.session_state.current_playing_audio:
                    st.info("🔊 Audio is currently playing")
                else:
                    st.info("🔇 No audio currently playing")
            
            # Tips
            with st.expander("💡 Usage Tips", expanded=False):
                st.markdown("""
                #### For Best Experience:
                1. **Wait for generation**: First playback may take 2-5 seconds
                2. **Clear cache** if experiencing issues
                3. **Stop audio** before switching chats
                4. **Use headphones** for better clarity
                
                #### Supported Browsers:
                - Chrome, Firefox, Edge, Safari
                - Mobile browsers (iOS/Android)
                
                #### Limitations:
                - Requires internet for audio generation
                - Max ~4000 characters per conversion
                - Audio quality depends on connection
                """)
    
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

# OCR Status Indicator
if active_chat and active_chat.pdf_names:
    st.caption("📄 " + ", ".join(active_chat.pdf_names))
    
    # Optional: Add OCR processing info display
    with st.expander("📊 Document Processing Info", expanded=False):
        st.write("Documents are automatically processed with:")
        st.write("- 🔤 Text extraction for searchable PDFs")
        st.write("- 🖼️ OCR for scanned/image PDFs")
        st.write("*Processing method is determined automatically*")
    st.divider()

if active_chat.index_name:
    st.markdown("### 💬 Conversation")

    # Display chat history
    for idx, msg in enumerate(active_chat.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"<div style='font-size:16px; margin-bottom:6px;'><b>🧑 You:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            source = msg.get("source", "pdf")
            message_content = msg["content"]

            # Create columns for message and TTS controls
            col1, col2 = st.columns([6, 1])
            
            with col1:
                # Display the assistant message
                st.markdown(
                    f"<div style='font-size:16px; line-height:1.6; margin-bottom:6px;'>"
                    f"<b>🤖 DocuKnow AI:</b><br>{message_content}</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                # TTS Play Button
                if len(message_content.strip()) > 10:  # Only show for substantial answers
                    if is_currently_playing(idx):
                        # Show pause button if this message is playing
                        if st.button("⏸️", key=f"pause_{idx}", help="Pause audio"):
                            pause_tts()
                            st.rerun()
                    else:
                        # Show play button
                        if st.button("🔊", key=f"play_{idx}", help="Listen to this answer"):
                            play_tts(message_content, idx)
                            st.rerun()

            # Apply STRICT rules for source display
            if source == "pdf":
                # PDF source: Show source indicator
                st.caption("✅ Answer sourced from document")
                
                # Check if we have contexts stored for this message
                if f"contexts_{idx}" in st.session_state.get("message_contexts", {}):
                    contexts = st.session_state.message_contexts[f"contexts_{idx}"]
                    if contexts:
                        confidence = calculate_confidence(contexts)
                        
                        # Display confidence score
                        st.success(f"Confidence Score: {confidence['score']}")
                        
                        # Display citations in expander
                        citations = format_citations(contexts)
                        if citations:
                            with st.expander("📄 Sources"):
                                for src in citations:
                                    st.markdown(f"- {src}")
            else:
                # Internet source: Only show source indicator
                st.caption("✅ Answer sourced from internet")
                # Do NOT show citations or confidence score for internet answers
            
            # Show audio player if this message is playing
            if st.session_state.current_playing_index == idx and st.session_state.current_playing_audio:
                st.markdown("---")
                
                # Audio player
                audio_html = f"""
                <audio id="audioPlayer_{idx}" controls autoplay>
                    <source src="data:audio/mp3;base64,{st.session_state.current_playing_audio}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                
                <script>
                    var audio = document.getElementById('audioPlayer_{idx}');
                    
                    // Handle pause state
                    if ({str(st.session_state.tts_paused).lower()}) {{
                        audio.pause();
                    }}
                    
                    // When audio ends, clear playing state
                    audio.onended = function() {{
                        // This would need Streamlit communication to clear state
                        console.log('Audio ended');
                    }};
                </script>
                """
                
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Audio controls
                col_controls = st.columns([1, 1, 1, 7])
                
                with col_controls[0]:
                    if st.session_state.tts_paused:
                        if st.button("▶️", key=f"resume_{idx}", help="Resume audio"):
                            resume_tts()
                            st.rerun()
                    else:
                        if st.button("⏸️", key=f"pause2_{idx}", help="Pause audio"):
                            pause_tts()
                            st.rerun()
                
                with col_controls[1]:
                    if st.button("⏹️", key=f"stop_{idx}", help="Stop audio"):
                        stop_tts()
                        st.rerun()
                
                with col_controls[2]:
                    # Volume control (simplified)
                    current_volume = st.select_slider(
                        "Volume",
                        options=["🔈", "🔉", "🔊"],
                        value="🔊",
                        key=f"volume_{idx}",
                        label_visibility="collapsed"
                    )
                
                st.markdown("---")

    # Chat input for new messages
    query = st.chat_input("Message DocuKnow AI…")

    if query:
        # Add user message to chat
        chat_manager.add_user_message(query)

        with st.spinner("Thinking..."):
            # Retrieve contexts from PDF
            contexts = retrieve_context(query, active_chat.index_name)
            context_text = "\n".join(c["text"][:500] for c in contexts)
            
            # Count input tokens
            tracker.count_input(query, context_text)

            # Generate answer from PDF context
            pdf_answer = generate_answer(query, contexts)
            confidence = calculate_confidence(contexts)

            # Decision: PDF vs Internet source
            if confidence["level"] == "Low":
                # Low confidence: Use web search
                api_key = None
                if active_chat.model_pref["type"] == "api":
                    api_key = active_chat.model_pref.get("api_key")

                raw_web = web_search(query, api_key=api_key)
                answer = raw_web[:800].strip() if raw_web else "No reliable information found."
                source_type = "internet"
                # For internet answers: store empty contexts
                stored_contexts = []
            else:
                # High/Medium confidence: Use PDF answer
                answer = pdf_answer
                source_type = "pdf"
                # For PDF answers: store contexts for citations display
                stored_contexts = contexts

            # Count output tokens
            tracker.count_output(answer)
            
            # Add assistant message to chat
            chat_manager.add_assistant_message(answer, source_type)
            
            # Store contexts for the new assistant message to show citations later
            # Get the index of the new message (last message in history)
            new_message_idx = len(active_chat.chat_history) - 1
            
            # Initialize message_contexts if not exists
            if "message_contexts" not in st.session_state:
                st.session_state.message_contexts = {}
            
            # Store contexts for this specific message
            st.session_state.message_contexts[f"contexts_{new_message_idx}"] = stored_contexts
            
            # Rerun to update UI
            st.rerun()

else:
    st.info("⬅ Upload PDFs from Settings to start chatting.")