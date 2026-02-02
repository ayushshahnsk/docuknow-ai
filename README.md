# 🧠 DocuKnow AI

**DocuKnow AI** is an intelligent document assistant built using **Retrieval-Augmented Generation (RAG)** that allows users to ask questions from **single or multiple PDFs** and receive **accurate, citation-aware, confidence-scored answers**.

It is designed as a **real-world AI system**, not a demo toy.

---

## 🚀 Key Features

- 📄 **Single PDF & Multiple PDF support**
- 🧠 **Multi-Document Intelligence**
- ✂️ **Smart Chunking** (page & paragraph aware)
- ⚡ **Fast Semantic Search** using FAISS
- 🤖 **LLM-powered Answers** (Gemma via Ollama)
- 📌 **Citations with Page Numbers**
- 🟢🟡🔴 **Confidence-Based Answers**
- 💬 **Chat-style Interface**
- 🎨 **Modern Streamlit UI**

---

## 🧩 System Architecture



- User Uploads PDFs
↓
- PDF Loader (PyMuPDF)
↓
- Smart Chunking
↓
- Embeddings (Sentence Transformers)
↓
- FAISS Vector Database
↓
- Semantic Retriever (Top-K)
↓
- LLM Generator (Gemma)
↓
- Answer + Confidence + Citations

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit**

### Backend / AI
- **Python 3.10+**
- **Sentence Transformers**
<!-- - **FAISS** -->
- **Gemma3:4B (via Ollama)**

### Utilities
- PyMuPDF
- NumPy
- Requests

---

### 📄 DocuKnow AI – Planned Changes & Versions

✅ Version 1.0 – Core Stability & UX Fixes (CURRENT PRIORITY)
1️⃣ Hide Chat When Settings Is Open

What to do

When user clicks ⚙️ Settings:

Chat area must be completely hidden

Only the Settings panel should be visible

When user clicks ❌ Close Settings:

Chat must return to its previous state

Sidebar should remain visible

---

2️⃣ PDF Manager – PDFs Disappear After New Chat (Bug Fix)

What to do

When a new chat is created:

PDF Manager should start empty

PDFs must be:

Chat-specific

Not shared between chats

Switching back to an old chat:

Previously uploaded PDFs must reappear correctly

---

3️⃣ Citation & Source Not Visible (Regression Fix)

What to do

Restore visibility of:

Answer source

Citations

---

4️⃣ Answer Source Rules (STRICT)

What to do

If answer comes from PDF

Show:

✅ Answer sourced from document

✅ Citations

✅ Confidence score

If answer comes from Internet

Show:

✅ Answer sourced from internet

Do NOT show:

❌ Citations

❌ Confidence score

---

🚀 Version 2.0 – OCR Support
What to add

OCR processing for:

Scanned PDFs

Image-based PDFs

Flow:

Detect non-text PDF

Run OCR

Merge OCR text into existing chunking pipeline

---

🔊 Version 3.0 – Text to Speech (TTS)
What to add

Convert AI answers to speech

UI control:

Play / Pause button

Scope:

Answer text only (not citations)

---

🎨 Version 4.0 – UI Overhaul (shadcn)
What to change

Replace current Streamlit UI styles

Use shadcn-style components

Scope:

Chat UI

Settings UI

PDF Manager UI

Logic must remain unchanged
