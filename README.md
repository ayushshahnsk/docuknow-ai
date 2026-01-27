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