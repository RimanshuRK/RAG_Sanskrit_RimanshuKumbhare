# 📘 Sanskrit RAG System – Workflow & Setup Documentation

---

# 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** for Sanskrit documents.

The system:

* Accepts Sanskrit queries
* Retrieves relevant content from documents
* Generates meaningful answers using an LLM

---

# 🤖 LLM Used

* **OpenAI LLM (CPU-based inference)**
* Used for final answer generation
* Model configurable via `.env`

👉 Example:

```
OPENAI_MODEL=gpt-4o-mini
```

---

# 📄 Supported Document Types

The system supports:

* `.txt`
* `.pdf`
* `.docx` ✅ (Important addition)

👉 DOCX files are parsed and processed like other documents

---

# 🧠 System Architecture (High-Level)

```
User Query
   ↓
Embedding Generation
   ↓
FAISS Vector DB
   ↓
Relevant Context Retrieval
   ↓
OpenAI LLM (Generator)
   ↓
Final Answer
```

---

# 🔄 End-to-End Workflow

## 1️⃣ Application Startup

* Load documents from `/data/`
* Supported formats: `.txt`, `.pdf`, `.docx`
* Preprocess text (clean + chunk)
* Generate embeddings
* Store in FAISS

---

## 2️⃣ Document Ingestion

### 📥 Preloaded Documents

* Stored in `/data/`
* Automatically loaded on startup

### 📤 User Upload (Optional)

* Upload via `/upload` API
* Same pipeline applied:

  * Extract → Clean → Chunk → Embed → Store

---

## 3️⃣ Preprocessing Pipeline

### 🔹 Text Cleaning

* Remove noise
* Normalize Sanskrit text

### 🔹 Chunking

* Break text into smaller chunks
* Maintain overlap for context

---

## 4️⃣ Embedding Generation

* Convert text chunks into vectors
* Used for semantic similarity

---

## 5️⃣ Vector Storage (FAISS)

* Store embeddings
* Perform fast similarity search

---

## 6️⃣ Query Handling

User interacts via:

### 🔹 REST API

```
POST /query
```

### 🔹 WebSocket

* Real-time communication

---

## 7️⃣ Retrieval Phase

* Query → embedding
* FAISS → top-k results
* Returns relevant chunks

---

## 8️⃣ Generation Phase

* Context + Query → OpenAI LLM
* Generates final answer in Hinglish / simple format

---

## ⚠️ 9️⃣ Error Handling

### 🔴 Rate Limit / API Failure

If OpenAI fails:

* Fallback response used
* Based on retrieved context

---

## 🔟 Final Response Format

```json
{
  "answer": "Generated explanation",
  "query": "User question",
  "contexts": [
    {
      "content": "Relevant chunk",
      "score": 0.64
    }
  ]
}
```

---

# 🧩 Core Components

| Component         | Role                  |
| ----------------- | --------------------- |
| Document Loader   | Load TXT/PDF/DOCX     |
| Preprocessor      | Clean + chunk text    |
| Embedding Service | Text → vector         |
| FAISS DB          | Store + search        |
| Retriever         | Fetch relevant chunks |
| Generator         | OpenAI LLM response   |
| RAG Service       | Orchestrates flow     |

---

# ⚙️ Project Setup

## 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate:

#### Windows:

```bash
venv\Scripts\activate
```

#### Mac/Linux:

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Environment Variables Setup

Create `.env` file:

```env
APP_NAME=SanskritRAG
DATA_FOLDER=./app/data
FAISS_INDEX_PATH=./app/faiss_index
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
TOP_K_RESULTS=3
LOG_LEVEL=INFO
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=1024
OPENAI_TEMPERATURE=0.3

```

---

# ▶️ Run the Project

```bash
uvicorn app.main:app --reload
```

👉 Server start:

```
http://127.0.0.1:8000
```

---

# 📡 API Usage

## 🔹 Query API

```bash
POST /query
```

### Request Body:

```json
{
  "question": "murkh sevak story kya hai?",
  "top_k": 3
}
```

---

## 🔹 Upload API (Optional)

```bash
POST /upload
```

* Upload `.txt`, `.pdf`, `.docx`

---

## 🔹 Swagger UI

👉 Open:

```
http://127.0.0.1:8000/docs
```

---

# ⚡ Performance Notes

* Fully CPU-based system
* Lightweight LLM recommended
* FAISS ensures fast retrieval

---

# 🔁 Reusability & Design

* Modular architecture
* Clean separation of concerns
* Easy to extend:

  * Add new models
  * Add new file formats

---

# 🎯 Final Flow Summary

```
Startup:
→ Load Docs → Preprocess → FAISS Index

Query:
→ User Input → Embed → Retrieve → LLM → Answer
```

---

# ✅ Conclusion

This project builds a **production-ready Sanskrit RAG system** using:

* FastAPI
* FAISS
* OpenAI LLM
* Clean Architecture

👉 Result:
Efficient, scalable, and CPU-friendly AI system for Sanskrit Q&A.
