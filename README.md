# Campus Care — Dayananda Sagar University (DSU) AI Student Support Assistant

A local **Retrieval-Augmented Generation (RAG)** chatbot built for **Dayananda Sagar University (DSU)** students, applicants, and parents. It retrieves verified context from DSU documents (`.pdf`, `.md`, and `.txt`) stored in **ChromaDB** and generates source-cited answers via **Groq (`openai/gpt-oss-120b`)** inside a modern **Gradio** web interface.

---

## Features

- **Grounded DSU Knowledge Base (`data/`):** Covers DSU campuses & helplines (Harohalli Main Campus & Kudlu Gate City Innovation Campus), schools & programs, DSAT/admissions, fee ranges, scholarships, placements, hostels, library, transport, CDSIMER hospital, and AIC-DSU innovation labs.
- **Modern Gradio UI (`gradio_app.py`):** Branded in DSU Deep Royal Blue (`#0B3D91`) and Warm Gold (`#F59E0B`), with a live indexed-document counter, quick campus contact card, and clickable starter prompts.
- **Live Document Upload & Auto-Indexing:** Upload `.pdf`, `.txt`, or `.md` files directly from the sidebar (`Upload & Index`) to append vectors to ChromaDB and hot-reload the retriever without restarting.
- **Source Citations:** Every assistant response includes a **Sources** footer citing the exact document filename and official URL when available.
- **Multi-Turn Memory & PDF Export:** Maintains conversational context (`ConversationBufferMemory`) and exports chat transcripts to PDF (`FPDF`).
- **100% Free-Tier & Local Stack:** Uses Groq free-tier inference, local HuggingFace embeddings (`all-MiniLM-L6-v2`), and local ChromaDB persistence.

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Groq (`openai/gpt-oss-120b`) |
| **Embeddings** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) |
| **Vector Database** | ChromaDB (`langchain-chroma`) |
| **Orchestration** | LangChain (`ConversationalRetrievalChain`, `ConversationBufferMemory`) |
| **Web UI** | Gradio (`6.32.0`) |
| **Document Processing** | `PyPDF2`, `pytesseract` (OCR fallback), `BeautifulSoup4`, `FPDF` |

---

## Project Structure

```text
College_RAG_Chatbot/
├── data/                                        # DSU Markdown docs & PDF handbooks
├── report/
│   ├── Campus_Care_DSU_Project_Report.md        # Detailed project report
│   └── Campus_Care_DSU_Research_Paper_Draft.txt # Research paper draft
├── vector_db_dir/                               # Persistent ChromaDB vector store
├── gradio_app.py                                # Gradio Web UI (http://localhost:7860)
├── vectorize_documents.py                       # Chunking (1200/250), embedding & live ingestion
├── fetch_dsu_docs.py                            # Official DSU web/PDF scraper
├── requirements.txt                             # Python dependencies
└── .env.example                                 # Environment variable template
```

---

## Setup & Installation (Windows PowerShell)

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/misbha-khanum/College_RAG_Chatbot.git
   cd College_RAG_Chatbot
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Configure your Groq API key:**
   Copy `.env.example` to `.env` and add your free key from [Groq Console](https://console.groq.com/keys):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Running the Project

1. **(Optional) Rebuild the vector database** after adding or editing files in `data/`:
   ```powershell
   .\.venv\Scripts\python.exe vectorize_documents.py
   ```

2. **Launch the Gradio UI:**
   ```powershell
   .\.venv\Scripts\python.exe gradio_app.py
   ```
   Open **`http://localhost:7860`** in Chrome.
