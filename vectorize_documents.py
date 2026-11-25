import os
import re
import shutil
import pytesseract
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db_dir")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# Set Tesseract path on Windows if installed
if os.name == "nt":
    _tesseract_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_tesseract_default):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_default


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _extract_source_url(text: str) -> str:
    """Extract 'Source URL: ...' from the header of scraped Markdown files."""
    for line in text.splitlines()[:6]:
        match = re.match(r"^Source URL:\s*(https?://\S+)", line.strip(), re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF using PyPDF2, with pytesseract OCR fallback if empty."""
    reader = PdfReader(file_path)
    text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)
    if text.strip():
        return text

    # Optional OCR fallback if pdf2image + Tesseract are available
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(file_path)
        ocr_pages = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_pages)
    except Exception:
        return text


def load_single_document(file_path: str) -> list[Document]:
    """Load a single .pdf, .txt, or .md file into LangChain Document(s)."""
    filename = os.path.basename(file_path)
    if not os.path.isfile(file_path) or filename == "information.md":
        return []

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = _extract_pdf_text(file_path)
        source_url = ""
    elif ext in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        source_url = _extract_source_url(text)
    else:
        return []

    if not text.strip():
        return []

    metadata = {"source": filename}
    if source_url:
        metadata["source_url"] = source_url

    return [Document(page_content=text, metadata=metadata)]


def list_indexed_documents(data_dir: str = DATA_DIR) -> list[str]:
    """Return sorted list of knowledge-base files in data/."""
    if not os.path.exists(data_dir):
        return []
    files = [
        f
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and f != "information.md"
        and os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
    ]
    return sorted(files)


def ingest_files(
    file_paths: list[str],
    persist_directory: str = VECTOR_DB_DIR,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> tuple[int, list[str]]:
    """
    Read .pdf, .txt, and .md files, chunk them (1200/250), and append them
    to the persistent ChromaDB without wiping existing vectors.
    Returns (total_chunks_added, status_lines).
    """
    os.makedirs(persist_directory, exist_ok=True)
    if embeddings is None:
        embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    status_lines = []

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            docs = load_single_document(path)
            if not docs:
                status_lines.append(f"- `{filename}`: skipped (empty or unsupported)")
                continue
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            status_lines.append(f"- `{filename}`: {len(chunks)} chunks")
        except Exception as e:
            status_lines.append(f"- `{filename}`: error ({e})")

    if all_chunks:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        vectordb.add_documents(all_chunks)

    return len(all_chunks), status_lines


def rebuild_vector_store(data_dir: str = DATA_DIR, persist_directory: str = VECTOR_DB_DIR):
    """Clear old vector_db_dir/ vectors and re-index all files in data/."""
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(persist_directory):
        for entry in os.listdir(persist_directory):
            if entry == "information.md":
                continue
            entry_path = os.path.join(persist_directory, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
    else:
        os.makedirs(persist_directory, exist_ok=True)

    files = [os.path.join(data_dir, f) for f in list_indexed_documents(data_dir)]
    if not files:
        print("No documents found in 'data/' to index.")
        return

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = get_embeddings()
    print(f"Ingesting {len(files)} document(s) into '{persist_directory}'...")
    total_chunks, summary = ingest_files(files, persist_directory=persist_directory, embeddings=embeddings)
    for line in summary:
        print(line)
    print(f"Successfully vectorized {total_chunks} total chunks into '{persist_directory}'.")


if __name__ == "__main__":
    rebuild_vector_store()
