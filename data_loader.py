# ==========================================================
# TaxAssist AI - Aggressive File Data Loader (FIXED METADATAS)
# Purpose:
#   1. Force read all files inside your documents directory
#   2. Chunk text using a robust sliding window (700 characters)
#   3. Strip out duplicate matching chunks so ChromaDB doesn't crash
#   4. Save elements securely into a clean ChromaDB instance
# ==========================================================

import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from docx import Document  # Word reader fallback

# ==========================================================
# Configuration Settings
# ==========================================================
CHROMA_PATH = "chroma_db"
DATA_PATH = r"C:\Users\HODAdmin\Documents\TaxAssistAIV2\documents"
COLLECTION_NAME = "iras_tax_knowledge"

# Define the exact embedding function used across the app
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def force_extract_text(file_path, filename):
    """Attempts multiple extraction methods to guarantee content parsing."""
    text = ""
    # Method 1: Try reading as a native Microsoft Word Document
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if paragraphs:
            return "\n\n".join(paragraphs)
    except Exception:
        pass

    # Method 2: Try reading as a standard plain text file UTF-8
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        pass

    # Method 3: Try reading as a plain text file ANSI/Latin
    try:
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            return f.read()
    except Exception:
        pass

    return text


def split_text_sliding_window(text, chunk_size=700, chunk_overlap=150):
    """Splits text into chunks using a character-based sliding window approach."""
    chunks = []
    start = 0
    text_len = len(text)

    if text_len <= chunk_size:
        return [text]

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        if end < text_len:
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            boundary = max(last_period, last_newline)
            
            if boundary > (chunk_size * 0.5):
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start += (len(chunk) - chunk_overlap)
        
        if len(chunk) <= chunk_overlap:
            break

    return [c for c in chunks if c]


def main():
    # 1. Ensure source documents folder exists
    if not os.path.exists(DATA_PATH):
        print(f"[INFO] Folder '{DATA_PATH}' not found. Creating it now...")
        os.makedirs(DATA_PATH)
        print(f"[IMPORTANT] Please place your files into: '{DATA_PATH}'")
        return

    # 2. Get ALL files inside the directory (ignoring hidden internal system files)
    all_files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f)) and not f.startswith("~")]
    
    if not all_files:
        print(f"[WARNING] The directory is completely empty: {DATA_PATH}")
        return

    # 3. Reset database to clear old vector fragments
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing old database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 4. Connect to local ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    print(f"Processing folder files directly from: {DATA_PATH}")
    all_chunks = []
    all_ids = []
    all_metadata = []
    chunk_counter = 0
    
    # Track unique text chunks to prevent duplicates crashing collection.add()
    seen_chunks = set()

    for filename in all_files:
        file_path = os.path.join(DATA_PATH, filename)
        raw_text = force_extract_text(file_path, filename)
        
        if not raw_text.strip():
            print(f" -> [SKIP] {filename} is empty or formatting is unreadable.")
            continue

        file_chunks = split_text_sliding_window(raw_text)
        
        valid_file_chunks_count = 0
        for idx, chunk_text in enumerate(file_chunks):
            # Skip if this precise window layout was already added
            if chunk_text in seen_chunks:
                continue
                
            seen_chunks.add(chunk_text)
            chunk_counter += 1
            valid_file_chunks_count += 1
            
            all_chunks.append(chunk_text)
            all_ids.append(f"id_{chunk_counter}")
            all_metadata.append({"source": filename, "chunk_index": idx})
            
        print(f" -> Found {valid_file_chunks_count} unique chunks inside file: {filename}")

    # 5. Batch insert unique data into vector store
    if all_chunks:
        print(f"Uploading {len(all_chunks)} unique elements to ChromaDB store...")
        # FIXED: Changed 'metas' keyword to 'metadatas' for ChromaDB standard compatibility
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadata
        )
        print("Database building successfully completed!")
    else:
        print("[ERROR] Couldn't read any unique text out of the files in your directory.")


if __name__ == "__main__":
    main()

