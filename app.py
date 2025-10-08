#!/usr/bin/env python3
"""
Build and Save Local ChromaDB Vector Database
---------------------------------------------
Reads a DOCX file, embeds it, and stores vectors locally for later use.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import os

# === CONFIG ===
DOCX_PATH = "SRS.docx"
DB_DIR = "./vector_db"
COLLECTION_NAME = "srs_docs"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Step 1: Extract text ===
def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

print(f"[+] Loading document: {DOCX_PATH}")
text = extract_text_from_docx(DOCX_PATH)

# === Step 2: Split into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text)
print(f"[+] Total chunks: {len(chunks)}")

# === Step 3: Embed ===
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(chunks)

# === Step 4: Create and persist ChromaDB ===
os.makedirs(DB_DIR, exist_ok=True)
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR
))

collection = client.get_or_create_collection(COLLECTION_NAME)

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk],
        embeddings=[embeddings[i]],
        metadatas=[{"source": DOCX_PATH, "chunk": i}]
    )

# === Step 5: Persist ===
client.persist()
print(f"[✅] Vector database saved locally at: {DB_DIR}")
