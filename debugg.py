# debug_chroma_path.py
import os, sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

DOC = "SRS.docx"
candidates = [
    r"C:\Users\techt\Desktop\Code\RAG\vector_db",
    "C:/Users/techt/Desktop/Code/RAG/vector_db",
    str(Path.cwd() / "vector_db"),
    str(Path.home() / "chromadb_vector_db"),
]

print("Python:", sys.version.splitlines()[0])
print("cwd:", repr(os.getcwd()))
print("path candidates:")
for p in candidates:
    p2 = os.path.abspath(p)
    print("  - raw:", repr(p), "-> abs:", repr(p2))
    print("     exists:", os.path.exists(p2), "isdir:", os.path.isdir(p2), "len:", len(p2))

# make sure one directory exists
safe_dir = os.path.abspath(candidates[0])
os.makedirs(safe_dir, exist_ok=True)
print("\nUsing safe_dir:", repr(safe_dir))

settings = Settings(persist_directory=safe_dir)
print("\nSettings:", settings)

try:
    print("\nTrying PersistentClient...")
    client = chromadb.PersistentClient(settings)
    print("PersistentClient created OK")
    client.persist()
    print("persist() OK")
except Exception as e:
    import traceback
    print("PersistentClient FAILED")
    traceback.print_exc()
    # fallback: try in-memory client to ensure chroma python import ok
    try:
        print("\nTrying in-memory Client()...")
        client2 = chromadb.Client()  # ephemeral, purely python: tests import & bindings
        print("In-memory client OK")
    except Exception as e2:
        print("In-memory client FAILED")
        traceback.print_exc()
