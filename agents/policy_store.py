"""Chroma-backed retrieval over the local credit policy documents."""

from pathlib import Path

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

POLICY_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "policy_docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"
COLLECTION_NAME = "credit_policy"
EMBEDDING_MODEL = "models/gemini-embedding-001"

_collection = None


def _chunk_document(path: Path) -> list[str]:
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    if not blocks:
        return []
    title = blocks[0].lstrip("#").strip()
    return [f"{title}: {block}" for block in blocks[1:]]


def _ingest(collection) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    ids, documents, metadatas = [], [], []
    for path in sorted(POLICY_DOCS_DIR.glob("*.md")):
        for i, chunk in enumerate(_chunk_document(path)):
            ids.append(f"{path.stem}-{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name})
    if not documents:
        return
    vectors = embeddings.embed_documents(documents)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        _ingest(collection)
    _collection = collection
    return _collection


def query_policy(query_text: str, k: int = 4) -> list[tuple[str, str]]:
    """Return up to k (chunk_text, source_filename) pairs relevant to query_text."""
    collection = get_collection()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector = embeddings.embed_query(query_text)
    results = collection.query(query_embeddings=[vector], n_results=k)
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    sources = [m["source"] for m in metas[0]] if metas[0] else []
    return list(zip(docs[0], sources))
