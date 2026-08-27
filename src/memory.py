import logging
import uuid
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class VectorStore:

  def __init__(self, db_path: str):
    self.client = chromadb.PersistentClient(path=db_path)
    self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    self.collection = self.client.get_or_create_collection(
        name="context_store",
        embedding_function=self.embedder,
        metadata={"hnsw:space": "cosine"},
    )

  def insert(self, text: str, metadata: dict | None = None) -> str:
    doc_id = str(uuid.uuid4())
    meta = metadata or {"source": "manual"}
    self.collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
    logger.info("Saved record %s", doc_id)
    return doc_id

  def query(self, text: str, k: int = 3) -> list[str]:
    count = self.collection.count()
    if count == 0:
      return []

    res = self.collection.query(
        query_texts=[text],
        n_results=min(k, count),
    )
    return res["documents"][0] if res and res.get("documents") else []

  def fetch_all(self) -> list[dict]:
    data = self.collection.get()
    if not data or not data.get("documents"):
      return []

    records = []
    for doc_id, text, meta in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
      records.append({"id": doc_id, "text": text, "metadata": meta})
    return records

  def reset(self) -> None:
    self.client.delete_collection("context_store")
    self.collection = self.client.get_or_create_collection(
        name="context_store",
        embedding_function=self.embedder,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Database reset complete.")