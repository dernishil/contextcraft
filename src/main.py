import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import settings
from src.memory import VectorStore
from src.pipeline import PromptResolver

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ContextCraft Service")
store = VectorStore(settings.db_path)
resolver = PromptResolver(store)


class ResolveRequest(BaseModel):
  text: str
  history: list[str] = []


class ExecuteRequest(BaseModel):
  text: str
  history: list[str] = []


class IngestRequest(BaseModel):
  content: str
  metadata: dict | None = None


@app.get("/health")
def healthcheck():
  return {"status": "ok"}


@app.post("/api/v1/resolve")
def resolve_query(req: ResolveRequest):
  if not req.text.strip():
    raise HTTPException(status_code=400, detail="Text field cannot be empty.")
  return resolver.expand_query(req.text, req.history)


@app.post("/api/v1/chat")
def handle_chat(req: ExecuteRequest):
  if not req.text.strip():
    raise HTTPException(status_code=400, detail="Text field cannot be empty.")

  resolved_data = resolver.expand_query(req.text, req.history)
  reply = resolver.run_completion(resolved_data["resolved"])

  store.insert(
      f"Q: {req.text} | Intent: {resolved_data['resolved']}",
      metadata={"source": "chat_log"},
  )

  return {
      "raw": req.text,
      "resolved": resolved_data["resolved"],
      "context": resolved_data["context_used"],
      "response": reply,
  }


@app.post("/api/v1/memory")
def add_memory(req: IngestRequest):
  if not req.content.strip():
    raise HTTPException(status_code=400, detail="Content cannot be empty.")
  doc_id = store.insert(req.content, req.metadata)
  return {"status": "created", "id": doc_id}


@app.get("/api/v1/memory")
def list_memory():
  return {"data": store.fetch_all()}


@app.delete("/api/v1/memory")
def clear_memory():
  store.reset()
  return {"status": "cleared"}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host=settings.host, port=settings.port)