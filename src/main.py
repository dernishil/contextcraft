import json
import logging
import time
from pathlib import Path

import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import settings
from src.memory import VectorStore
from src.pipeline import PromptResolver

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ContextCraft Service")
store = VectorStore(settings.db_path)
resolver = PromptResolver(store)

# Real tokenizer, same family used by modern OpenAI-compatible models.
# Used only to MEASURE what the naive full-history approach would have
# cost -- we don't actually make that wasteful API call, we just count.
encoder = tiktoken.get_encoding("cl100k_base")
TOKEN_LOG_PATH = Path("token_log.jsonl")


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

  result = resolver.resolve_and_answer(req.text, req.history)
  contextcraft_tokens = result["tokens_used"]

  # Naive-equivalent: how many tokens a single-call, "resend the full
  # history every turn" approach would have cost THIS turn. We include
  # a comparable system prompt and REUSE the actual answer length as the
  # completion cost, since a naive approach still has to generate an
  # answer of roughly the same size -- this keeps the comparison fair
  # (same output, different amount of input context).
  naive_system_prompt = (
      "You are a helpful assistant. Use the full conversation history "
      "below to answer the user's latest question."
  )
  full_history_text = "\n".join(req.history + [req.text])
  naive_input_tokens = len(
      encoder.encode(naive_system_prompt + "\n" + full_history_text)
  )
  naive_completion_tokens = len(encoder.encode(result["response"]))
  naive_tokens = naive_input_tokens + naive_completion_tokens

  log_entry = {
      "timestamp": time.time(),
      "turn": len(req.history) + 1,
      "query": req.text,
      "contextcraft_tokens": contextcraft_tokens,
      "naive_tokens": naive_tokens,
  }
  with TOKEN_LOG_PATH.open("a") as f:
    f.write(json.dumps(log_entry) + "\n")

  store.insert(
      f"Q: {req.text} | Intent: {result['resolved']}",
      metadata={"source": "chat_log"},
  )

  return {
      "raw": req.text,
      "resolved": result["resolved"],
      "context": result["context_used"],
      "response": result["response"],
      "contextcraft_tokens": contextcraft_tokens,
      "naive_tokens": naive_tokens,
  }


@app.get("/api/v1/token-stats")
def get_token_stats():
  """Returns the logged per-turn token comparison, for the dashboard
  and analysis scripts to visualize."""
  if not TOKEN_LOG_PATH.exists():
    return {"data": []}

  entries = []
  with TOKEN_LOG_PATH.open() as f:
    for line in f:
      if line.strip():
        entries.append(json.loads(line))
  return {"data": entries}


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