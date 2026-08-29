# ContextCraft

A lightweight middleware service that reduces LLM token usage by managing conversation context intelligently — instead of resending full chat history on every request, it retrieves only the relevant memory needed to understand and resolve a user's query.

## The Problem

Most chat applications send the entire conversation history to the AI model on every single request. As conversations grow longer, this becomes:
- **Expensive** — more tokens = higher API costs
- **Slow** — larger payloads take longer to process
- **Unnecessary** — most of that history isn't relevant to the current message

## The Approach

ContextCraft sits between the user and the LLM as a middleware layer:

1. **Semantic Memory** — Facts and past exchanges are stored in a vector database (ChromaDB), each converted into an embedding (a numerical representation of meaning, not just words) using a sentence-transformer model.
2. **Context Retrieval** — When a new message arrives, the system searches memory by *meaning*, not exact keyword matching, and pulls back only the top-k most relevant entries.
3. **Coreference Resolution** — Vague or context-dependent queries (e.g. "fix that error", "what's my name?") are rewritten into clear, standalone instructions using the retrieved memory and recent conversation history.
4. **Enriched Completion** — Only this compact, resolved package — not the full raw history — is sent to the downstream LLM (via Groq) for a response.

This keeps prompts smaller and more targeted as conversations scale, rather than growing linearly with every turn.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM Provider | Groq |
| Frontend | Streamlit |
| Config | Pydantic Settings |
| Data Analysis | pandas, numpy, matplotlib |

## Requirements

- Python 3.10+
- A valid Groq API key ([console.groq.com](https://console.groq.com)) — optional for local mock testing

## Setup

1. Clone the repository and navigate into it.

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables — copy `.env.example` to `.env` and fill in your details:
```bash
cp .env.example .env
```
```dotenv
GROQ_API_KEY=your_key_here
DB_PATH=./data/chroma
MODEL_NAME=openai/gpt-oss-20b
HOST=0.0.0.0
PORT=8000
```

5. Start the backend:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

6. In a separate terminal, start the frontend:
```bash
streamlit run dashboard.py
```

7. Open the dashboard at `http://localhost:8501`

## Memory Analysis

The dashboard includes a "Memory Analysis" section that pulls all stored memory entries and analyzes them using pandas and numpy — showing total entry count, average entry length, a breakdown of entries by source (manual vs. auto-saved chat logs), and a distribution histogram rendered with matplotlib.

This same analysis can also be run standalone from the command line:
```bash
python analyze_memory.py
```
This prints summary statistics to the terminal and saves a chart as `memory_analysis.png`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/resolve` | Resolve a query using memory + history (no LLM completion) |
| POST | `/api/v1/chat` | Full pipeline: resolve query, get LLM response, store exchange |
| POST | `/api/v1/memory` | Manually store a fact/context entry |
| GET | `/api/v1/memory` | List all stored memory entries |
| DELETE | `/api/v1/memory` | Clear all stored memory |

## Status

This is an early-stage prototype built to explore token-efficient context management for LLM applications. Built with AI-assisted development and debugged/deployed independently.

## Debugging Journey — Problems Found & Fixed

Building and testing this project surfaced several real issues, each diagnosed and fixed through hands-on debugging:

### 1. API key not loading from `.env`
**Problem:** The app always reported "Groq API key not configured," even with a valid key set.
**Cause:** `config.py` read `os.getenv("GROQ_API_KEY", "")` directly at class-definition time, before `.env` had been loaded — so it always resolved to an empty string.
**Fix:** Switched to Pydantic's `Field(validation_alias="GROQ_API_KEY")`, letting `pydantic-settings` load the value correctly from `.env` at startup.

### 2. Model not found (404 from Groq)
**Problem:** Completion requests failed with `model_not_found`.
**Cause:** The configured model, `llama-3.1-8b-instant`, had moved to Groq's Enterprise tier and was no longer available on a free account.
**Fix:** Switched `MODEL_NAME` to `openai/gpt-oss-20b`, a model available on free accounts.

### 3. Retrieved context never reached the answer step
**Problem:** The AI would hallucinate answers (e.g. inventing a project name) instead of using facts that were clearly present in memory.
**Cause:** `expand_query()` correctly retrieved relevant memory, but that context was only used to rewrite the *question* — it was never passed into `run_completion()`, which generates the final *answer*. The answer step had no facts to draw from and was guessing.
**Fix:** `run_completion()` now accepts the retrieved context directly and includes it explicitly in the prompt, with an instruction to answer only from the given facts and say "I don't know" if the answer isn't present.

### 4. Retrieval crowded out by repeated questions
**Problem:** After asking the same question multiple times, memory search started returning near-duplicate copies of the *question itself* instead of the actual relevant fact.
**Cause:** Every chat exchange (even failed ones) is saved back into memory as `Q: ... | Intent: ...`. A repeated question is highly similar to its own past copies, so those copies out-ranked the real fact in the top-k semantic search results.
**Fix:** Memory retrieval now queries manually-saved facts and chat-log entries separately, prioritizing manual facts, so a genuine stored fact can no longer be crowded out by repeated question logs.

### 5. Precision vs. recall trade-off in answering
**Problem:** After fixing the hallucination issue, the model became overly cautious — refusing to answer questions that required combining two known facts (e.g. inferring a project's language from a stated favorite language), even though the inference was reasonable.
**Cause:** The instruction to avoid guessing was strict enough to also block valid, low-risk inference.
**Fix:** Adjusted the prompt to explicitly allow combining multiple retrieved facts to infer an answer, while still requiring an honest "I don't know" when the answer truly isn't supported by memory. This is an ongoing balance rather than a fully solved problem — see Future Work.

## Future Work

- Further tune the precision/recall balance between hallucination-avoidance and reasonable multi-fact inference
- Configurable memory decay / expiry for older, less relevant entries
- Support for additional LLM providers
- Token usage metrics/dashboard to quantify savings vs. naive full-history approaches
- Automated tests for the retrieval and resolution pipeline