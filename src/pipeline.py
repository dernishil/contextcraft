import logging
from groq import Groq
from src.config import settings
from src.memory import VectorStore

logger = logging.getLogger(__name__)


class PromptResolver:

  def __init__(self, store: VectorStore):
    self.store = store
    self.client = Groq(api_key=settings.api_key) if settings.api_key else None

  def expand_query(self, raw_input: str, history: list[str]) -> dict:
    manual_facts = self.store.query(
      raw_input, k=settings.top_k, where={"source": "manual"}
    )
    chat_logs = self.store.query(
        raw_input, k=settings.top_k, where={"source": "chat_log"}
    )
    retrieved = manual_facts + [c for c in chat_logs if c not in manual_facts]
    retrieved = retrieved[: settings.top_k]

    mem_section = (
        "\n".join([f"- {item}" for item in retrieved])
        if retrieved
        else "None."
    )
    hist_section = (
        "\n".join(history[-4:]) if history else "No previous dialog."
    )

    system_msg = (
        "You rewrite vague user messages into clear, stand-alone instructions."
        " Resolve all ambiguous references using the conversation history and"
        " context provided. Preserve the original goal. Do not execute or"
        " answer the query."
    )

    user_msg = f"""Relevant Reference Data:
{mem_section}

Recent Conversation:
{hist_section}

Raw Query:
{raw_input}

Rewritten Query:"""

    if not self.client:
      return {
          "raw": raw_input,
          "resolved": raw_input,
          "context_used": retrieved,
          "mock": True,
      }

    try:
      resp = self.client.chat.completions.create(
          model=settings.model_name,
          messages=[
              {"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg},
          ],
          temperature=0.0,
      )
      resolved_text = resp.choices[0].message.content.strip()
    except Exception as exc:
      logger.error("LLM call failed during prompt rewrite: %s", exc)
      resolved_text = raw_input

    return {
        "raw": raw_input,
        "resolved": resolved_text,
        "context_used": retrieved,
        "mock": False,
    }

  def run_completion(self, prompt: str, context: list[str] | None = None) -> str:
    if not self.client:
      return "Groq API key not configured."

    context_block = (
        "\n".join([f"- {item}" for item in context])
        if context
        else "No relevant context found."
    )

    full_prompt = f"""Use the following known facts about the user to answer their question.
You may combine multiple facts below to reasonably infer an answer.
If the answer truly cannot be inferred from these facts, say you don't know rather than guessing.

Known facts:
{context_block}

Question:
{prompt}"""

    try:
      resp = self.client.chat.completions.create(
          model=settings.model_name,
          messages=[{"role": "user", "content": full_prompt}],
          temperature=0.3,
      )
      return resp.choices[0].message.content.strip()
    except Exception as exc:
      logger.error("LLM call failed during completion: %s", exc)
      return f"Error executing completion: {exc}"