import json
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
    """Resolve-only path (used by /api/v1/resolve for inspection/debugging).
    Kept separate from the combined chat path below."""
    retrieved = self.store.query(raw_input, k=settings.top_k)

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
          "tokens_used": 0,
      }

    resolved_text = raw_input
    tokens_used = 0
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
      if resp.usage:
        tokens_used = resp.usage.total_tokens
    except Exception as exc:
      logger.error("LLM call failed during prompt rewrite: %s", exc)

    return {
        "raw": raw_input,
        "resolved": resolved_text,
        "context_used": retrieved,
        "mock": False,
        "tokens_used": tokens_used,
    }

  def resolve_and_answer(self, raw_input: str, history: list[str]) -> dict:
    """Single-call path used by /api/v1/chat. Combines query resolution
    AND answering into ONE LLM request instead of two, to avoid paying
    for two separate system prompts + completions per turn."""
    retrieved = self.store.query(raw_input, k=settings.top_k)

    mem_section = (
        "\n".join([f"- {item}" for item in retrieved])
        if retrieved
        else "None."
    )
    hist_section = (
        "\n".join(history[-4:]) if history else "No previous dialog."
    )

    if not self.client:
      return {
          "raw": raw_input,
          "resolved": raw_input,
          "context_used": retrieved,
          "response": "Groq API key not configured.",
          "mock": True,
          "tokens_used": 0,
      }

    system_msg = (
        "You do two things in a single response: (1) rewrite the user's "
        "raw query into a clear, stand-alone question by resolving vague "
        "references using the conversation history and known facts, and "
        "(2) answer that resolved question helpfully.\n\n"
        "For general knowledge, creative, or reasoning questions (jokes, "
        "explanations, math, writing help, etc.) that do NOT depend on "
        "the user's personal information, answer normally using your own "
        "knowledge -- the known facts below are not required for these.\n\n"
        "For questions specifically about the user's personal details, "
        "preferences, or things they've told you before, ground your "
        "answer in the known facts below. You may combine multiple facts "
        "to make a reasonable inference. If personal information needed "
        "to answer isn't present in the facts, say so honestly rather "
        "than inventing it.\n\n"
        "Respond with ONLY valid JSON in this exact shape, no other text:\n"
        '{"resolved": "<rewritten question>", "answer": "<your answer>"}'
    )

    user_msg = f"""Known Facts:
{mem_section}

Recent Conversation:
{hist_section}

Raw Query:
{raw_input}"""

    resolved_text = raw_input
    answer_text = "Error: no response generated."
    tokens_used = 0

    try:
      resp = self.client.chat.completions.create(
          model=settings.model_name,
          messages=[
              {"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg},
          ],
          temperature=0.2,
          response_format={"type": "json_object"},
      )
      raw_content = resp.choices[0].message.content.strip()
      if resp.usage:
        tokens_used = resp.usage.total_tokens

      try:
        parsed = json.loads(raw_content)
        resolved_text = parsed.get("resolved", raw_input)
        answer_text = parsed.get("answer", raw_content)
      except json.JSONDecodeError:
        # Model didn't return clean JSON -- fall back gracefully rather
        # than crashing, and still surface something useful.
        logger.warning("Non-JSON response from model, using raw content.")
        answer_text = raw_content

    except Exception as exc:
      logger.error("LLM call failed during resolve_and_answer: %s", exc)
      answer_text = f"Error executing completion: {exc}"

    return {
        "raw": raw_input,
        "resolved": resolved_text,
        "context_used": retrieved,
        "response": answer_text,
        "mock": False,
        "tokens_used": tokens_used,
    }