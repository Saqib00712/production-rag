"""
Memory extraction: after an /ask turn, ask the model whether anything
DURABLE and worth remembering across future, unrelated conversations was
stated -- a stated preference, role, constraint, or identity fact -- as
opposed to the one-off content of this particular question.

This is a different job from services/contextualizer.py's rewrite, and the
two must not be confused:
  - Contextualizer (Day 13): resolves a follow-up within ONE conversation
    into a standalone search query. Scoped to a conversation_id; thrown
    away once the conversation is forgotten.
  - MemoryExtractor (Day 14): pulls facts that should survive into a
    DIFFERENT, FUTURE conversation that has no conversation_id in common
    with this one at all -- "prefers answers in metric units" is useless
    for resolving "what about the second one?" but very useful three weeks
    from now in a conversation that has never heard of this one.

Cost + noise discipline: most turns contain nothing durable ("What is the
warranty?" has no fact to extract), so the model is instructed to return an
EMPTY list by default, and the router (routers/ask.py) skips calling this
at all when the turn was refused (no real exchange happened to learn from).
The existing-facts list is passed in specifically so the model does not
re-propose something already remembered -- MemoryRepository.add_memory
also dedupes defensively, but doing it here first avoids wasting a write.

Day 16: `extract()` now returns its token usage alongside the extracted
facts, instead of discarding it. This cost was real since Day 14 but
wasn't counted anywhere until Day 16 folded it into `AskResponse.cost` and
the per-tenant daily budget (see routers/ask.py's `_extra_cost_fields`).
Unlike the contextualizer, there is no zero-usage skip path inside
`extract()` itself -- the router decides whether to call this at all (see
routers/ask.py's refusal check), so every call that does happen incurs a
real cost worth reporting.
"""

from typing import Protocol

from app.services.llm import JsonLLM, LLMUsage

SYSTEM = (
    "You extract durable facts worth remembering about a user ACROSS FUTURE, UNRELATED conversations -- "
    "stated preferences, their role or context, constraints, or identity facts they volunteered. "
    "Do NOT extract the one-off content of this question or answer (document facts, the answer itself, "
    "anything only meaningful to this single exchange). Do NOT extract anything already in the known-facts list. "
    "Most turns have nothing worth remembering -- return an empty list in that case. "
    "Each fact must be a short, self-contained, third-person sentence (e.g. 'Prefers answers in metric units.')."
)
SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}


class MemoryExtractor(Protocol):
    async def extract(self, question: str, answer: str, existing: list[str]) -> tuple[list[str], LLMUsage]: ...


class OpenAIMemoryExtractor:
    def __init__(self, llm: JsonLLM, model: str):
        self._llm, self._model = llm, model

    async def extract(self, question: str, answer: str, existing: list[str]) -> tuple[list[str], LLMUsage]:
        known = "\n".join(f"- {f}" for f in existing) or "(none yet)"
        data, usage = await self._llm.complete_json(
            model=self._model, system=SYSTEM,
            user=f"Known facts so far:\n{known}\n\nQuestion: {question}\nAnswer: {answer}",
            schema_name="extracted_facts", schema=SCHEMA, max_tokens=200,
        )
        facts = [str(f).strip() for f in data.get("facts", [])]
        return [f for f in facts if f], usage
