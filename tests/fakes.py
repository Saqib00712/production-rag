"""Test doubles. No network, no cost."""
import hashlib

from app.services.embedder import EmbeddingBatchResult, EmbeddingError


class HashEmbedder:
    model = "fake-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        vecs = [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]
        return EmbeddingBatchResult(vectors=vecs, total_tokens=sum(len(t) // 4 + 1 for t in texts))


class FailingEmbedder:
    model = "fake-embed"

    async def embed(self, texts):
        raise EmbeddingError("provider down")


class BowEmbedder:
    """Deterministic 'semantic' embedder: synonyms share a dimension, so
    'money' lands near 'refund' without any shared word. Lets tests prove
    that vector search finds what keyword search cannot."""

    model = "bow"
    GROUPS = [
        {"refund", "money", "return", "reimburse"},
        {"warranty", "defect", "defects", "broken", "guarantee"},
        {"shipping", "delivery", "arrive", "courier"},
        {"privacy", "data", "personal", "erase"},
    ]

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        from app.services.embedder import EmbeddingBatchResult

        self.calls.append(list(texts))
        vecs = []
        for t in texts:
            words = [w.strip(".,!?").lower() for w in t.split()]
            vecs.append([0.001 + sum(w in g for w in words) for g in self.GROUPS])
        return EmbeddingBatchResult(vectors=vecs, total_tokens=sum(len(t) // 4 + 1 for t in texts))


class ScriptedRerankerFactory:
    """Factory returning a reranker whose scores you control per test."""

    def __init__(self, score_map: dict[str, float] | None = None, raise_error: bool = False):
        self.score_map = score_map or {}
        self.raise_error = raise_error
        self.calls = 0

    def __call__(self):
        return self._make()

    def _make(self):
        from app.services.llm import LLMError
        from app.services.reranker import RerankResult
        from app.services.llm import LLMUsage

        outer = self

        class _Reranker:
            async def rerank(self, query, candidates):
                outer.calls += 1
                if outer.raise_error:
                    raise LLMError("rerank down")
                scores = {cid: outer.score_map.get(cid, 5.0) for cid, _ in candidates}
                return RerankResult(scores=scores, usage=LLMUsage(20, 5))

        return _Reranker()


class ScriptedGeneratorFactory:
    """Factory returning a generator whose JSON reply you control per test."""

    def __init__(self, reply: dict | None = None, raise_error: bool = False, none_generator: bool = False):
        self.reply = reply or {"answer": "Default answer [S1].", "citations": ["S1"], "insufficient_evidence": False}
        self.raise_error = raise_error
        self.none_generator = none_generator
        self.calls = 0
        self.last_sources = None
        self.last_memory_notes = None

    def __call__(self):
        if self.none_generator:
            return None
        return self._make()

    def _make(self):
        from app.services.llm import LLMError, LLMUsage
        from app.services.generator import GenerationResult

        outer = self

        class _Generator:
            async def generate(self, question, sources, memory_notes=None):
                outer.calls += 1
                outer.last_sources = sources
                outer.last_memory_notes = memory_notes
                if outer.raise_error:
                    raise LLMError("generation down")
                r = outer.reply
                return GenerationResult(
                    answer=r["answer"], cited_ids=list(r["citations"]),
                    insufficient_evidence=r["insufficient_evidence"], usage=LLMUsage(200, 40),
                )

        return _Generator()


class FakeStreamingGenerator:
    """Yields StreamChunk text pieces then a final usage chunk -- the same
    shape OpenAIStreamingGenerator produces, without any network call."""

    def __init__(self, text_chunks: list[str], prompt_tokens: int = 100, completion_tokens: int = 20):
        self.text_chunks = text_chunks
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = 0
        self.last_memory_notes = None

    async def generate_stream(self, question, sources, memory_notes=None):
        from app.services.llm import LLMUsage, StreamChunk

        self.calls += 1
        self.last_memory_notes = memory_notes
        for piece in self.text_chunks:
            yield StreamChunk(text=piece)
        yield StreamChunk(usage=LLMUsage(self.prompt_tokens, self.completion_tokens))


class ScriptedContextualizerFactory:
    """Factory returning a contextualizer whose rewrite you control per
    test, without any network call. Records every (question, history) pair
    it was asked to rewrite so tests can assert it was/wasn't called."""

    def __init__(self, rewrite: str | None = None, none_contextualizer: bool = False):
        self.rewrite = rewrite
        self.none_contextualizer = none_contextualizer
        self.calls: list[tuple[str, list]] = []

    def __call__(self):
        if self.none_contextualizer:
            return None
        outer = self

        class _Contextualizer:
            async def contextualize(self, question, history):
                from app.services.llm import LLMUsage

                outer.calls.append((question, list(history)))
                if not history:
                    return question, LLMUsage()
                return outer.rewrite or question, LLMUsage(50, 10)

        return _Contextualizer()


class ScriptedMemoryExtractorFactory:
    """Factory returning a memory extractor whose extracted facts you
    control per test, without any network call. Records every
    (question, answer, existing) triple it was asked to extract from."""

    def __init__(self, facts: list[str] | None = None, none_extractor: bool = False, raise_error: bool = False):
        self.facts = facts if facts is not None else []
        self.none_extractor = none_extractor
        self.raise_error = raise_error
        self.calls: list[tuple[str, str, list]] = []

    def __call__(self):
        if self.none_extractor:
            return None
        outer = self

        class _Extractor:
            async def extract(self, question, answer, existing):
                from app.services.llm import LLMUsage

                outer.calls.append((question, answer, list(existing)))
                if outer.raise_error:
                    from app.services.llm import LLMError

                    raise LLMError("extractor down")
                return list(outer.facts), LLMUsage(60, 15)

        return _Extractor()


class FailingStreamingGenerator:
    async def generate_stream(self, question, sources, memory_notes=None):
        from app.services.llm import LLMError, StreamChunk

        yield StreamChunk(text="partial answer before it breaks ")
        raise LLMError("stream broke")
        yield StreamChunk(text="unreachable")  # pragma: no cover


class ScriptedOcrFactory:
    """Factory returning an Ocr whose transcription you control per test,
    without any real vision call or PyMuPDF rendering. Records every image
    it was asked to transcribe so tests can assert call counts (the
    ocr_max_pages_per_document cap)."""

    def __init__(self, text: str = "transcribed text", none_ocr: bool = False, raise_error: bool = False):
        self.text = text
        self.none_ocr = none_ocr
        self.raise_error = raise_error
        self.calls: list[bytes] = []

    def __call__(self):
        if self.none_ocr:
            return None
        outer = self

        class _Ocr:
            async def transcribe(self, image_bytes: bytes):
                from app.services.ocr import OcrError, OcrResult

                outer.calls.append(image_bytes)
                if outer.raise_error:
                    raise OcrError("vision call down")
                return OcrResult(text=outer.text, prompt_tokens=120, completion_tokens=20)

        return _Ocr()
