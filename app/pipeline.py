from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .composer import compose
from .dictionary import Match, SignDictionary
from .karsl100_vocab import karsl100_vocabulary
from .llm import LLMService
from .rag import LocalRAG


class ConversationPipeline:
    def __init__(self, dictionary: SignDictionary, rag: LocalRAG, llm: LLMService, output_root: Path):
        self.dictionary = dictionary
        self.rag = rag
        self.llm = llm
        self.output_root = output_root

    def text_to_sign(self, text: str, compose_video: bool = True) -> dict:
        simplified = self.llm.simplify_for_signing(text, [e.gloss for e in self.dictionary.entries])
        return self._matches_to_result(text, simplified.sentence, simplified.sign_glosses, compose_video)

    def glosses_to_sign(self, glosses: list[str], compose_video: bool = True, source_text: str = "") -> dict:
        vocab = karsl100_vocabulary()
        sanitized = vocab.sanitize_glosses(glosses)
        sentence = source_text.strip() or " ".join(sanitized)
        return self._matches_to_result(sentence, sentence, sanitized, compose_video)

    def _match_gloss(self, gloss: str) -> Match:
        vocab = karsl100_vocabulary()
        canonical = vocab.resolve(gloss) or gloss
        exact = self.dictionary.match(canonical)
        if exact.gloss and exact.score >= 0.99:
            return exact
        if vocab.resolve(gloss):
            return Match(gloss, canonical, 1.0, exact.media_path, fallback=None if exact.media_path else "fingerspell")
        return self.dictionary.match(gloss)

    def _matches_to_result(
        self, source_text: str, simplified_sentence: str, sign_glosses: list[str], compose_video: bool
    ) -> dict:
        matches = [self._match_gloss(gloss) for gloss in sign_glosses]
        result = {
            "source_text": source_text,
            "simplified_sentence": simplified_sentence,
            "sign_glosses": sign_glosses,
            "matches": [asdict(m) for m in matches],
            "review_required": any(m.fallback for m in matches),
            "video_path": None,
        }
        if compose_video and matches:
            result["video_path"] = str(compose(matches, self.output_root / f"sign-{uuid4().hex}.mp4"))
        return result

    def answer_glosses(self, glosses: list[str], compose_video: bool = True) -> dict:
        vocab = karsl100_vocabulary()
        recognized = vocab.sanitize_glosses(glosses)
        question = self.llm.reconstruct(glosses)
        rag_query = f"{' '.join(recognized or glosses)} {question}"
        retrieved = self.rag.retrieve_karsl100(rag_query, recognized_glosses=recognized or glosses, k=4)
        context = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in retrieved)
        answer, llm_glosses = self.llm.grounded_answer_karsl100(
            question, context, list(vocab.labels), recognized or glosses
        )
        answer_glosses = vocab.sanitize_text_to_glosses(answer, llm_glosses)
        if not answer_glosses:
            answer_glosses = vocab.extract_glosses_from_text(context)
        if answer_glosses:
            answer = " ".join(answer_glosses)
        result = self.glosses_to_sign(answer_glosses, compose_video, source_text=answer)
        result.update(
            {
                "recognized_glosses": glosses,
                "reconstructed_question": question,
                "retrieved_context": [
                    {"source": chunk.source, "text": chunk.text} for chunk in retrieved
                ],
                "answer": answer,
                "answer_glosses": answer_glosses,
                "answer_mode": "karsl100_rag_llm",
            }
        )
        return result
