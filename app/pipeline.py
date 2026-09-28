from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from .composer import compose
from .dictionary import AlphabetDictionary, Match, SignDictionary
from .embeddings import embed
from .karsl100_vocab import karsl100_vocabulary, karsl502_vocabulary
from .llm import LLMService
from .rag import LocalRAG


class ConversationPipeline:
    def __init__(self, dictionary: SignDictionary, rag: LocalRAG, llm: LLMService, output_root: Path,
                 alphabet: AlphabetDictionary | None = None):
        self.dictionary = dictionary
        self.rag = rag
        self.llm = llm
        self.output_root = output_root
        self.alphabet = alphabet or AlphabetDictionary({})
        self._output_labels: tuple[str, ...] | None = None
        self._output_vectors = None

    def _sign_candidates(self, text: str, limit: int = 60) -> list[str]:
        vocab = karsl502_vocabulary()
        if self._output_vectors is None:
            self._output_labels = vocab.labels
            self._output_vectors = embed(list(vocab.labels))
        scores = self._output_vectors @ embed(text)[0]
        indexes = np.argsort(scores)[::-1][:limit]
        candidates = [vocab.labels[int(index)] for index in indexes]
        # Always retain labels stated literally in the answer.
        for label in vocab.extract_glosses_from_text(text):
            if label not in candidates:
                candidates.insert(0, label)
        return candidates

    def text_to_sign(self, text: str, compose_video: bool = True) -> dict:
        simplified = self.llm.simplify_for_signing(text, [e.gloss for e in self.dictionary.entries])
        return self._matches_to_result(text, simplified.sentence, simplified.sign_glosses, compose_video)

    def glosses_to_sign(self, glosses: list[str], compose_video: bool = True, source_text: str = "") -> dict:
        vocab = karsl502_vocabulary()
        # Preserve unavailable concepts: _match_gloss will try semantic reuse,
        # then the alphabet fallback will spell them instead of silently dropping them.
        resolved = [vocab.resolve(item) or item.strip() for item in glosses if item.strip()]
        sentence = source_text.strip() or " ".join(resolved)
        return self._matches_to_result(sentence, sentence, resolved, compose_video)

    def _match_gloss(self, gloss: str) -> Match:
        vocab = karsl502_vocabulary()
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
        word_matches = [self._match_gloss(gloss) for gloss in sign_glosses]
        matches: list[Match] = []
        for match in word_matches:
            if match.media_path:
                matches.append(match)
            else:
                spelled = self.alphabet.spell(match.query)
                matches.extend(spelled or [match])
        result = {
            "source_text": source_text,
            "simplified_sentence": simplified_sentence,
            "sign_glosses": sign_glosses,
            "matches": [asdict(m) for m in matches],
            "word_matches": [asdict(m) for m in word_matches],
            "review_required": any(m.fallback for m in matches),
            "video_path": None,
        }
        if compose_video and matches:
            result["video_path"] = str(compose(matches, self.output_root / f"sign-{uuid4().hex}.mp4"))
        return result

    def answer_glosses(self, glosses: list[str], compose_video: bool = True) -> dict:
        input_vocab = karsl100_vocabulary()
        output_vocab = karsl502_vocabulary()
        recognized = input_vocab.sanitize_glosses(glosses)
        question = self.llm.reconstruct(glosses)
        rag_query = f"{' '.join(recognized or glosses)} {question}"
        retrieved = self.rag.retrieve_karsl100(rag_query, recognized_glosses=recognized or glosses, k=4)
        context = "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in retrieved)
        answer = self.llm.grounded_answer(question, context)
        if retrieved:
            recommended = next(
                (line for line in retrieved[0].text.splitlines() if line.startswith("الإجابة المختصرة الموصى بها:")),
                "",
            )
            if recommended:
                answer = recommended.split(":", 1)[1].strip()
        signing = self.llm.simplify_for_signing(answer, self._sign_candidates(answer))
        answer_glosses = signing.sign_glosses
        resolved = [output_vocab.resolve(item) or item.strip() for item in answer_glosses if item.strip()]
        sign_sentence = " ".join(resolved)
        result = self._matches_to_result(answer, sign_sentence, resolved, compose_video)
        result.update(
            {
                "recognized_glosses": glosses,
                "reconstructed_question": question,
                "retrieved_context": [
                    {"source": chunk.source, "text": chunk.text} for chunk in retrieved
                ],
                "answer": answer,
                "answer_glosses": answer_glosses,
                "sign_language_sentence": sign_sentence,
                "answer_mode": "natural_rag_answer_then_karsl502_translation",
            }
        )
        return result
