from __future__ import annotations

import json
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class StructuredArabic:
    sentence: str
    sign_glosses: list[str]


class LLMService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.settings.llm_api_key:
                raise RuntimeError(f"OPENAI_API_KEY was not found in {self.settings.shared_env}")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.llm_api_key)
        return self._client

    def _json(self, system: str, user: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(response.choices[0].message.content)

    def reconstruct(self, recognized_glosses: list[str]) -> str:
        data = self._json(
            "You reconstruct noisy Arabic sign glosses into one concise Modern Standard Arabic sentence. "
            "Do not add facts. Return JSON with key sentence.",
            json.dumps({"recognized_glosses": recognized_glosses}, ensure_ascii=False),
        )
        return str(data["sentence"]).strip()

    def grounded_answer(self, question: str, context: str) -> str:
        data = self._json(
            "Answer in concise Modern Standard Arabic using only the supplied context. "
            "Use at most two short sentences and about 30 Arabic words. Prioritize the direct, actionable answer. "
            "If the context is insufficient, say that clearly. Return JSON with key answer.",
            json.dumps({"question": question, "context": context}, ensure_ascii=False),
        )
        return str(data["answer"]).strip()

    def grounded_answer_karsl100(
        self, question: str, context: str, allowed_glosses: list[str], recognized_glosses: list[str]
    ) -> tuple[str, list[str]]:
        data = self._json(
            "You answer for a closed Arabic sign-language output vocabulary (KArSL-502). "
            "The context contains rules with a line «الجواب المعتمد:» — copy that answer "
            "when the recognized glosses or question match the same rule. "
            "Use ONLY the supplied context and ONLY glosses from allowed_glosses. "
            "Never say that context is insufficient. "
            "Return JSON with keys: answer (short Arabic sentence using only allowed concepts) and "
            "sign_glosses (ordered array of strings, each MUST be an exact item from allowed_glosses).",
            json.dumps(
                {
                    "question": question,
                    "recognized_glosses": recognized_glosses,
                    "context": context,
                    "allowed_glosses": allowed_glosses,
                },
                ensure_ascii=False,
            ),
        )
        answer = str(data.get("answer", "")).strip()
        glosses = [str(x).strip() for x in data.get("sign_glosses", []) if str(x).strip()]
        return answer, glosses

    def simplify_for_signing(self, sentence: str, vocabulary: list[str]) -> StructuredArabic:
        data = self._json(
            "Act as an Arabic Sign Language gloss planner, not a word-for-word Arabic translator. "
            "Reduce the sentence to a short, natural concept sequence for signing. Remove Arabic articles, "
            "pronouns, copulas, case/verb inflections, conjunctions and other grammatical filler unless they "
            "carry essential meaning. Remove concepts already implied by another selected sign. Preserve content "
            "concepts and preserve negation, question intent, important time, quantity and direction. "
            "Output 1-6 glosses only; this is a hard maximum. Select the closest semantically correct item from "
            "available_vocabulary and copy its spelling EXACTLY; map inflected forms to their dictionary lemma. "
            "Similarity is not enough: never choose a related-looking item that changes the meaning. If an "
            "essential proper name or content concept has no safe equivalent, keep at most ONE concise Arabic "
            "lemma unchanged for fingerspelling. Do not keep missing function words for fingerspelling. "
            "Return JSON with sentence (a readable summary of the planned concepts) and sign_glosses (ordered array).",
            json.dumps({"sentence": sentence, "available_vocabulary": vocabulary}, ensure_ascii=False),
        )
        return StructuredArabic(str(data.get("sentence", sentence)), [str(x) for x in data["sign_glosses"]])
