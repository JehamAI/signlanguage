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
            "If the context is insufficient, say that clearly. Return JSON with key answer.",
            json.dumps({"question": question, "context": context}, ensure_ascii=False),
        )
        return str(data["answer"]).strip()

    def grounded_answer_karsl100(
        self, question: str, context: str, allowed_glosses: list[str], recognized_glosses: list[str]
    ) -> tuple[str, list[str]]:
        data = self._json(
            "You answer for a closed Arabic sign-language vocabulary (KArSL-100). "
            "The context contains rules with a line «الجواب المعتمد (KArSL-100 فقط):» — copy that answer "
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
            "Convert the Arabic sentence into a short ordered sequence of sign glosses. Preserve meaning; "
            "use a supplied vocabulary item only when it has the same meaning. Never replace a missing concept "
            "with an unrelated available word. Keep a missing word unchanged so the system can flag it for "
            "fingerspelling or human review. Omit only function words that do not change meaning. "
            "Return JSON with sentence and sign_glosses (an array of strings).",
            json.dumps({"sentence": sentence, "available_vocabulary": vocabulary}, ensure_ascii=False),
        )
        return StructuredArabic(str(data.get("sentence", sentence)), [str(x) for x in data["sign_glosses"]])
