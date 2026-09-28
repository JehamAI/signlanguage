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
            "Give ONE direct, actionable sentence of at most 18 Arabic words. Do not list every possibility. "
            "For safety-related questions, prioritize the clearest condition and recommended action. "
            "If the best retrieved passage contains «الإجابة المختصرة الموصى بها:», use that recommendation. "
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
            "Translate the Arabic answer into a COMPLETE Saudi Arabic Sign Language (Saudi ArSL) gloss sentence, "
            "not a summary and not word-for-word spoken Arabic. Use an appropriate sign-language ordering such as "
            "topic/condition first and comment/action second. Remove spoken-Arabic grammatical particles only as "
            "required by sign-language structure, but preserve every meaning-bearing concept: conditions, severity, "
            "negation, actions, destinations, people, time, quantity and direction. "
            "For EACH concept, choose the closest meaning-preserving item from available_vocabulary and copy its "
            "spelling EXACTLY. Map inflected forms to their dictionary lemma. Never choose a merely similar-looking "
            "word that changes meaning. If no safe dictionary equivalent exists, keep that concise Arabic gloss "
            "unchanged so it can be fingerspelled; there is no limit on essential missing glosses. "
            "The sentence field MUST be the full space-separated sign-gloss sentence. The sign_glosses array MUST "
            "contain EVERY gloss in that sentence, once and in exactly the same order. Normally use 3-12 glosses. "
            "Example meaning: 'if headache is severe or with fever, go to hospital and consult a doctor'. Valid "
            "Saudi-ArSL-style gloss plan: [صداع, شديد, حمى, اذهب, مستشفى, طبيب]. "
            "Return JSON with sentence and sign_glosses.",
            json.dumps({"sentence": sentence, "available_vocabulary": vocabulary}, ensure_ascii=False),
        )
        return StructuredArabic(str(data.get("sentence", sentence)), [str(x) for x in data["sign_glosses"]])
