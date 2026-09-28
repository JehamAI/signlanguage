from __future__ import annotations

import json
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class StructuredArabic:
    sentence: str
    sign_glosses: list[str]
    dictionary_mappings: list[dict]


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
        plan = self._json(
            "Translate the Arabic answer into a COMPLETE Saudi Arabic Sign Language (Saudi ArSL) gloss sentence, "
            "not a summary and not word-for-word spoken Arabic. Use an appropriate sign-language ordering such as "
            "topic/condition first and comment/action second. Remove spoken-Arabic grammatical particles only as "
            "required by sign-language structure, but preserve every meaning-bearing concept: conditions, severity, "
            "negation, actions, destinations, people, time, quantity and direction. Do not consult or anticipate a "
            "dictionary in this step. Produce concise semantic gloss concepts that preserve the full answer. "
            "The sentence field MUST be the full space-separated sign-gloss sentence. The sign_glosses array MUST "
            "contain EVERY gloss in that sentence, once and in exactly the same order. Normally use 3-12 glosses. "
            "Example meaning: 'if headache is severe or with fever, go to hospital and consult a doctor'. Valid "
            "Saudi-ArSL-style gloss plan: [صداع, شديد, حمى, اذهب, مستشفى, طبيب]. "
            "Return JSON with sentence and sign_glosses.",
            json.dumps({"sentence": sentence}, ensure_ascii=False),
        )
        concepts = [str(x).strip() for x in plan.get("sign_glosses", []) if str(x).strip()]
        mapping = self._json(
            "Map each Saudi-ArSL semantic concept to a closed sign dictionary. This is semantic equivalence, not "
            "spelling similarity. For each input concept, select dictionary_gloss only when it preserves the same "
            "meaning in this sentence; copy it EXACTLY from available_vocabulary. Inflection-to-lemma and true "
            "synonyms are allowed. Related but different concepts are forbidden. If no safe equivalent exists, set "
            "dictionary_gloss to null so the system can fingerspell the original concept. Never omit a concept and "
            "preserve input order. Return JSON with mappings, an array of objects containing concept, "
            "dictionary_gloss, relation (exact, lemma, synonym, or none), confidence from 0 to 1, and a short reason.",
            json.dumps(
                {"sentence": sentence, "concepts": concepts, "available_vocabulary": vocabulary},
                ensure_ascii=False,
            ),
        )
        allowed = set(vocabulary)
        raw_mappings = mapping.get("mappings", [])
        safe_glosses: list[str] = []
        audited: list[dict] = []
        for index, concept in enumerate(concepts):
            item = raw_mappings[index] if index < len(raw_mappings) and isinstance(raw_mappings[index], dict) else {}
            candidate = item.get("dictionary_gloss")
            confidence = float(item.get("confidence", 0.0) or 0.0)
            accepted = isinstance(candidate, str) and candidate in allowed and confidence >= 0.85
            selected = candidate if accepted else concept
            safe_glosses.append(selected)
            audited.append(
                {
                    "concept": concept,
                    "dictionary_gloss": candidate if isinstance(candidate, str) else None,
                    "relation": str(item.get("relation", "none")),
                    "confidence": confidence,
                    "reason": str(item.get("reason", "")),
                    "accepted": accepted,
                    "selected_gloss": selected,
                }
            )
        if any(not item["accepted"] for item in audited):
            refinement = self._json(
                "Minimally revise a Saudi-ArSL gloss plan after closed-dictionary lookup. You have the complete "
                "original Arabic answer, its semantic concepts, the first lookup audit, and the full dictionary. "
                "You may reorder the glosses or replace a missing concept with a short meaning-equivalent expression "
                "using one or more exact dictionary items. Preserve every condition, negation, action, destination, "
                "person, time, quantity, direction, and safety qualification. Do not add advice or broaden meaning. "
                "If a safe equivalent is uncertain, retain the original concise concept unchanged for fingerspelling. "
                "Return JSON with sign_glosses, covered_concepts, omitted_concepts, confidence, and reason. Every item "
                "in sign_glosses must be either copied EXACTLY from available_vocabulary or copied EXACTLY from the "
                "original concepts.",
                json.dumps(
                    {
                        "answer": sentence,
                        "original_concepts": concepts,
                        "first_lookup": audited,
                        "available_vocabulary": vocabulary,
                    },
                    ensure_ascii=False,
                ),
            )
            revised = [str(x).strip() for x in refinement.get("sign_glosses", []) if str(x).strip()]
            permitted = allowed | set(concepts)
            omitted = [str(x) for x in refinement.get("omitted_concepts", []) if str(x).strip()]
            confidence = float(refinement.get("confidence", 0.0) or 0.0)
            refinement_accepted = bool(revised) and all(x in permitted for x in revised) and not omitted and confidence >= 0.85
            audited.append(
                {
                    "stage": "minimal_sentence_refinement",
                    "accepted": refinement_accepted,
                    "confidence": confidence,
                    "covered_concepts": refinement.get("covered_concepts", []),
                    "omitted_concepts": omitted,
                    "reason": str(refinement.get("reason", "")),
                    "selected_glosses": revised,
                }
            )
            if refinement_accepted:
                safe_glosses = revised
        return StructuredArabic(" ".join(safe_glosses), safe_glosses, audited)
