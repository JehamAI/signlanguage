from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .arabic import normalize_arabic
from .config import ROOT
from .karsl100_vocab import KArSL100Vocabulary, karsl100_vocabulary


@dataclass(frozen=True)
class QARule:
    rule_id: str
    trigger_glosses: tuple[str, ...]
    answer_glosses: tuple[str, ...]
    answer_ar: str
    question_ar: str = ""


def _norm_gloss_list(values: list[str]) -> tuple[str, ...]:
    return tuple(normalize_arabic(value) for value in values if normalize_arabic(value))


@lru_cache(maxsize=1)
def load_qa_rules(rules_path: Path | None = None) -> tuple[QARule, ...]:
    path = rules_path or (ROOT / "data" / "karsl100_qa_rules.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = karsl100_vocabulary()
    rules: list[QARule] = []
    for item in payload["rules"]:
        triggers = []
        for gloss in item["trigger_glosses"]:
            canonical = vocab.resolve(gloss)
            if not canonical:
                raise ValueError(f"Unknown trigger gloss in rule {item['id']}: {gloss}")
            triggers.append(canonical)
        answers = []
        for gloss in item["answer_glosses"]:
            canonical = vocab.resolve(gloss)
            if not canonical:
                raise ValueError(f"Unknown answer gloss in rule {item['id']}: {gloss}")
            answers.append(canonical)
        rules.append(
            QARule(
                rule_id=str(item["id"]),
                trigger_glosses=tuple(triggers),
                answer_glosses=tuple(answers),
                answer_ar=str(item.get("answer_ar") or " ".join(answers)),
                question_ar=str(item.get("question_ar") or " ".join(triggers)),
            )
        )
    return tuple(rules)


def match_rule(recognized_glosses: list[str], rules: tuple[QARule, ...] | None = None) -> QARule | None:
    rules = rules or load_qa_rules()
    vocab = karsl100_vocabulary()
    canonical = [vocab.resolve(g) for g in recognized_glosses]
    canonical = [g for g in canonical if g]
    if not canonical:
        return None
    norm_seq = _norm_gloss_list(canonical)
    for rule in rules:
        rule_seq = _norm_gloss_list(list(rule.trigger_glosses))
        if norm_seq == rule_seq:
            return rule
    if len(canonical) == 1:
        single = normalize_arabic(canonical[0])
        for rule in rules:
            if len(rule.trigger_glosses) == 1 and normalize_arabic(rule.trigger_glosses[0]) == single:
                return rule
    return None


def validate_rules(vocab: KArSL100Vocabulary | None = None) -> None:
    vocab = vocab or karsl100_vocabulary()
    for rule in load_qa_rules():
        for gloss in (*rule.trigger_glosses, *rule.answer_glosses):
            if not vocab.resolve(gloss):
                raise ValueError(f"Rule {rule.rule_id} uses gloss outside KArSL-100: {gloss}")
