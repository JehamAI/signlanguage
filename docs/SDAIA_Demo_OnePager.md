# وصال (Wusal) — One-Page Demo & IP Brief

**Jeham AI** · Research PoC · Patent-pending architecture · [github.com/JehamAI/signlanguage](https://github.com/JehamAI/signlanguage)

---

## What it is (30 seconds)

**وصال** is a bidirectional **Arabic sign-language conversation assistant**: the user uploads **sign video** → the system **recognizes signs**, **reconstructs Arabic**, **retrieves trusted local knowledge (RAG)**, **generates a grounded answer**, **plans Saudi ArSL glosses**, **matches a sign dictionary**, and **composes an answer video**.

> **Demo scope today:** proof-of-concept for **licensing / joint R&D** — not a deployed national service.

---

## Patent & IP (why it is defensible)

The invention is the **integrated method**, not a single model:

| Claim area | Implemented in PoC |
|------------|-------------------|
| Sign video → lexical units | KArSL-100 temporal recognizer (100 classes) + optional pause segmentation |
| Units → Arabic question | LLM reconstruction (no invented facts) |
| **Retrieval-before-generation** | Local RAG over curated Arabic knowledge (`karsl502_service_knowledge_ar.md`) |
| Grounded Arabic answer | LLM constrained to retrieved context + recommended short answers |
| Answer → **sign output** | Saudi ArSL gloss planning + KArSL-502 dictionary (463 word signs) + alphabet fallback |
| Safety | Confidence thresholds, `review_required`, lexical guard on semantic match, explicit fingerspelling |

**Package offered:** patent application / know-how, source code, demo UI, documentation, transition support.

---

## Live demo flow (5–7 minutes)

**Recommended scenario:** health / service — e.g. signs for **مريض · صداع · ألم** or single sign **هيكل عظمي**.

1. Open **http://127.0.0.1:8000** — **وصال** home (user video | assistant sign video | chat).
2. Upload one sign video (or continuous video with pauses).
3. Show **chat**: Arabic question + natural Arabic answer.
4. Open **«تتبّع المعالجة»** and walk **7 trace stages** (recognition → RAG → glosses → dictionary → video).

**Say explicitly:** *“This demonstrates the patented pipeline; accuracy and vocabulary expand in Phase 2 with national data partners.”*

---

## Architecture (one diagram in speech)

```
Sign video → [Recognizer] → Glosses → [LLM: sentence]
    → [RAG retrieve] → [LLM: grounded answer]
    → [LLM: ArSL gloss plan] → [KArSL-502 match + spell]
    → [Video composer] → Sign-language answer MP4
```

---

## Scale with SDAIA (Phase 2 — partnership narrative)

| Layer | Today (PoC) | With SDAIA-aligned data & compute |
|-------|-------------|-----------------------------------|
| **Knowledge** | Local markdown RAG | **SDAIA / gov knowledge bases** (services, health info, procedures) under PDPL + NDMO governance |
| **Input recognition** | 100 isolated medical/action signs | Retrain on **expanded Saudi corpus**: more glosses, **multi-signer**, signer-held-out evaluation |
| **Output signs** | KArSL-502 clips + alphabet | Enriched **national sign library** (licensed clips or captured with deaf-community partners) |
| **LLM** | External API (demo speed) | **On-prem Saudi Arabic LLM** — same RAG chain, data residency |
| **Personalization** | Single-model threshold | **Per-user / per-signer adaptation** (fine-tune or calibration layers) so different faces, speeds, and regional variants are recognized |

**Training story for “different persons”:** collect diverse signers under consent → landmark/temporal model (BiLSTM/Transformer) → **signer-independent test split** → continuous monitoring — standard path SDAIA can fund via innovation grants or joint lab.

---

## Ask in the room

- **Option A:** Exclusive **KSA government-sector license** + PoC handover + 6–12 month technical transition  
- **Option B:** **Joint feasibility / pilot** (SDAIA or entity under SDAIA ecosystem) retaining IP with staged licensing  
- **Option C:** **IP acquisition** (patent + code + demo) with defined data/licensing Phase 2

---

## Contacts & assets

- **Product name:** وصال (Wusal) — Arabic Sign Conversation Assistant  
- **Repo:** JehamAI/signlanguage  
- **Disclaimer:** Research prototype; not medical or legal advice; sign language requires validation with deaf users and certified interpreters.

*Document version: demo one-pager · align with pitch deck `SDAIA_Pitch_Deck.md`*
