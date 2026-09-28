# End-to-end paper examples

These artifacts evaluate the controlled proof-of-concept pipeline:

`three paused KArSL-100 signs → segmentation → recognition → Arabic reconstruction → local RAG → grounded Arabic answer → Saudi-ArSL gloss planning → full KArSL-502 lookup → conservative nearest-neighbor fallback → Arabic fingerspelling → answer video`

## Test scope

- Five curated input sentence videos.
- Three isolated KArSL signs per video, joined with a 0.70-second neutral pause.
- Fifteen input sign tokens in total.
- The 15/15 result below is a functional result on these selected sequences. It is not a signer-independent accuracy claim.
- KArSL-502 contains 463 non-alphabet output classes plus 39 alphabet/orthographic classes used for fallback spelling.

## Results

| Example | Recognized input glosses | Grounded RAG answer | Saudi-ArSL output plan | Direct dictionary concepts | Spelled concepts | Output |
|---|---|---|---|---:|---:|---|
| Airport medical help | مريض / مرض، ألم، دواء | موظف المطار يوجه المريض إلى الصيدلية أو الإسعافات الأولية. | موظف-مطار، مريض / مرض، يوجّه، صيدلية، أو، إسعافات أولية | 3/6 | 3 | `output/airport-medical-answer.mp4` |
| Airport directions | يصعد، ينزل، يفتح | موظف المطار يحدد الباب؛ اتجه يميناً أو يساراً، ثم اصعد أو انزل. | موظف، مطار، يحدد، بوابة، اتجه، يمين، أو، يسار، ثم، يصعد، أو، ينزل | 5/12 | 7 | `output/airport-directions-answer.mp4` |
| Library accessibility | يسمع، يسكت، يفتح | موظف المكتبة يوفر مترجم لغة الإشارة أو فيديو للتواصل. | مكتبة، موظف، يوفر، مترجم لغة الإشارة، أو، فيديو، تواصل | 3/7 | 4 | `output/library-access-answer.mp4` |
| Headache | مريض / مرض، صداع، ألم | الصداع مع الحمى يحتاج طبيباً أو مستشفى. | صداع، مع، حمى، يحتاج، طبيب، أو، مستشفى | 4/7 | 3 | `output/headache-answer.mp4` |
| Diabetes | مرض السكر / سكري، دواء، مريض / مرض | مريض السكري يحتاج تحليل دم ودواء ومتابعة طبيب. | مريض / مرض، مرض السكر / سكري، تحليل دم، دواء، متابعة، طبيب، يحتاج | 5/7 | 2 | `output/diabetes-answer.mp4` |

All five videos were segmented into three signs and all 15 sign tokens matched the expected label and order. Across the 39 output concepts, 20 used direct sign videos and 19 used alphabet fallback. No semantic nearest-neighbor substitution passed the conservative safety threshold in this run; therefore the system spelled missing concepts instead of inserting a weakly related sign.

## Interpretation

The examples demonstrate end-to-end execution and auditable fallback behavior. They do not establish general recognition accuracy or Saudi-ArSL linguistic validity. A publishable evaluation should additionally use held-out signers, uncurated recordings, Deaf-community review of the gloss plans, and human scoring of semantic preservation and sign naturalness.

## Reproduction

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m scripts.build_paper_examples
```

To rerun RAG, LLM mapping, and answer-video generation without repeating segmentation and recognition:

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m scripts.build_paper_examples --answers-only
```

Machine-readable results are stored in `results.json`. The complete RAG source document is `../knowledge/karsl502_service_knowledge_ar.md`.
