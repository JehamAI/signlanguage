# Five demo query videos (combined sign uploads)

Each file is **three KArSL-502 dictionary signs** joined with **~0.7s black pauses**. Upload **one file** in وصال with **«فيديو جملة واحد مع وقفات بين الإشارات»** enabled.

| # | File | Intended glosses (left → right) | Scenario |
|---|------|----------------------------------|----------|
| 1 | `01-airport-medical.mp4` | مريض / مرض · ألم · يساعد | Medical help at the airport |
| 2 | `02-airport-luggage.mp4` | حقيبة سفر · ثقيل · يساعد | Heavy luggage assistance |
| 3 | `03-library-access.mp4` | إعاقة سمعية · مترجم لغة الإشارة · يساعد | Sign-language interpreter at the library |
| 4 | `04-restroom-directions.mp4` | دورة مياه (حمام) · قريب · هنا | Nearby restroom |
| 5 | `05-headache-fever.mp4` | صداع · حمى · دواء | Headache and fever / medication |

Regenerate from dictionary sources:

```powershell
cd C:\Users\Jeham\signlanguage
$env:PYTHONPATH='C:\Users\Jeham\signlanguage'
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\build_paper_examples.py --skip-llm
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\prepare_demo_queries.py
```

Full recognition + RAG + answer video:

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\build_paper_examples.py
```

Results: `paper_examples/results.json`, answer clips in `paper_examples/output/`.
