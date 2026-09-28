# Arabic Sign Conversation Prototype

This project implements the core flow described in `Utility_SL.pdf`: sign recognition, word-to-sentence reconstruction, grounded response generation, sentence simplification, local semantic matching to a sign dictionary, fallback handling, and sign-video construction.

## What works

- Uses the already-installed `C:\Users\Jeham\gpu-env` environment and its cached `sentence-transformers/all-MiniLM-L6-v2` model (384 dimensions).
- Reads the LLM key and model at runtime from the existing Evalia `.env`; it never copies or logs the secret.
- Indexes Arabic word labels from image or video dictionary folders.
- Performs exact Arabic-normalized matching first and local embedding similarity second.
- Retrieves local knowledge before asking the LLM to answer.
- Composes matched images and clips into an MP4.
- Flags unmapped words for fingerspelling/human review rather than silently substituting a weak semantic neighbor.
- Provides a FastAPI service and Arabic right-to-left browser UI.

## Run

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m pytest -q
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m scripts.build_dictionary
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m scripts.train_visual_baseline --max-per-class 200
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The main sign-video route now uses the included temporal KArSL-100 BiLSTM checkpoint. It
recognizes 100 Arabic word/phrase classes (SignIDs 0071-0170), reconstructs the question,
runs local RAG, generates a grounded answer, matches available output signs, and builds an
H.264 response video.

Test it with a real KArSL sequence that is already prepared as an uploadable MP4:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/sign-conversation" -F "files=@test_inputs/karsl_0071.mp4;type=video/mp4"
```

The full input vocabulary is available at `http://127.0.0.1:8000/api/recognition-vocabulary`.
Input recognition and output-video vocabulary are separate: recognizing a word does not mean
that a licensed response clip for that word is available.

To retrain with a whole signer held out for honest evaluation:

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\prepare_karsl100.py
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\extract_karsl_landmarks.py
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' scripts\train_karsl100.py --test-signer 3
```

The pretrained checkpoint is stored at `artifacts/pretrained/sign_word_t5_classifier_best_3d.pth`. If it is missing, download it with:

```powershell
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -c "from huggingface_hub import hf_hub_download; hf_hub_download('FatimahEmadEldin/ArSL-Models','sign_word_t5_classifier_best_3d.pth',local_dir=r'artifacts\pretrained')"
```

## API

- `GET /api/health` reports model and dictionary readiness without exposing credentials.
- `GET /api/vocabulary` lists available sign glosses.
- `GET /api/recognition-vocabulary` lists all 100 KArSL input classes.
- `POST /api/text-to-sign` converts Arabic text into mapped signs and optionally an MP4.
- `POST /api/recognize` accepts one isolated sign image/video and returns a visual-baseline prediction.
- `POST /api/conversation` reconstructs a question from recognized glosses, retrieves local context, generates an answer, and converts it back to signs.
- `POST /api/sign-conversation` accepts an ordered list of isolated sign-word clips and executes the full patent workflow through the answer video.

### Pretrained input vocabulary

The MIT pretrained recognizer supports ten labels: `ينام`, `يسكت`, `حب`, `يدخن`, `دعم`, `مرتبك`, `قلق`, `هنا`, `السلام عليكم`, and `شكرا`. Upload one isolated clip per word, in sentence order. It is not a continuous-sign segmenter.

The browser displays every pipeline stage: recognition confidence, reconstructed Arabic sentence, retrieved RAG passages, grounded answer, output glosses, and the generated H.264 sign video. The first recognition request may take longer while the T5 checkpoint loads; later requests reuse it in memory.

### Included upload test clip

`test_inputs/يسكت_demo.mp4` is a four-second H.264 synthetic clip assembled from the ARSLW `اصمت` still-image class. The pretrained recognizer returns `يسكت` at approximately 0.50 confidence. Use it to exercise the full upload-to-RAG-to-answer-video path. It is a test fixture, not a natural continuous signing sample. Rebuild it with `python -m scripts.create_demo_clip`.

## Patent-oriented extensions

The implementation adds safeguards that the filing describes only broadly:

1. Arabic orthographic normalization before exact or semantic matching.
2. A similarity threshold with explicit fallback instead of unconditional nearest-neighbor replacement.
3. Provenance and license tracking for every dataset.
4. Grounded generation constrained to locally retrieved material.
5. A `review_required` signal for missing signs and weak mappings.
6. Separate visual-recognition and language-embedding models, preventing the text embedder from being misused as a gesture recognizer.

## Limitations before deployment

- Sign languages are not word-for-word spoken Arabic. A native Deaf signer or certified interpreter must validate gloss order, regional dialect, facial grammar, fingerspelling, and every dictionary clip.
- The included visual classifier is a fast appearance baseline for integration testing, not a production recognizer. A bounded 40-samples-per-class run reached only 15% on its held-out frame split, which is evidence that appearance-only HOG is inadequate. Train a temporal hand/pose/face model and evaluate with signer-independent splits.
- Visual predictions below `SIGN_VISUAL_THRESHOLD` (default 0.60) are rejected and marked for review instead of being fed into the conversation.
- Semantic dictionary matches below `SIGN_SEMANTIC_THRESHOLD` (default 0.88) are rejected; this prevents unrelated available signs from replacing missing answer concepts.
- The Zenodo archive does not match its published sample count; see `DATASETS.md`.
- The ARSLW frames are useful for dictionary and image classification tests but do not validate continuous signing or motion.
- Do not claim accessibility, accuracy, or dialect coverage until measured with representative Deaf users.
