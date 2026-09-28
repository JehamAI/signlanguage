from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, Settings
from .dictionary import SignDictionary
from .llm import LLMService
from .pipeline import ConversationPipeline
from .rag import LocalRAG
from .recognizer import VisualRecognizer
from .pretrained_recognizer import pretrained_recognizer
from .karsl_recognizer import karsl100_recognizer
from .embeddings import backend_name
from .segmenter import split_paused_signs


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    compose_video: bool = True


class GlossRequest(BaseModel):
    glosses: list[str] = Field(min_length=1, max_length=100)
    compose_video: bool = True


@lru_cache(maxsize=1)
def pipeline() -> ConversationPipeline:
    settings = Settings.load()
    dictionary = SignDictionary.scan(
        settings.dictionary_root, settings.video_root, settings.karsl_dictionary_root
    )
    return ConversationPipeline(dictionary, LocalRAG(settings.knowledge_root), LLMService(settings), settings.output_root)


app = FastAPI(title="Arabic Sign Conversation", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    settings = Settings.load()
    dictionary = pipeline().dictionary
    return {
        "status": "ok",
        "dictionary_words": len(dictionary.entries),
        "embedding_model": backend_name(),
        "llm_model": settings.llm_model,
        "llm_credentials_available": bool(settings.llm_api_key),
        "pretrained_word_model_available": (
            settings.artifact_root / "pretrained" / "sign_word_t5_classifier_best_3d.pth"
        ).exists(),
        "pretrained_word_vocabulary_size": 10,
        "karsl100_model_available": bool(
            list((ROOT / "references" / "karsl_word_recognition").glob("*3_signers*Accuracy_*.h5"))
        ),
        "karsl100_vocabulary_size": 100,
    }


@app.get("/api/vocabulary")
def vocabulary():
    return [{"gloss": e.gloss, "category": e.category, "media_type": e.media_type} for e in pipeline().dictionary.entries]


@app.get("/api/recognition-vocabulary")
def recognition_vocabulary():
    """Words recognized from input video; this differs from output media availability."""
    import json

    manifest = json.loads((ROOT / "data" / "karsl100_manifest.json").read_text(encoding="utf-8"))
    return sorted(manifest["classes"], key=lambda item: item["class_index"])


@app.post("/api/text-to-sign")
def text_to_sign(request: TextRequest):
    try:
        return pipeline().text_to_sign(request.text, request.compose_video)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/conversation")
def conversation(request: GlossRequest):
    try:
        return pipeline().answer_glosses(request.glosses, request.compose_video)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported image or video type")
    upload_root = ROOT / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    path = upload_root / f"{uuid4().hex}{suffix}"
    try:
        with path.open("wb") as stream:
            shutil.copyfileobj(file.file, stream)
        recognizer = VisualRecognizer(Settings.load().artifact_root / "visual_baseline.joblib")
        if suffix in {".jpg", ".jpeg", ".png"}:
            gloss, confidence = recognizer.predict_image(path)
        else:
            gloss, confidence = recognizer.predict_video(path)
        accepted = confidence >= Settings.load().visual_threshold
        return {
            "gloss": gloss if accepted else None,
            "candidate_gloss": gloss,
            "confidence": confidence,
            "accepted": accepted,
            "review_required": not accepted,
            "model": "visual_baseline",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/sign-conversation")
def sign_conversation(
    files: list[UploadFile] = File(...),
    allow_low_confidence: bool = False,
    continuous_video: bool = False,
):
    """Run the patent flow from ordered isolated-sign clips through the generated sign video."""
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 isolated-sign clips")
    upload_root = ROOT / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    temporary_directories: list[Path] = []
    predictions = []
    try:
        # The KArSL model is temporal and covers 100 Arabic word/phrase classes.
        # Keep the 10-word model available as a lightweight legacy fallback.
        try:
            recognizer = karsl100_recognizer()
            model_name = "karsl100_bilstm"
        except (FileNotFoundError, ImportError, ValueError):
            recognizer = pretrained_recognizer()
            model_name = "legacy_10_word_t5"
        original_names: list[str] = []
        for file in files:
            suffix = Path(file.filename or "sign.mp4").suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                raise HTTPException(status_code=400, detail="Unsupported sign video")
            path = upload_root / f"{uuid4().hex}{suffix}"
            paths.append(path)
            original_names.append(file.filename or "sign.mp4")
            with path.open("wb") as stream:
                shutil.copyfileobj(file.file, stream)

        recognition_paths = list(paths)
        boundaries = []
        if continuous_video:
            if len(paths) != 1:
                raise HTTPException(status_code=400, detail="Continuous mode accepts exactly one sentence video")
            segment_dir = upload_root / f"segments-{uuid4().hex}"
            temporary_directories.append(segment_dir)
            segments = split_paused_signs(paths[0], segment_dir)
            recognition_paths = [segment.path for segment in segments]
            paths.extend(recognition_paths)
            boundaries = [
                {"start_seconds": item.start_seconds, "end_seconds": item.end_seconds}
                for item in segments
            ]
            if not recognition_paths:
                raise HTTPException(status_code=422, detail="No paused sign segments were detected")

        for position, path in enumerate(recognition_paths):
            prediction = recognizer.predict_video(path)
            predictions.append(
                {
                    "position": position,
                    "filename": path.name if continuous_video else original_names[position],
                    "gloss": prediction.gloss or None,
                    "confidence": prediction.confidence,
                    "accepted": prediction.accepted,
                    "frames_with_hands": prediction.frames_with_hands,
                    "sampled_frames": prediction.sampled_frames,
                    "model": model_name,
                }
            )
        rejected = [item for item in predictions if not item["accepted"]]
        if rejected and not allow_low_confidence:
            return {
                "status": "review_required",
                "stage": "recognition",
                "predictions": predictions,
                "message": "One or more clips were below the recognition threshold; the LLM was not called.",
            }
        glosses = [str(item["gloss"]) for item in predictions if item["gloss"]]
        if not glosses:
            raise HTTPException(status_code=422, detail="No hand signs were recognized")
        result = pipeline().answer_glosses(glosses, compose_video=True)
        result.update(
            {
                "status": "completed",
                "recognition": predictions,
                "segmentation": boundaries,
                "pipeline_stages": [
                    "sign_video_to_words",
                    "words_to_sentence",
                    "rag_retrieval",
                    "grounded_answer",
                    "answer_to_sign_glosses",
                    "dictionary_matching",
                    "sign_video_construction",
                ],
            }
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        for directory in temporary_directories:
            shutil.rmtree(directory, ignore_errors=True)


@app.get("/api/video/{filename}")
def video(filename: str):
    path = (Settings.load().output_root / filename).resolve()
    if path.parent != Settings.load().output_root.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
