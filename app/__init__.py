"""Arabic sign-language conversation prototype."""

import os

# Both the sentence embedder and sign recognizer are intentionally local at runtime.
# These must be set before either transformers or huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
