"""Local model configuration. Every model runs on this machine; no API keys are needed."""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "local_data"
MODEL_CACHE_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# Loopback only: chat requests never leave this machine.
OLLAMA_URL = "http://127.0.0.1:11434"

CLIP_MODEL = "openai/clip-vit-base-patch32"
FRAME_EMBEDDING_FPS = 1.0
WHISPER_MODEL = "base"

# One multimodal model can serve every role, which avoids swapping models on the GPU.
DEFAULT_PLANNER_MODEL = "qwen3.5:4b"
DEFAULT_VISION_MODEL = "qwen3.5:4b"
DEFAULT_VERIFIER_MODEL = "qwen3.5:4b"
