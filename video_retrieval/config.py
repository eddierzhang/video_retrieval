#Creates configurations and defines model for the different video retrieval elements
from __future__ import annotations

import os
from getpass import getpass

OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

EMBEDDING_MODEL = "google/gemini-embedding-2"
EMBEDDING_DIM = 3072
METADATA_MODEL = "google/gemini-3.7-flash"
TRANSCRIPTION_MODEL = "openai/whisper-1"
QUERY_PLANNER_MODEL = "google/gemini-3.1-pro-preview"
FLASH_VERIFIER_MODEL = "google/gemini-3.7-flash"
PRO_VERIFIER_MODEL = "google/gemini-3.1-pro-preview"
OCR_MODEL = "google/gemini-3.7-flash"

#Gets API Key 
def get_openrouter_api_key(prompt_if_missing: bool = False) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and prompt_if_missing:
        key = getpass("OpenRouter API key: ")
        os.environ["OPENROUTER_API_KEY"] = key
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set the environment variable "
            "or call get_openrouter_api_key(prompt_if_missing=True)."
        )
    return key
