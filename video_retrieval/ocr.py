#Extracting text from video frames
from __future__ import annotations

from .visual_text import (
    consensus_ocr_text,
    normalize_ocr_text,
    run_visual_text_extraction,
)

#Runs visual OCR extraction depending on the query and the target object and region specified by the user. Returns the extracted text from the video frames.
def run_visual_ocr(
    manifest: dict,
    query: str,
    *,
    ocr_target: str = "visible text-bearing region",
    ocr_instruction: str = "Read the visible text relevant to the user query exactly.",
    **kwargs,
):
    return run_visual_text_extraction(
        manifest,
        query,
        target_object="text-bearing object relevant to the user query",
        target_region=ocr_target,
        text_description="visible text requested by the user",
        extraction_instruction=ocr_instruction,
        **kwargs,
    )

__all__ = [
    "normalize_ocr_text",
    "consensus_ocr_text",
    "run_visual_text_extraction",
    "run_visual_ocr",
    "run_license_plate_ocr",
]
