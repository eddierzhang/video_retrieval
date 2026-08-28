#Audio transcription plus semantic and BM25 transcript retrieval
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from .config import OPENROUTER_TRANSCRIPTION_URL, TRANSCRIPTION_MODEL, get_openrouter_api_key
from .embeddings import embed_text, normalize_embedding

#Separates audio into multiple overlapping chunks for transcription, removes overlapping areas after transcription
def create_audio_chunks(
    manifest,
    output_dir="transcript/audio_chunks",
    window=120.0,
    stride=110.0,
    bitrate="64k",
):

    source_video = Path(
        manifest["video"]["path"]
    )

    duration = float(
        manifest["video"]["duration"]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlap = window - stride

    chunks = []

    start = 0.0
    index = 0

    while start < duration:

        end = min(
            start + window,
            duration
        )

        chunk_id = f"audio_{index:06d}"

        output_path = (
            output_dir /
            f"{chunk_id}.mp3"
        )

        if index == 0:
            keep_start = start
        else:
            keep_start = start + overlap / 2

        if end >= duration:
            keep_end = end
        else:
            keep_end = end - overlap / 2

        if not output_path.exists():

            command = [
                "ffmpeg",
                "-y",

                "-ss", str(start),
                "-i", str(source_video),

                "-t", str(end - start),

                # no video
                "-vn",

                # mono
                "-ac", "1",

                # speech-friendly sample rate
                "-ar", "16000",

                "-c:a", "libmp3lame",
                "-b:a", bitrate,

                str(output_path),
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.decode(
                        "utf-8",
                        errors="ignore"
                    )
                )

        chunks.append({
            "chunk_id": chunk_id,

            "start": start,
            "end": end,

            "keep_start": keep_start,
            "keep_end": keep_end,

            "path": str(output_path),
        })

        if end >= duration:
            break

        start += stride
        index += 1

    return chunks

def audio_to_base64(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")

#Transcribes a single audio file
def transcribe_audio_chunk(
    audio_path,
    language=None,
    word_timestamps=True,
):
    encoded_audio = audio_to_base64(
        audio_path
    )

    payload = {
        "model": TRANSCRIPTION_MODEL,

        "input_audio": {
            "data": encoded_audio,
            "format": "mp3",
        },

        "response_format": "verbose_json",

        "temperature": 0,
    }

    if word_timestamps:
        payload["timestamp_granularities"] = [
            "word",
            "segment",
        ]

    else:
        payload["timestamp_granularities"] = [
            "segment"
        ]

    if language is not None:
        payload["language"] = language

    headers = {
        "Authorization":
            f"Bearer {get_openrouter_api_key()}",

        "Content-Type":
            "application/json",
    }

    response = requests.post(
        OPENROUTER_TRANSCRIPTION_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if not response.ok:

        raise RuntimeError(
            "Transcription failed:\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    return response.json()

#  Convert chunk-relative timestamps into original-video timestamps. Also drops duplicated words/segments from overlapping audio chunks
def convert_to_absolute_timestamps(
    transcription,
    audio_chunk
):

    offset = audio_chunk["start"]

    keep_start = audio_chunk["keep_start"]
    keep_end = audio_chunk["keep_end"]

    words = []
    segments = []

    for word in transcription.get(
        "words",
        []
    ):

        absolute_start = (
            offset
            + float(word["start"])
        )

        absolute_end = (
            offset
            + float(word["end"])
        )

        midpoint = (
            absolute_start
            + absolute_end
        ) / 2

        if not (
            keep_start
            <= midpoint
            < keep_end
        ):
            continue

        words.append({
            "word":
                word.get("word", "").strip(),

            "start":
                absolute_start,

            "end":
                absolute_end,
        })

    for segment in transcription.get(
        "segments",
        []
    ):

        absolute_start = (
            offset
            + float(segment["start"])
        )

        absolute_end = (
            offset
            + float(segment["end"])
        )

        midpoint = (
            absolute_start
            + absolute_end
        ) / 2

        if not (
            keep_start
            <= midpoint
            < keep_end
        ):
            continue

        segments.append({
            "text":
                segment.get(
                    "text",
                    ""
                ).strip(),

            "start":
                absolute_start,

            "end":
                absolute_end,
        })

    return words, segments

#Transcribe the audio in a video 
def transcribe_video(
    manifest,

    output_dir="transcript",

    audio_window=120.0,
    audio_stride=110.0,

    language=None,

    retry_count=3,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = (
        output_dir /
        "raw_transcriptions.jsonl"
    )

    audio_chunks = create_audio_chunks(
        manifest,

        output_dir=(
            output_dir /
            "audio_chunks"
        ),

        window=audio_window,
        stride=audio_stride,
    )

    completed = {}

    if raw_path.exists():

        with open(
            raw_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                try:
                    record = json.loads(
                        line
                    )
                    completed[
                        record["chunk_id"]
                    ] = record
                except Exception:
                    pass
    print(
        f"{len(completed)} audio chunks already transcribed."
    )
    for audio_chunk in tqdm(
        audio_chunks,
        desc="Transcribing audio"
    ):
        chunk_id = audio_chunk[
            "chunk_id"
        ]
        if chunk_id in completed:
            continue
        success = False
        for attempt in range(
            retry_count
        ):
            try:

                result = transcribe_audio_chunk(
                    audio_chunk["path"],
                    language=language,
                )

                record = {
                    "chunk_id":
                        chunk_id,

                    "start":
                        audio_chunk["start"],

                    "end":
                        audio_chunk["end"],

                    "keep_start":
                        audio_chunk["keep_start"],

                    "keep_end":
                        audio_chunk["keep_end"],

                    "transcription":
                        result,
                }

                success = True
                break

            except Exception as e:

                print(
                    f"\n{chunk_id} "
                    f"attempt {attempt + 1} failed:\n"
                    f"{e}"
                )

                if attempt < retry_count - 1:
                    time.sleep(
                        2 ** attempt
                    )

        if not success:

            print(
                "Skipping",
                chunk_id
            )

            continue

        # Immediately save
        with open(
            raw_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(record)
                + "\n"
            )

    #Segments and words determiend by transcription model 
    all_words = []
    all_segments = []

    with open(
        raw_path,
        "r",
        encoding="utf-8"
    ) as f:

        records = [
            json.loads(line)
            for line in f
        ]

    records.sort(
        key=lambda x: x["start"]
    )

    for record in records:

        words, segments = (
            convert_to_absolute_timestamps(
                record["transcription"],
                record
            )
        )

        all_words.extend(words)
        all_segments.extend(segments)

    all_words.sort(
        key=lambda x: x["start"]
    )

    all_segments.sort(
        key=lambda x: x["start"]
    )

    with open(
        output_dir /
        "words.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_words,
            f,
            indent=2
        )

    with open(
        output_dir /
        "segments.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_segments,
            f,
            indent=2
        )

    return all_words, all_segments

#Build overlapping timestamped transcript windows (in accordance with video chunks).
def create_transcript_windows(
    words,
    video_duration,
    window=30.0,
    stride=15.0,
):

    windows = []

    start = 0.0
    index = 0

    while start < video_duration:

        end = min(
            start + window,
            video_duration
        )

        selected_words = [
            word
            for word in words
            if (
                word["start"] < end
                and word["end"] > start
            )
        ]

        text = " ".join(
            word["word"]
            for word in selected_words
        ).strip()

        windows.append({
            "chunk_id":
                f"transcript_{index:06d}",

            "start":
                start,

            "end":
                end,

            "text":
                text,
        })

        if end >= video_duration:
            break

        start += stride
        index += 1

    return windows

#Embed transcript windows with Gemini Embedding 2 
def embed_transcript_windows(
    windows,

    save_dir="transcript_index",

    retry_count=3,
):

    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = []
    metadata = []

    for window in tqdm(
        windows,
        desc="Embedding transcript"
    ):

        success = False

        for attempt in range(
            retry_count
        ):

            try:

                vector = embed_text(
                    window["text"]
                )

                vector = (
                    normalize_embedding(
                        vector
                    )
                )

                success = True
                break

            except Exception as e:

                print(
                    f"\nEmbedding failed for "
                    f"{window['chunk_id']}: "
                    f"{e}"
                )

                if attempt < retry_count - 1:

                    time.sleep(
                        2 ** attempt
                    )

        if not success:
            continue

        embeddings.append(
            vector
        )

        metadata.append(
            window.copy()
        )

    embeddings = np.vstack(
        embeddings
    ).astype(
        np.float32
    )

    np.save(
        save_dir /
        "transcript_embeddings.npy",
        embeddings
    )

    with open(
        save_dir /
        "transcript_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return embeddings, metadata

#Build a FAISS index for the transcript embeddings and save it to disk.
def build_transcript_faiss(
    embeddings,

    save_path=(
        "transcript_index/"
        "transcript.faiss"
    ),
):

    import faiss

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(save_path)
    )

    print(
        f"Indexed {index.ntotal} "
        "transcript windows."
    )

    return index

def tokenize_text(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )

#Build a BM25 index for the transcript metadata 
def build_bm25(
    transcript_metadata
):

    tokenized_documents = [
        tokenize_text(
            item["text"]
        )
        for item
        in transcript_metadata
    ]

    from rank_bm25 import BM25Okapi

    return BM25Okapi(
        tokenized_documents
    )
#Search transcript metadata using semantic (FAISS) and BM25 methods. 
def search_transcript_semantic(
    query,

    index,
    metadata,

    top_k=10,
):

    query_embedding = (
        normalize_embedding(
            embed_text(query)
        )
    )

    scores, ids = index.search(
        query_embedding[
            None, :
        ].astype(
            np.float32
        ),

        min(
            top_k,
            len(metadata)
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        item = metadata[
            idx
        ].copy()

        item["score"] = float(
            score
        )

        results.append(
            item
        )

    return results

def search_transcript_bm25(
    query,

    bm25,
    metadata,

    top_k=10,
):

    query_tokens = tokenize_text(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_ids = np.argsort(
        scores
    )[::-1][
        :top_k
    ]

    results = []

    for idx in top_ids:

        item = metadata[
            idx
        ].copy()

        item["score"] = float(
            scores[idx]
        )

        results.append(
            item
        )

    return results

#Combine semantic with BM25 Seaches
def search_transcript(
    query,

    semantic_index,
    bm25_index,
    metadata,

    semantic_top_k=30,
    bm25_top_k=30,
):

    semantic = (
        search_transcript_semantic(
            query,

            semantic_index,
            metadata,

            semantic_top_k
        )
    )

    bm25 = (
        search_transcript_bm25(
            query,

            bm25_index,
            metadata,

            bm25_top_k
        )
    )

    return {
        "semantic": semantic,
        "bm25": bm25,
    }


def load_transcript_index(save_dir="transcript_index"):
    """Load transcript FAISS metadata and rebuild BM25."""
    import faiss
    save_dir = Path(save_dir)
    with open(save_dir / "transcript_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    index_path = save_dir / "transcript.faiss"
    if index_path.exists():
        semantic_index = faiss.read_index(str(index_path))
    else:
        embeddings = np.load(save_dir / "transcript_embeddings.npy").astype(np.float32)
        semantic_index = build_transcript_faiss(embeddings, save_path=index_path)
    bm25 = build_bm25(metadata)
    return semantic_index, bm25, metadata

