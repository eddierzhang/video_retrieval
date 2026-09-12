#Audio transcription plus semantic and BM25 transcript retrieval
from __future__ import annotations

from . import local_backend

import json
import re
import time
from pathlib import Path

import numpy as np

from .embeddings import embed_document, embed_query, normalize_embedding


#Transcribe the whole video once with local Whisper and save absolute word/segment timestamps
def transcribe_video(
    manifest,
    output_dir="transcript",
    language=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript = local_backend.transcribe_media(manifest["video"]["path"], language=language)
    words = sorted(transcript["words"], key=lambda x: x["start"])
    segments = sorted(transcript["segments"], key=lambda x: x["start"])

    with open(output_dir / "words.json", "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)

    with open(output_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    return words, segments

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

    for window in local_backend.track(windows, "Embedding transcript"):

        success = False

        for attempt in range(
            retry_count
        ):

            try:

                vector = embed_document(window["text"])

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
            embed_query(query)
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

