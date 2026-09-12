#Breaks a video down into chunks, generates metadata for each chunk, and builds a searchable index of the metadata
from __future__ import annotations

from . import local_backend

import json
import time
from pathlib import Path

import numpy as np

from .embeddings import embed_text, normalize_embedding

#Metadata to include and extract from the video
VIDEO_METADATA_SCHEMA = {
    "type": "object",

    "properties": {

        "summary": {
            "type": "string",
            "description":
                "Dense factual description of the entire video."
        },

        "scene": {
            "type": "string",
            "description":
                "Physical environment and scene context."
        },

        "people": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "objects": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "actions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "interactions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "motion_events": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "state_changes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "visible_text": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "search_terms": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "important_events": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "start_seconds": {
                        "type": "number"
                    },

                    "end_seconds": {
                        "type": "number"
                    },

                    "description": {
                        "type": "string"
                    }
                },

                "required": [
                    "start_seconds",
                    "end_seconds",
                    "description"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "summary",
        "scene",
        "people",
        "objects",
        "actions",
        "interactions",
        "motion_events",
        "state_changes",
        "visible_text",
        "search_terms",
        "important_events"
    ],

    "additionalProperties": False
}

#Caption Prompt
METADATA_PROMPT = """
Describe this passage of video so it can be found later by a text search.

You are given sampled frames in chronological order and, when there is speech,
the transcript for the same interval. The frames are sparse samples: describe
only what they actually show, and never invent what happens between them.

The video could be anything - a lecture, a cooking video, a sports broadcast, a
screen recording, an interview, a home video, a fixed security camera. Do not
assume a setting or a subject.

Record, as far as the evidence supports:

1. Actions, in plain verbs: walking, entering, opening, lifting, typing, pouring,
   pointing, handing something over, driving, demonstrating.

2. State changes, as a before-to-after transition.
   Example: a door is shut, a hand turns the handle, the door stands open.

3. People: how many, what they wear, where they are, and what distinguishes them.
   Give a role only when it is visually obvious. Do not guess identity, age,
   relationships, or intent.

4. Objects that matter to the scene, including anything prominent or unusual.

5. Interactions between people, and between people and objects.

6. The setting: indoors or outdoors, the kind of place, time of day and weather
   if visible, and what is displayed if this is a recording of a screen.

7. Motion and camera changes, such as a pan, a cut, or a static camera.

8. Text that is genuinely readable in the frames. Never guess at blurred text.

9. Important events, with approximate start and end offsets in seconds from the
   BEGINNING OF THIS CLIP.

10. Search terms: words someone might plausibly search for to find this passage,
    including synonyms for the actions and objects that appear.

For anything spoken, rely on the supplied transcript. Never invent dialogue, and
do not report speech that the transcript does not contain.

Write the summary densely and factually, favoring concrete nouns and verbs over
interpretation, so that matching it against a search query is easy.
"""

#Analyze one interval of the source video with the local vision model and return structured metadata
def analyze_video_clip(manifest, start, end):
    return local_backend.interval_json(
        manifest["video"]["path"],
        start,
        end,
        METADATA_PROMPT,
        VIDEO_METADATA_SCHEMA,
    )

def add_chunk_context(
    metadata,
    chunk
):
    metadata = metadata.copy()

    metadata["chunk_id"] = chunk["chunk_id"]
    metadata["scale"] = chunk["scale"]

    metadata["start"] = chunk["start"]
    metadata["end"] = chunk["end"]

    for event in metadata["important_events"]:

        event["absolute_start"] = (
            chunk["start"]
            + event["start_seconds"]
        )

        event["absolute_end"] = (
            chunk["start"]
            + event["end_seconds"]
        )

    return metadata

#Generates metadata for all chunks in a manifest and saves to a JSONL file. Returns the chunks that failed.
def generate_metadata(
    manifest,
    scale="medium",
    output_path="metadata/medium_metadata.jsonl",
    retry_count=3,
    max_chunks=None,
):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = manifest["chunks"][scale]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    # Completed chunks are skipped, so an interrupted run resumes where it stopped.
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass

    print(f"{len(completed)} chunks already completed.")

    failures = []
    for chunk in local_backend.track(chunks, f"Describing {scale} scenes"):
        if chunk["chunk_id"] in completed:
            continue

        error = None
        for attempt in range(retry_count):
            try:
                metadata = add_chunk_context(
                    analyze_video_clip(manifest, chunk["start"], chunk["end"]),
                    chunk,
                )
                break
            except Exception as e:
                error = e
                print(f"\n{chunk['chunk_id']} attempt {attempt + 1} failed:\n{e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
        else:
            print("Skipping", chunk["chunk_id"])
            failures.append({"chunk_id": chunk["chunk_id"], "error": str(error)})
            continue

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

    return failures

def load_jsonl(path):

    output = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            output.append(
                json.loads(line)
            )

    return output

#Convert metadata into searchable text
def metadata_to_search_text(metadata):

    sections = []
    if metadata.get("summary"):
        sections.append(
            "Summary: "
            + metadata["summary"]
        )
    if metadata.get("scene"):
        sections.append(
            "Scene: "
            + metadata["scene"]
        )

    fields = [
        ("People", "people"),
        ("Objects", "objects"),
        ("Actions", "actions"),
        ("Interactions", "interactions"),
        ("Motion", "motion_events"),
        ("State changes", "state_changes"),
        ("Visible text", "visible_text"),
        ("Search terms", "search_terms"),
    ]

    for label, key in fields:

        values = metadata.get(key, [])

        if values:
            sections.append(f"{label}: " + "; ".join(values))

    events = metadata.get("important_events",[])

    if events:
        event_descriptions = [
            event["description"]
            for event in events
        ]

        sections.append(
            "Important events: "
            + "; ".join(
                event_descriptions
            )
        )

    return "\n".join(sections)

#Create metadata embeddings 
def embed_metadata_records(
    records,
    save_dir="metadata_index",
    retry_count=3,
):

    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = []
    index_metadata = []

    for record in local_backend.track(records, "Indexing scene descriptions"):
        text = metadata_to_search_text(record)
        success = False
        for attempt in range(retry_count):
            try:
                embedding = embed_text(text)
                embedding = normalize_embedding(embedding)
                success = True
                break
            except Exception as e:
                print(
                    f"\nEmbedding failed for "
                    f"{record['chunk_id']}: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
        if not success:
            continue
        embeddings.append(
            embedding
        )
        index_metadata.append({
            "chunk_id":
                record["chunk_id"],
            "start":
                record["start"],
            "end":
                record["end"],
            "summary":
                record["summary"],
            "search_text":
                text,
            "important_events":
                record["important_events"],
        })

    embeddings = np.vstack(
        embeddings
    ).astype(np.float32)

    np.save(
        save_dir /
        "medium_metadata_embeddings.npy",
        embeddings
    )

    with open(
        save_dir /
        "medium_metadata_records.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            index_metadata,
            f,
            indent=2
        )

    return embeddings, index_metadata

#Build FAISS Index for metadata 
def build_metadata_faiss(
    metadata_embeddings,
    save_path="metadata_index/medium_metadata.faiss"
):

    import faiss
    embeddings = np.asarray(metadata_embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]

    # normalized embeddings + IP = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(save_path)
    )

    print(
        f"Indexed {index.ntotal} metadata documents."
    )

    return index

def search_metadata(
    query,
    index,
    records,
    top_k=10,
):

    query_embedding = embed_text(query)

    query_embedding = normalize_embedding(query_embedding)

    scores, ids = index.search(
        query_embedding[
            None, :
        ].astype(np.float32),
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        item = records[
            idx
        ].copy()

        item["score"] = float(
            score
        )

        results.append(
            item
        )

    return results


def load_metadata_index(save_dir="metadata_index"):
    """Load metadata index assets created by embed_metadata_records."""
    import faiss
    save_dir = Path(save_dir)
    embeddings = np.load(save_dir / "medium_metadata_embeddings.npy").astype(np.float32)
    with open(save_dir / "medium_metadata_records.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    index_path = save_dir / "medium_metadata.faiss"
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = build_metadata_faiss(embeddings, save_path=index_path)
    return index, records

