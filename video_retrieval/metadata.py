#Breaks a video down into chunks, generates metadata for each chunk, and builds a searchable index of the metadata 
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from .config import METADATA_MODEL, OPENROUTER_CHAT_URL, get_openrouter_api_key
from .embeddings import embed_text, normalize_embedding
from .video import materialize_metadata_clip, video_to_data_url

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
Analyze this video for a temporal video retrieval system.

Your goal is to preserve as much SEARCHABLE visual information as possible.

Describe only things supported by the video. Do not guess hidden events,
identities, intentions, dialogue, or details that cannot be seen.

Pay particular attention to:

1. Actions
   - walking
   - running
   - entering/exiting
   - opening/closing
   - grabbing
   - reaching
   - sitting/standing
   - driving
   - interacting with another person
   - object manipulation

2. State changes
   Example:
   person inside car -> car door opens -> person exits car

3. People
   - clothing
   - approximate role if visually obvious
   - position
   - distinguishing visual characteristics

4. Objects
   - vehicles
   - bags
   - tools
   - phones
   - weapons if clearly visible
   - doors
   - furniture
   - equipment

5. Interactions between people and objects.

6. Scene/environment
   - indoors/outdoors
   - road
   - building
   - parking lot
   - daylight/night
   - weather if visible

7. Motion events and transitions.

8. Any readable visible text.
   Only record text that can actually be read.

9. Important temporal events.
   Provide approximate start/end offsets relative to the BEGINNING
   OF THIS CLIP.

10. Search terms.
    Include useful synonymous concepts someone might use when searching
    for this scene.

Make the summary detailed rather than generic.
"""

#Analyze a video clip and return structured metadata
def analyze_video_clip(
    video_path,
    model=METADATA_MODEL,
    timeout=300,
):

    video_data = video_to_data_url(
        video_path
    )

    payload = {
        "model": model,

        "messages": [
            {
                "role": "user",

                "content": [
                    {
                        "type": "text",
                        "text": METADATA_PROMPT
                    },

                    {
                        "type": "video_url",

                        "video_url": {
                            "url": video_data
                        }
                    }
                ]
            }
        ],

        "response_format": {
            "type": "json_schema",

            "json_schema": {
                "name": "video_metadata",
                "strict": True,
                "schema": VIDEO_METADATA_SCHEMA
            }
        },
        "provider": {
            "require_parameters": True
        },

        "temperature": 0.0
    }

    headers = {
        "Authorization":
            f"Bearer {get_openrouter_api_key()}",

        "Content-Type":
            "application/json",
    }

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            "Metadata generation failed:\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()

    content = (
        result["choices"][0]
        ["message"]
        ["content"]
    )

    # Normal OpenRouter response is a JSON string
    if isinstance(content, str):
        metadata = json.loads(content)

    else:
        raise RuntimeError(
            f"Unexpected response format: {content}"
        )

    return metadata

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

#Generates metadata for all chunks in a manifest and saves to a JSONL file
def generate_metadata(
    manifest,
    scale="medium",
    output_path="metadata/medium_metadata.jsonl",
    video_cache="metadata_video_cache",
    retry_count=3,
    sleep_seconds=0.25,
    max_chunks=None,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks = manifest["chunks"][scale]

    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        
    completed = set()

    if output_path.exists():

        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:
                    row = json.loads(line)

                    completed.add(
                        row["chunk_id"]
                    )

                except Exception:
                    pass

    print(
        f"{len(completed)} chunks already completed."
    )
    
    for chunk in tqdm(
        chunks,
        desc=f"Generating {scale} metadata"
    ):

        if chunk["chunk_id"] in completed:
            continue

        video_path = materialize_metadata_clip(
            manifest=manifest,
            chunk=chunk,
            output_dir=video_cache,
        )

        success = False

        for attempt in range(retry_count):

            try:

                metadata = analyze_video_clip(
                    video_path
                )

                metadata = add_chunk_context(
                    metadata,
                    chunk
                )

                success = True
                break

            except Exception as e:

                print(
                    f"\n{chunk['chunk_id']} "
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
                chunk["chunk_id"]
            )

            continue

        with open(
            output_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(metadata)
                + "\n"
            )

        time.sleep(
            sleep_seconds
        )

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

    for record in tqdm(records, desc="Embedding metadata"):
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

