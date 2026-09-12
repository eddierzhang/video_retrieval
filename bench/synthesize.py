"""Generate synthetic query -> interval pairs from the scene descriptions in each index.

    python -m bench.synthesize                                 every indexed video
    python -m bench.synthesize --video "Tennis v2.mov" --per-chunk 2
    python -m bench.run --dataset bench/synthetic.json --sample 12

Each indexed chunk already carries a description written by the vision model. Asking the
planner to turn that description into the queries a person might type gives a labeled pair
for free: the query, and the chunk it came from.

What these labels are worth: they measure "can retrieval find what the scene model saw",
not "is this description true". They inherit its blind spots, so they are good for tuning
and for catching regressions, and they are no substitute for the handful of rows in
dataset.json that were labeled by eye.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from video_retrieval.local_backend import chat_json, use_models
from webapp.library import Library

ROOT = Path(__file__).resolve().parent
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


def describe(chunk):
    """A compact, searchable summary of one indexed chunk."""
    parts = [str(chunk.get("summary", "")).strip()]
    for key, label in (("actions", "Actions"), ("objects", "Objects"), ("visible_text", "Visible text")):
        values = [str(v).strip() for v in (chunk.get(key) or []) if str(v).strip()][:6]
        if values:
            parts.append(f"{label}: " + "; ".join(values))
    return "\n".join(part for part in parts if part)


def queries_for(chunk, count, has_speech):
    prompt = f"""
Someone wants to find this exact moment inside a longer video by typing a search in plain English.

WHAT HAPPENS IN THIS MOMENT:
{describe(chunk)}

Write {count} different search queries a person might type to find THIS moment.

Rules:
- Each query must name something distinctive to this moment, so it would not match every
  other part of the video equally well.
- Write them the way someone types a search: short, natural, specific. Not captions.
- Vary them: one about an action, one about a visible object or place{", and one about what is said" if has_speech else ""}.
- Never copy the description verbatim, and never mention timestamps, "the video", "scene" or "clip".
"""
    result = chat_json(prompt, QUERY_SCHEMA, role="planner")
    cleaned = []
    for query in result.get("queries", []):
        query = re.sub(r"\s+", " ", str(query)).strip().strip('"')
        if 3 <= len(query.split()) <= 20:
            cleaned.append(query)
    return cleaned[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", action="append", help="library name or id; repeatable. Default: all indexed videos")
    parser.add_argument("--per-chunk", type=int, default=3)
    parser.add_argument("--mode", choices=("quick", "verified"), default="quick")
    parser.add_argument("--out", default=str(ROOT / "synthetic.json"))
    args = parser.parse_args()

    library = Library()
    records = [record for record in library.list() if record.get("index")]
    if args.video:
        wanted = set(args.video)
        records = [record for record in records if record["id"] in wanted or record["name"] in wanted]
    if not records:
        raise SystemExit("No indexed videos. Index one in the app first.")

    rows, seen = [], set()
    with use_models():
        for record in records:
            folder = library.video_dir(record["id"]) / record["index"]["dir"]
            path = folder / "metadata.jsonl"
            if not path.is_file():
                print(f"{record['name']}: no scene descriptions, skipping")
                continue
            chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            has_speech = bool(record["index"].get("has_transcript"))
            print(f"{record['name']}: {len(chunks)} chunks", flush=True)
            for chunk in chunks:
                if not str(chunk.get("summary", "")).strip():
                    continue
                try:
                    generated = queries_for(chunk, args.per_chunk, has_speech)
                except Exception as exc:
                    print(f"   {chunk.get('chunk_id')}: {type(exc).__name__}: {exc}")
                    continue
                for i, query in enumerate(generated):
                    key = query.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "id": f"syn-{record['id'][:6]}-{chunk['chunk_id']}-{i}",
                        "video": record["name"],
                        "mode": args.mode,
                        "query": query,
                        "expect": {"start": float(chunk["start"]), "end": float(chunk["end"])},
                        "source": "synthetic",
                        "label_confidence": "synthetic",
                    })
                print(f"   {chunk.get('chunk_id')}: {generated}", flush=True)

    output = {
        "about": [
            "Generated by bench/synthesize.py from the scene descriptions in each index.",
            "These measure whether retrieval can find what the scene model saw - not whether the",
            "description was right. Use them for tuning and regression checks, and keep the",
            "hand-labeled rows in dataset.json as the honesty anchor.",
            "A query written for one chunk may legitimately match a neighbouring one, so treat",
            "individual misses as noise and watch the aggregate.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()
