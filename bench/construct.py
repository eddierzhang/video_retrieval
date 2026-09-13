"""Build benchmark videos whose answers are known because they were put there.

    python -m bench.construct                    build timelines and write bench/constructed.json
    python -m bench.construct --ingest           also add them to the library and index them
    python -m bench.construct --timelines 4 --segments 5 --seed 2
    python -m bench.construct --append --timelines 7 --ingest   add seven more to the existing set
    python -m bench.construct --event-seconds 4 20 --segments 10 --out bench/constructed_short.json --ingest
    python -m bench.run --dataset bench/constructed.json --mode quick

Every other label in this project traces back to the vision model: synthetic queries are its
descriptions rephrased, candidate verdicts are its opinions. None of them can tell you when that
model is wrong, because they all agree with it. This builds the one kind of ground truth that
does not: an edit list.

  splice   whole 30-second chunks from indexed videos are cut, normalised to one format, and
           concatenated into a new timeline. The manifest records where each one landed, to
           the frame, so "the tennis serve is at 2:30-3:00" is true because it was placed there.
  text     a random code ("K7X-4629") is drawn onto known frames in a known box. The expected
           answer is the string itself, and when it was on screen.

Two things the edit list cannot make exact, and which the rows say out loud:

  query text     a spliced chunk's query is still written from the scene model's description
                 of it (the bench.synthesize pseudo-query when there is one). The interval is
                 exact; whether the words fit the footage is as good as that description.
  look-alikes    two chunks from the same video can look alike. Chunks from one source must
                 differ in what the scene model listed for them, and any that share a source are
                 recorded as `confusable_with`, so a miss on one of them can be read for what it is.

Constructed boundaries fall on hard cuts, which real events do not. A method that snaps to cuts
will look better here than it is on real footage - see bench.boundaries, which saves the model without cut features for that reason.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

from bench.tracking import Run, file_digest, seed_everything
from video_retrieval.config import DATA_DIR
from video_retrieval.video import probe_video
from webapp.library import Library

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "constructed.json"
SYNTHETIC = ROOT / "synthetic.json"
OUTPUT = DATA_DIR / "constructed"
WIDTH, HEIGHT, FPS = 1280, 720, 30
# No O/0, I/1, S/5 or B/8: the benchmark should measure reading, not guessing between glyphs.
LETTERS = "ACDEFHJKLMNPRTUVWXY"
DIGITS = "234679"
FONTS = (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
         Path("/System/Library/Fonts/Supplemental/Arial.ttf"))
TEXT_QUERY = "read the code shown in the dark box near the top of the screen"


# ------------------------------------------------------------------ choosing

def load_synthetic(path=SYNTHETIC):
    """Pseudo-queries keyed by the chunk they were written for."""
    if not Path(path).is_file():
        return {}
    by_chunk = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")).get("rows", []):
        expect = row.get("expect") or {}
        if row.get("query") and expect.get("end") is not None:
            key = (row["video"], round(float(expect["start"]), 1), round(float(expect["end"]), 1))
            by_chunk.setdefault(key, []).append(row["query"])
    return by_chunk


def chunk_pool(library, synthetic, min_seconds=10.0):
    """Every described medium chunk in the library, with what the scene model said about it."""
    pool = []
    for record in library.list():
        if not record.get("index") or str(record.get("name", "")).startswith("Constructed "):
            continue  # never splice a constructed video into another one
        folder = library.video_dir(record["id"])
        descriptions = folder / record["index"]["dir"] / "metadata.jsonl"
        if not descriptions.is_file():
            continue
        duration = float(record.get("duration") or 0.0)
        for line in descriptions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            start, end = float(chunk["start"]), min(float(chunk["end"]), duration or float(chunk["end"]))
            if end - start < min_seconds:
                continue
            key = (record["name"], round(start, 1), round(float(chunk["end"]), 1))
            pool.append({
                "video": record["name"],
                "video_id": record["id"],
                "source": str(folder / record["source"]),
                "has_audio": bool(record.get("audio_codec")),
                "start": start,
                "end": end,
                "summary": str(chunk.get("summary", "")),
                "terms": [str(term) for term in (chunk.get("search_terms") or [])],
                "actions": [str(action) for action in (chunk.get("actions") or [])],
                "queries": synthetic.get(key, []),
            })
    return pool


def tokens(chunk):
    words = " ".join(chunk["terms"] + chunk["actions"]).lower().split()
    return {word.strip(".,;:!?\"'()") for word in words if len(word) > 2}


def similarity(left, right):
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def choose_segments(pool, count, rng, max_similarity=0.35, balanced=False):
    """Chunks for one timeline: never overlapping in their source, never near-duplicates.

    `balanced` draws round-robin across source videos instead of uniformly across chunks, so one
    long video with many chunks does not fill a timeline with look-alikes.
    """
    order = list(range(len(pool)))
    rng.shuffle(order)
    if balanced:
        by_video = {}
        for index in order:
            by_video.setdefault(pool[index]["video_id"], []).append(index)
        queues, order = list(by_video.values()), []
        while any(queues):
            rng.shuffle(queues)
            order.extend(queue.pop() for queue in queues if queue)
    chosen = []
    for index in order:
        chunk = pool[index]
        clash = False
        for other in chosen:
            if other["video_id"] != chunk["video_id"]:
                continue
            if chunk["start"] < other["end"] and other["start"] < chunk["end"]:
                clash = True  # the same seconds twice would put the answer in two places
            elif similarity(chunk, other) > max_similarity:
                clash = True
            if clash:
                break
        if not clash:
            chosen.append(chunk)
        if len(chosen) == count:
            break
    return chosen


def query_for(chunk, rng):
    if chunk["queries"]:
        return rng.choice(chunk["queries"]), "synthetic"
    sentence = chunk["summary"].split(". ")[0]
    return " ".join(sentence.split()[:16]).rstrip("."), "summary"


SLICE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def describe_slice(chunk):
    """A query written for exactly this slice, since the chunk's own description covers 30 seconds."""
    from video_retrieval.local_backend import interval_json

    prompt = (
        f"These frames are one continuous {chunk['end'] - chunk['start']:.0f}-second clip. Write the search "
        "query a person would type to find exactly this clip in a longer video: 4 to 12 words naming "
        "what is visibly happening or shown that makes it recognisable. No timestamps, and do not use "
        "the words clip, video, frame or scene."
    )
    answer = interval_json(chunk["source"], chunk["start"], chunk["end"], prompt, SLICE_SCHEMA,
                           role="vision", include_speech=False)
    return " ".join(str(answer.get("query", "")).split()[:14]).strip().rstrip(".")


def slice_chunk(chunk, rng, shortest, longest):
    """A random stretch of `shortest` to `longest` seconds from inside a described chunk."""
    available = chunk["end"] - chunk["start"]
    length = min(available, rng.uniform(shortest, longest))
    offset = rng.uniform(0.0, max(0.0, available - length))
    start = round(chunk["start"] + offset, 3)
    return {**chunk, "start": start, "end": round(start + length, 3), "queries": []}


def query_overlap(left, right):
    a = {word.strip(".,;:!?\"'()").lower() for word in left.split()}
    b = {word.strip(".,;:!?\"'()").lower() for word in right.split()}
    return len(a & b) / len(a | b) if a and b else 0.0


def make_code(rng):
    return "".join(rng.choice(LETTERS) for _ in range(3)) + "-" + "".join(rng.choice(DIGITS) for _ in range(4))


# ----------------------------------------------------------------- rendering

def _ffmpeg(arguments, cwd):
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", *arguments], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-600:]}")


def render_segment(chunk, target, workdir, overlay=None):
    """Cut one chunk into the shared format, optionally with a code drawn on known frames."""
    duration = chunk["end"] - chunk["start"]
    filters = [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        f"fps={FPS}",
    ]
    if overlay:
        # The font sits in the working directory so its path needs no filtergraph escaping.
        filters.append(
            f"drawtext=fontfile=font.ttf:text='{overlay['text']}':fontsize=64:fontcolor=white"
            f":box=1:boxcolor=black@0.75:boxborderw=18:x=(w-text_w)/2:y=h*0.06"
            f":enable='between(t,{overlay['offset']:.3f},{overlay['offset'] + overlay['seconds']:.3f})'"
        )
    arguments = ["-ss", f"{chunk['start']:.3f}", "-t", f"{duration:.3f}", "-i", chunk["source"]]
    if chunk["has_audio"]:
        audio = ["-map", "0:a:0"]
    else:
        arguments += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        audio = ["-map", "1:a:0"]
    arguments += ["-vf", ",".join(filters), "-map", "0:v:0", *audio,
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", f"{duration:.3f}", str(target)]
    _ffmpeg(arguments, workdir)
    return probe_video(target)["duration"]


def concatenate(parts, target, workdir):
    listing = Path(workdir) / "parts.txt"
    listing.write_text("".join(f"file '{Path(part).name}'\n" for part in parts), encoding="utf-8")
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", listing.name, "-c", "copy", "-movflags", "+faststart",
             str(target)], workdir)


def build_timeline(number, pool, args, rng, font):
    segments = choose_segments(pool, args.segments, rng, args.max_similarity, balanced=bool(args.event_seconds))
    if len(segments) < 2:
        raise SystemExit("Not enough distinct described chunks in the library to build a timeline.")
    if args.event_seconds:
        # Short events: a slice of each chunk, so answers are not all exactly thirty seconds long.
        segments = [slice_chunk(chunk, rng, *args.event_seconds) for chunk in segments]
    name = f"{args.prefix} {number}.mp4"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / name.replace(" ", "_")
    overlays = set(rng.sample(range(len(segments)), min(args.text, len(segments)))) if font else set()

    placed, rows, cursor = [], [], 0.0
    with tempfile.TemporaryDirectory() as workdir:
        if font:
            shutil.copyfile(font, Path(workdir) / "font.ttf")
        parts = []
        for position, chunk in enumerate(segments):
            length = chunk["end"] - chunk["start"]
            overlay = None
            if position in overlays and length >= 12:
                overlay = {"text": make_code(rng), "seconds": 5.0,
                           "offset": round(rng.uniform(3.0, length - 8.0), 3)}
            part = Path(workdir) / f"part{position:02d}.mp4"
            # The measured length, not the requested one, is what the next segment starts after.
            actual = render_segment(chunk, part, workdir, overlay)
            parts.append(part)
            if args.event_seconds:
                query, query_source = describe_slice(chunk), "vision-slice"
            else:
                query, query_source = query_for(chunk, rng)
            placed.append({
                "position": position, "video": chunk["video"], "source_start": chunk["start"],
                "source_end": chunk["end"], "start": round(cursor, 3), "end": round(cursor + actual, 3),
                "query": query, "query_source": query_source, "overlay": overlay,
            })
            cursor += actual
            print(f"   {position + 1}/{len(segments)} {chunk['video']} {chunk['start']:.0f}-{chunk['end']:.0f}s"
                  f"{'  + text ' + overlay['text'] if overlay else ''}")
        concatenate(parts, target, workdir)

    for segment in placed:
        confusable = [other["position"] for other in placed
                      if other is not segment and other["video"] == segment["video"]]
        # Two segments whose queries say nearly the same thing cannot both be found by any setting;
        # a miss on one of them is noise, so the row says so and bench.tune leaves it out.
        ambiguous = [other["position"] for other in placed
                     if other is not segment and query_overlap(other["query"], segment["query"]) >= 0.5]
        base = {"video": name, "mode": "verified", "source": "constructed", "label_confidence": "exact",
                "timeline": number, "segment": segment["position"], "confusable_with": confusable,
                "ambiguous_with": ambiguous}
        rows.append({**base, "id": f"{args.id_prefix}-{number}-{segment['position']}-event", "query": segment["query"],
                     "query_source": segment["query_source"],
                     "expect": {"start": segment["start"], "end": segment["end"]}})
        if segment["overlay"]:
            shown = segment["start"] + segment["overlay"]["offset"]
            rows.append({**base, "id": f"{args.id_prefix}-{number}-{segment['position']}-text", "query": TEXT_QUERY,
                         "query_source": "constructed",
                         "expect": {"text": segment["overlay"]["text"], "start": round(shown, 3),
                                    "end": round(shown + segment["overlay"]["seconds"], 3)}})
    return {"number": number, "name": name, "file": str(target), "duration": round(cursor, 3),
            "sha256": file_digest(target), "segments": placed}, rows


# ------------------------------------------------------------------ indexing

def ingest_and_index(path, name, library, models):
    """Add a constructed video to the library and index it, the way the web app does."""
    from video_retrieval.local_indexing import index_key, prepare_video

    digest = file_digest(path)
    incoming = library.incoming_file()
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, incoming)
    record, created = library.ingest(incoming, digest, name)
    if record.get("index"):
        print(f"   {record['name']}: already indexed")
        return record
    folder = library.video_dir(record["id"])
    source = folder / record["source"]
    library.update(record["id"], status="indexing")
    started = time.time()
    prepare_video(source, models, index_root=folder / "index")
    key, _ = index_key(source, models)
    ready = json.loads((folder / "index" / key / "ready.json").read_text(encoding="utf-8"))
    index = {"dir": f"index/{key}", "models": ready["models"], "stats": ready.get("stats", {}),
             "has_transcript": ready["has_transcript"], "built_at": time.time()}
    library.update(record["id"], status="ready", error=None, index=index)
    print(f"   {record['name']}: indexed in {time.time() - started:.0f}s")
    return library.get(record["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timelines", type=int, default=3)
    parser.add_argument("--segments", type=int, default=6, help="chunks spliced into each timeline")
    parser.add_argument("--event-seconds", type=float, nargs=2, metavar=("SHORTEST", "LONGEST"),
                        help="splice slices of this many seconds instead of whole chunks, each with a query "
                             "the vision model writes for that slice")
    parser.add_argument("--prefix", default=None, help="library name for the timelines")
    parser.add_argument("--id-prefix", default=None, help="prefix for row ids")
    parser.add_argument("--text", type=int, default=2, help="segments per timeline that get a drawn code")
    parser.add_argument("--max-similarity", type=float, default=0.35,
                        help="how alike two chunks from one video may be, by what the scene model listed")
    parser.add_argument("--ingest", action="store_true", help="add to the library and index (slow)")
    parser.add_argument("--append", action="store_true",
                        help="keep the timelines already in --out and add --timelines more after them")
    parser.add_argument("--out", default=str(DATASET))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    short = bool(args.event_seconds)
    args.prefix = args.prefix or ("Constructed short timeline" if short else "Constructed timeline")
    args.id_prefix = args.id_prefix or ("short" if short else "con")
    if short and args.event_seconds[0] > args.event_seconds[1]:
        raise SystemExit("--event-seconds takes the shortest length first.")

    seed_everything(args.seed)
    with Run("construct", args, seed=args.seed) as run:
        print(f"run {run.id}")
        run.input(SYNTHETIC)
        rng = random.Random(args.seed)
        library = Library()
        pool = chunk_pool(library, load_synthetic())
        print(f"{len(pool)} described chunks available from "
              f"{len({chunk['video_id'] for chunk in pool})} indexed videos")
        font = next((path for path in FONTS if path.is_file()), None)
        if args.text and not font:
            print("No usable font found, so no text rows will be drawn.")

        timelines, rows = [], []
        if args.append and Path(args.out).is_file():
            # Existing timelines are already indexed; rebuilding them would re-encode different bytes
            # and put a second copy in the library under the same name.
            existing = json.loads(Path(args.out).read_text(encoding="utf-8"))
            timelines, rows = existing.get("timelines", []), existing.get("rows", [])
            print(f"keeping {len(timelines)} existing timelines and {len(rows)} rows")
        first = max((timeline["number"] for timeline in timelines), default=0) + 1
        built = []
        for number in range(first, first + args.timelines):
            print(f"\ntimeline {number}")
            if args.append:
                # Each appended timeline draws from its own stream, so which ones already exist
                # does not change what a given number contains.
                rng = random.Random(f"{args.seed}:{number}")
            timeline, timeline_rows = build_timeline(number, pool, args, rng, font)
            timelines.append(timeline)
            built.append(timeline)
            rows.extend(timeline_rows)
            run.log({"timeline": number, "segments": len(timeline["segments"]), "rows": len(timeline_rows),
                     "duration": timeline["duration"]})

        document = {
            "about": [
                "Generated by bench/construct.py. Intervals are exact: each chunk was placed where the",
                "timeline says, and each code was drawn on the frames the row names. Query wording for",
                "event rows still comes from the scene model's description of the chunk.",
                "Rows that share a source video list each other in confusable_with.",
            ],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "run": run.id,
            "timelines": timelines,
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(document, indent=2), encoding="utf-8")
        run.artifact(args.out, "constructed_dataset")
        events = sum(1 for row in rows if "text" not in row["expect"])
        run.summarize(timelines=len(timelines), rows=len(rows), event_rows=events, text_rows=len(rows) - events)
        print(f"\n{len(rows)} rows ({events} event, {len(rows) - events} text) written to {args.out}")

        if args.ingest:
            from video_retrieval.local_backend import LocalModels

            print("\nindexing - this runs the vision model over every timeline")
            models = LocalModels()
            for timeline in built:
                record = ingest_and_index(Path(timeline["file"]), timeline["name"], library, models)
                if record["name"] != timeline["name"]:
                    # Identical bytes were already in the library under another name.
                    for row in rows:
                        if row["video"] == timeline["name"]:
                            row["video"] = record["name"]
            Path(args.out).write_text(json.dumps(document, indent=2), encoding="utf-8")
            print(f"\nready: python -m bench.run --dataset {Path(args.out).as_posix()}")
        else:
            print("Not indexed yet. Run again with --ingest, or add the files in local_data/constructed "
                  "through the app, before bench.run can use these rows.")


if __name__ == "__main__":
    main()
