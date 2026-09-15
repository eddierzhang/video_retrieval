"""Score searches against hand-labelled intervals, through the running app's API.

    python -m bench.detect --modes detect verified --label baseline
    python -m bench.detect --compare bench/results/a.json bench/results/b.json

A returned clip is a hit when at least half of it lies inside a labelled interval, neutral when it
lies inside an interval marked acceptable, and a false positive otherwise. A labelled interval is
found when returned clips cover at least half of it or one second, whichever is less. Text
queries also check the reading against the accepted answers. Searches created for a run are
deleted afterwards unless --keep is given, so the app's history stays clean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = ROOT / "bench" / "detect_ground_truth.json"
RUNS = ROOT / "bench" / "results"


# ------------------------------------------------------------------ scoring

def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def covered(interval, clips):
    """Seconds of `interval` inside the union of `clips`."""
    pieces = sorted((max(interval[0], c[0]), min(interval[1], c[1])) for c in clips if overlap(interval, c) > 0)
    total, reach = 0.0, interval[0]
    for start, end in pieces:
        start = max(start, reach)
        if end > start:
            total += end - start
            reach = end
    return total


def normalise_text(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def score(truth, clips, texts=()):
    """Hits, neutrals and false positives among `clips`, and which labelled intervals were found."""
    intervals, acceptable = truth["intervals"], truth.get("acceptable", [])
    hits = neutral = false_positives = 0
    verdicts = []
    for clip in clips:
        length = max(1e-6, clip[1] - clip[0])
        if covered(clip, intervals) >= 0.5 * length:
            verdict = "hit"
            hits += 1
        elif covered(clip, intervals + acceptable) >= 0.5 * length:
            verdict = "neutral"
            neutral += 1
        else:
            verdict = "false_positive"
            false_positives += 1
        verdicts.append(verdict)
    found = [covered(interval, clips) >= min(1.0, 0.5 * (interval[1] - interval[0])) for interval in intervals]
    total = sum(end - start for start, end in intervals)
    result = {
        "clips": len(clips), "hits": hits, "neutral": neutral, "false_positives": false_positives,
        "verdicts": verdicts, "intervals": len(intervals), "found": sum(found),
        "recall": sum(found) / len(intervals) if intervals else None,
        "precision": hits / (hits + false_positives) if hits + false_positives else None,
        "coverage": sum(covered(interval, clips) for interval in intervals) / total if total else None,
    }
    if truth.get("answers"):
        readings = [normalise_text(text) for text in texts if text]
        answers = [normalise_text(answer) for answer in truth["answers"]]
        result["text_correct"] = any(answer and answer in reading for reading in readings for answer in answers)
    return result


def summarise(rows):
    """Totals over queries: micro precision, mean recall, and text accuracy where it applies."""
    scored = [row for row in rows if row.get("score")]
    hits = sum(row["score"]["hits"] for row in scored)
    false_positives = sum(row["score"]["false_positives"] for row in scored)
    recalls = [row["score"]["recall"] for row in scored if row["score"]["recall"] is not None]
    texts = [row["score"]["text_correct"] for row in scored if "text_correct" in row["score"]]
    return {
        "queries": len(scored), "clips": sum(row["score"]["clips"] for row in scored),
        "hits": hits, "false_positives": false_positives,
        "precision": hits / (hits + false_positives) if hits + false_positives else None,
        "mean_recall": sum(recalls) / len(recalls) if recalls else None,
        "intervals_found": f"{sum(row['score']['found'] for row in scored)}/{sum(row['score']['intervals'] for row in scored)}",
        "queries_with_a_hit": sum(1 for row in scored if row["score"]["hits"]),
        "text_correct": f"{sum(texts)}/{len(texts)}" if texts else None,
        "failed": sum(1 for row in rows if row.get("status") != "done"),
        "seconds": round(sum(row.get("seconds", 0) for row in rows)),
    }


# --------------------------------------------------------------------- API

class Client:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"X-Moments": "1", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def search(self, video_id, query, mode, options):
        search = self.call("POST", f"/api/videos/{video_id}/searches", {"query": query, "mode": mode, "options": options})
        while search["status"] not in ("done", "failed", "cancelled"):
            time.sleep(2)
            search = self.call("GET", f"/api/videos/{video_id}/searches/{search['id']}")
        return search


def run(base_url, modes, label, options=None, only=None, keep=False):
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    client = Client(base_url)
    videos = {video["name"]: video["id"] for video in client.call("GET", "/api/videos")["videos"]}
    report = {"label": label, "base_url": base_url, "started_at": time.time(), "options": options or {}, "modes": {}}
    for mode in modes:
        rows = []
        for item in truth["queries"]:
            if only and item["id"] not in only:
                continue
            if item["video"] not in videos:
                rows.append({"id": item["id"], "status": "missing video"})
                continue
            started = time.time()
            search = client.search(videos[item["video"]], item["query"], mode, options or {})
            result = search.get("result") or {}
            matches = result.get("matches", [])
            clips = [[float(m["start"]), float(m["end"])] for m in matches]
            texts = [m.get("extracted_text") for m in matches] + [e.get("text") for e in result.get("text_entities", [])]
            row = {"id": item["id"], "query": item["query"], "status": search["status"], "error": search.get("error"),
                   "seconds": round(time.time() - started, 1), "clips": clips,
                   "decided_by": [m.get("decided_by") for m in matches],
                   "routed": result.get("routed"), "plan": result.get("plan"),
                   "score": score(item, clips, texts) if search["status"] == "done" else None}
            rows.append(row)
            print(f"{mode:9s} {item['id']:14s} {row['status']:6s} {row['seconds']:5.0f}s  "
                  + (json.dumps({k: row["score"][k] for k in ("clips", "hits", "false_positives", "found", "intervals")})
                     if row["score"] else str(row["error"])), flush=True)
            if not keep:
                client.call("DELETE", f"/api/videos/{videos[item['video']]}/searches/{search['id']}")
        report["modes"][mode] = {"summary": summarise(rows), "queries": rows}
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"detect_benchmark_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return path, report


def compare(paths):
    """One line per run and mode: the summary numbers side by side."""
    lines = []
    keys = ("precision", "mean_recall", "intervals_found", "hits", "false_positives", "queries_with_a_hit", "text_correct", "seconds")
    lines.append(f"{'run':40s} " + " ".join(f"{key:>18s}" for key in keys))
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for mode, data in report["modes"].items():
            summary = data["summary"]
            cells = []
            for key in keys:
                value = summary.get(key)
                cells.append(f"{value:>18.2f}" if isinstance(value, float) else f"{str(value):>18s}")
            lines.append(f"{(report['label'] + ' / ' + mode)[:40]:40s} " + " ".join(cells))
    return "\n".join(lines)


def merge(paths, label):
    """One report from several runs of the same modes: a rerun of some queries replaces their rows."""
    truth_order = [item["id"] for item in json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["queries"]]
    modes = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for mode, data in report["modes"].items():
            rows = modes.setdefault(mode, {})
            for row in data["queries"]:
                if row.get("status") == "done" or row["id"] not in rows:
                    rows[row["id"]] = row
    merged = {"label": label, "merged_from": [str(path) for path in paths], "started_at": time.time(), "modes": {}}
    for mode, rows in modes.items():
        ordered = [rows[key] for key in truth_order if key in rows]
        merged["modes"][mode] = {"summary": summarise(ordered), "queries": ordered}
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"detect_benchmark_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    return path, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--modes", nargs="+", default=["detect"], choices=["detect", "verified", "quick"])
    parser.add_argument("--label", default="run")
    parser.add_argument("--options", default="{}", help="search options as JSON, e.g. '{\"vision_check\": false}'")
    parser.add_argument("--only", nargs="*", help="query ids to run")
    parser.add_argument("--keep", action="store_true", help="keep the searches in the app's history")
    parser.add_argument("--compare", nargs="+", help="print saved runs side by side instead of running")
    parser.add_argument("--merge", nargs="+", help="combine saved runs into one, later runs replacing earlier rows")
    args = parser.parse_args()
    if args.compare:
        print(compare(args.compare))
        return
    if args.merge:
        path, report = merge(args.merge, args.label)
        for mode, data in report["modes"].items():
            print(mode, json.dumps(data["summary"]))
        print("saved", path)
        return
    path, report = run(args.base_url, args.modes, args.label, json.loads(args.options), args.only, args.keep)
    for mode, data in report["modes"].items():
        print(mode, json.dumps(data["summary"]))
    print("saved", path)


if __name__ == "__main__":
    main()
