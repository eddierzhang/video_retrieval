"""Train the query adapter from the descriptions and speech already inside each index.

    python -m bench.adapt                      train on every indexed video
    python -m bench.adapt --dry-run            report the pairs and the baseline, train nothing
    python -m bench.adapt --rank 8 --steps 600

Every indexed chunk hands over free training pairs: the text the vision model wrote about a
stretch of video, and the frame embeddings for that same stretch. Scene summaries, the search
terms and actions the scene model listed, the pseudo-queries `bench.synthesize` wrote, and
optionally speech all describe the same spans. Fitting a linear map from query-text embeddings
into the frame-embedding space adapts retrieval to your own footage without one hand-labeled
example.

The map is applied to queries only, never to stored vectors, so training a new one never
invalidates an index. It is a low-rank residual on top of the identity, because with a few
hundred pairs a full 768x768 matrix would memorise rather than generalise.

Four things this does that a naive contrastive fit does not:

  grouped split      every text and frame belonging to one span stays on one side of the
                     train/holdout line, so a caption cannot be scored against a frame its
                     own twin trained on
  all five views     a chunk is scored at search time by its best-matching view - whole frame
                     plus four tiles - so training maximises the same quantity
  masked negatives   spans that overlap in time, or whose text says nearly the same thing, are
                     removed from the negatives rather than pushed apart
  hard negatives     batches are drawn mostly from one video, so the negatives are other
                     moments in the same place rather than trivially different footage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from video_retrieval.learning import ADAPTER_MODEL
from video_retrieval.local_backend import encode, frame_embeddings
from webapp.library import Library

SYNTHETIC = Path(__file__).with_name("synthetic.json")
SOURCES = ("scene", "terms", "actions", "synthetic", "speech")
DEFAULT_SOURCES = ("scene", "terms", "actions", "synthetic")


def normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-8, None)


def embed_queries(strings, batch=64):
    """Text vectors by the same route a query takes, minus the adapter being trained.

    Every text here is already cut to 40 words, which is one piece for `embed_text`, so a
    straight batched encode matches what a query gets at search time.
    """
    vectors = []
    for start in range(0, len(strings), batch):
        vectors.append(encode(text=[str(text) for text in strings[start:start + batch]]))
        print(f"   embedded {min(start + batch, len(strings))}/{len(strings)} texts", end="\r")
    print(" " * 48, end="\r")
    return normalize(np.concatenate(vectors))


def span_views(times, vectors, start, end):
    """Mean embedding per view across one span, or the nearest frame if it has none."""
    inside = (times >= float(start)) & (times < float(end))
    frames = vectors[inside] if inside.any() else vectors[[int(np.argmin(np.abs(times - (start + end) / 2)))]]
    return normalize(frames.mean(axis=0))


def frame_views(times, vectors, start, end, count):
    """A few individual frames from a span, so the fit is not dominated by means."""
    inside = np.flatnonzero((times >= float(start)) & (times < float(end)))
    if not len(inside) or count <= 0:
        return []
    step = max(1, len(inside) // count)
    return [normalize(vectors[position]) for position in inside[::step][:count]]


def load_synthetic(path=SYNTHETIC):
    """Pseudo-queries from bench.synthesize, keyed by video name."""
    if not Path(path).is_file():
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("rows", [])
    by_video = {}
    for row in rows:
        expect = row.get("expect") or {}
        if row.get("query") and expect.get("end") is not None:
            by_video.setdefault(row["video"], []).append(
                (row["query"], float(expect["start"]), float(expect["end"])))
    return by_video


def collect(library, sources, per_chunk_frames=4, synthetic_path=SYNTHETIC):
    """Group every text and frame that describes one span of one video.

    Returns `groups` - one per span, holding its texts and its image views - so the split
    and the negative masking can both work at span granularity.
    """
    synthetic = load_synthetic(synthetic_path) if "synthetic" in sources else {}
    groups = []
    for record in library.list():
        if not record.get("index"):
            continue
        folder = library.video_dir(record["id"])
        index = folder / record["index"]["dir"]
        try:
            times, vectors = frame_embeddings(folder / record["source"])
        except Exception as exc:
            print(f"{record['name']}: cannot read frame embeddings ({exc})")
            continue

        spans = {}

        def add(text, start, end, kind):
            text = " ".join(str(text).split()[:40]).strip()
            if len(text.split()) < 2:
                return
            spans.setdefault((round(float(start), 2), round(float(end), 2)), []).append((text, kind))

        descriptions = index / "metadata.jsonl"
        if descriptions.is_file():
            for line in descriptions.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "scene" in sources:
                    add(chunk.get("summary", ""), chunk["start"], chunk["end"], "scene")
                if "terms" in sources:
                    for term in chunk.get("search_terms") or []:
                        add(term, chunk["start"], chunk["end"], "terms")
                if "actions" in sources:
                    for action in chunk.get("actions") or []:
                        add(action, chunk["start"], chunk["end"], "actions")
        segments = index / "transcript" / "segments.json"
        if "speech" in sources and segments.is_file():
            for row in json.loads(segments.read_text(encoding="utf-8")):
                if len(str(row.get("text", "")).split()) >= 4:
                    add(row["text"], row["start"], row["end"], "speech")
        for query, start, end in synthetic.get(record["name"], []):
            add(query, start, end, "synthetic")

        for (start, end), texts in sorted(spans.items()):
            views = [span_views(times, vectors, start, end)]
            views.extend(frame_views(times, vectors, start, end, per_chunk_frames))
            groups.append({
                "video": record["id"],
                "name": record["name"],
                "start": start,
                "end": end,
                "texts": [text for text, _ in texts],
                "kinds": [kind for _, kind in texts],
                "views": np.stack(views),  # (items, views, dim)
            })
        counts = {}
        for entry in spans.values():
            for _, kind in entry:
                counts[kind] = counts.get(kind, 0) + 1
        summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items())) or "nothing usable"
        print(f"{record['name']}: {len(spans)} spans, {summary}")
    return groups


def overlap(left, right):
    """Temporal intersection over union of two spans."""
    if left["video"] != right["video"]:
        return 0.0
    inner = min(left["end"], right["end"]) - max(left["start"], right["start"])
    outer = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return max(0.0, inner) / outer if outer > 0 else 0.0


def build_mask(groups, text_groups, text_vectors, mask_iou=0.25, duplicate=0.9):
    """Which (text, span) pairs must not count as negatives.

    A query written for one chunk legitimately matches a chunk that overlaps it, and the scene
    model reuses wording - the same search term is listed against half a dozen neighbouring
    chunks. Pushing those apart teaches the adapter a distinction that does not exist, so they
    are masked out of the loss instead. A text's own span is never masked - that is the positive.
    """
    count = len(groups)
    overlapping = np.zeros((count, count), dtype=bool)
    for i in range(count):
        for j in range(i + 1, count):
            overlapping[i, j] = overlapping[j, i] = overlap(groups[i], groups[j]) > mask_iou
    np.fill_diagonal(overlapping, False)
    mask = overlapping[text_groups]

    # Then, per text: a span that already carries near-identical wording is not a negative for it.
    by_group = [np.flatnonzero(text_groups == index) for index in range(count)]
    same_video = np.array([[groups[i]["video"] == groups[j]["video"] for j in range(count)]
                           for i in range(count)])
    for row, group in enumerate(text_groups):
        scores = text_vectors[row] @ text_vectors.T
        for other in np.flatnonzero(same_video[group]):
            if other != group and len(by_group[other]) and scores[by_group[other]].max() > duplicate:
                mask[row, other] = True
    mask[np.arange(len(text_groups)), text_groups] = False
    return mask  # (texts, spans)


def split_groups(groups, holdout=0.2, by="span", seed=0):
    """Hold out whole spans (or whole videos), never rows, so no text straddles the line."""
    random = np.random.RandomState(seed)
    if by == "video":
        videos = sorted({group["video"] for group in groups})
        random.shuffle(videos)
        chosen = set(videos[:max(1, int(len(videos) * holdout))])
        held = np.array([group["video"] in chosen for group in groups])
    else:
        order = random.permutation(len(groups))
        held = np.zeros(len(groups), dtype=bool)
        held[order[:max(1, int(len(groups) * holdout))]] = True
    return ~held, held


def similarity(text_vectors, views, weights=None):
    """Query-to-span scores under the adapter, scored by each span's best-matching view."""
    query = text_vectors if weights is None else text_vectors @ weights
    query = normalize(query)
    # views: (spans, items, views, dim) is ragged, so score item by item and take the best.
    return np.stack([np.max(query @ group.reshape(-1, group.shape[-1]).T, axis=1) for group in views], axis=1)


def evaluate(text_vectors, text_groups, groups, mask, weights=None, ranks=(1, 5), min_gallery=5):
    """Recall at k of a text's own span, ranked against every span of its own video.

    Search never ranks one video's chunks against another's, so neither does this. Spans masked
    for a text are dropped from the ranking rather than counted against it.
    """
    scores = np.where(mask, -np.inf, similarity(text_vectors, [group["views"] for group in groups], weights))
    videos = {}
    for index, group in enumerate(groups):
        videos.setdefault(group["video"], []).append(index)
    positions = []
    for columns in videos.values():
        if len(columns) < min_gallery:
            continue  # ranking among two or three spans says nothing
        rows = np.flatnonzero(np.isin(text_groups, columns))
        if not len(rows):
            continue
        block = scores[np.ix_(rows, columns)]
        lookup = {column: position for position, column in enumerate(columns)}
        own = np.array([lookup[group] for group in text_groups[rows]])
        order = np.argsort(-block, axis=1)
        positions.extend(np.argmax(order == own[:, None], axis=1))
    if not positions:
        return {f"recall@{k}": float("nan") for k in ranks} | {"median_rank": float("nan"), "texts": 0}
    positions = np.array(positions)
    report = {f"recall@{k}": float((positions < k).mean()) for k in ranks}
    report["median_rank"] = float(np.median(positions) + 1)
    report["texts"] = len(positions)
    return report


def show(label, report):
    print(f"   {label:<10} " + "   ".join(
        f"{key} {value:.1%}" if key.startswith("recall") else f"{key} {value:.0f}"
        for key, value in report.items() if key != "texts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, default=8, help="rank of the residual update")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32, help="spans per step, all from one video")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                        help=f"text sources to train on, any of {','.join(SOURCES)}")
    parser.add_argument("--per-chunk-frames", type=int, default=4)
    parser.add_argument("--split", choices=("span", "video"), default="span")
    parser.add_argument("--mask-iou", type=float, default=0.25)
    parser.add_argument("--out", default=str(ADAPTER_MODEL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = tuple(name.strip() for name in args.sources.split(",") if name.strip())
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise SystemExit(f"Unknown source(s) {sorted(unknown)}; choose from {SOURCES}")

    groups = [group for group in collect(Library(), sources, args.per_chunk_frames) if group["texts"]]
    texts = [text for group in groups for text in group["texts"]]
    text_groups = np.array([index for index, group in enumerate(groups) for _ in group["texts"]])
    if len(groups) < 10 or len(texts) < 40:
        raise SystemExit(f"Only {len(texts)} texts across {len(groups)} spans. Index more video first.")
    print(f"{len(texts)} texts across {len(groups)} spans from "
          f"{len({group['video'] for group in groups})} videos")

    text_vectors = embed_queries(texts)
    mask = build_mask(groups, text_groups, text_vectors, args.mask_iou)
    print(f"{mask.mean():.1%} of (text, span) pairs masked out of the negatives")

    # Spans are held out, but the gallery a held-out text is ranked against stays the whole
    # video - that is what search does, and a six-span gallery would flatter any adapter.
    train_spans, holdout_spans = split_groups(groups, by=args.split)
    held_texts = np.flatnonzero(holdout_spans[text_groups])
    if holdout_spans.sum() < 3 or len(held_texts) < 20:
        raise SystemExit("Too few spans to hold any out. Index more video first.")
    before = evaluate(text_vectors[held_texts], text_groups[held_texts], groups, mask[held_texts])
    print(f"holdout: {len(held_texts)} texts, ranked against every span of their own video")
    show("identity", before)
    if not before["texts"]:
        raise SystemExit("No video has enough spans to evaluate against. Index a longer video first.")
    if args.dry_run:
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dimensions = text_vectors.shape[1]
    videos = {}
    for index in np.flatnonzero(train_spans):
        videos.setdefault(groups[index]["video"], []).append(int(index))
    # Every negative is another moment from the same video, because that is the only kind
    # of mistake retrieval can make: one video's chunks are never ranked against another's.
    videos = {video: spans for video, spans in videos.items() if len(spans) >= 4}
    if not videos:
        raise SystemExit("No video has enough training spans. Index a longer video first.")
    print(f"training on {sum(len(spans) for spans in videos.values())} spans from {len(videos)} videos")

    texts_by_group = [np.flatnonzero(text_groups == index) for index in range(len(groups))]
    tensors = {index: torch.tensor(groups[index]["views"], device=device)
               for spans in videos.values() for index in spans}
    text_tensor = torch.tensor(text_vectors, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    left = torch.zeros(dimensions, args.rank, device=device, requires_grad=True)
    right = torch.zeros(args.rank, dimensions, device=device, requires_grad=True)
    torch.nn.init.normal_(left, std=0.01)
    torch.nn.init.normal_(right, std=0.01)
    identity = torch.eye(dimensions, device=device)
    optimizer = torch.optim.Adam([left, right], lr=1e-3, weight_decay=1e-4)
    scale = 20.0
    random = np.random.RandomState(0)
    names = sorted(videos)
    sizes = np.array([len(videos[name]) for name in names], dtype=np.float64)

    best = {"recall@1": -1.0}
    best_weights = None
    for step in range(args.steps):
        spans = videos[names[int(random.choice(len(names), p=sizes / sizes.sum()))]]
        chosen = [int(index) for index in random.choice(spans, size=min(args.batch, len(spans)), replace=False)]
        # One text and one image per span, so both directions of the loss stay 1:1.
        rows = [int(random.choice(texts_by_group[index])) for index in chosen]
        images = torch.stack([tensors[index][int(random.randint(len(tensors[index])))] for index in chosen])

        weights = identity + left @ right
        projected = torch.nn.functional.normalize(text_tensor[rows] @ weights, dim=1)
        logits = torch.einsum("qd,gvd->qgv", projected, images).amax(dim=2) * scale
        logits = logits.masked_fill(mask_tensor[rows][:, chosen], float("-inf"))
        target = torch.arange(len(rows), device=device)
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target)
                      + torch.nn.functional.cross_entropy(logits.T, target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            trained = (identity + left @ right).detach().cpu().numpy().astype(np.float32)
            current = evaluate(text_vectors[held_texts], text_groups[held_texts], groups, mask[held_texts], trained)
            if current["recall@1"] > best["recall@1"]:
                best, best_weights = current, trained
            if step % 150 == 0 or step == args.steps - 1:
                print(f"   step {step:>4}  loss {loss.item():.3f}  "
                      f"holdout recall@1 {current['recall@1']:.1%}  recall@5 {current['recall@5']:.1%}")

    show("identity", before)
    show("adapter", best)
    if best["recall@1"] <= before["recall@1"] and best["recall@5"] <= before["recall@5"]:
        print("The adapter did not beat the identity on held-out spans, so it is not being saved.")
        print("Index more video and try again; with this little data that is the expected outcome.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, weights=best_weights, rank=args.rank, texts=len(texts), spans=len(groups),
             sources=",".join(sources), split=args.split,
             recall1_before=before["recall@1"], recall1_after=best["recall@1"],
             recall5_before=before["recall@5"], recall5_after=best["recall@5"])
    print("written to", args.out)
    print("Then check it end to end: python -m bench.run --compare bench/results/<an earlier run>")


if __name__ == "__main__":
    main()
