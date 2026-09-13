# Moments — natural-language video retrieval, entirely local

Ask a question about a video in plain English and get back timestamped clips:

> *“when does someone get out of the car?”* · *“who says ‘thank you’?”* · *“read the license plate”*

Every model runs on your machine — query planning, embeddings, transcription, scene
descriptions, verification and OCR. No API key, no account, and no video ever leaves
the computer.

---

## Quick start (Windows)

Requirements: Python 3.11+, [FFmpeg](https://ffmpeg.org/download.html) (`ffmpeg` and
`ffprobe` on `PATH`), and [Ollama](https://ollama.com/download/windows).

```powershell
python -m venv .venv
# Optional but recommended on an NVIDIA GPU: CUDA build first, so CLIP and Whisper use it
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.ps1
```

`start.ps1` starts the Ollama server if it isn't already running, downloads the two models it
needs on first run (`qwen3.5:4b` and `nomic-embed-text`, about 3.7 GB together), and opens
<http://127.0.0.1:8765>.

The first index also downloads the SigLIP 2 and Whisper weights (about 1.5 GB) into
`local_data/models/`. After that the whole system works offline.

## Using it

1. **Add a video** — drag a file anywhere, or use *Add video*. MP4, MOV, MKV, WebM and AVI.
2. **Wait for indexing** — this happens once per video. The player shows each stage as it runs;
   you can watch the video meanwhile, and indexing resumes from finished work if you stop it.
3. **Ask** — type a description, a spoken phrase, or text that appears on screen.
   * **Verified** checks every candidate with the local vision model and tightens the clip
     boundaries. Slower, and false positives are rejected.
   * **Quick** ranks moments straight from the indexes in seconds, with no vision checks.
4. **Read the answer** — each moment gives a clip, a confidence score, the evidence behind it,
   and a frame strip. The timeline shows where the query found support across the whole video,
   which regions were searched, and where the answers are.
5. **Look under the hood** — the *Query plan* tab shows the route taken, the evidence the planner
   decomposed your question into, how the retrieval channels were weighted, and the funnel from
   candidates to confirmed matches. *Transcript* and *History* sit beside it.

Settings (bottom-left) pick the local models and how many frames each vision call sees.
Changing the planner or verifier takes effect immediately; changing the scene model or the frame
count needs a re-index, and the app tells you so.

### Measured on this machine (RTX 4080 SUPER, CUDA build of PyTorch)

| Step | Time |
| --- | --- |
| Indexing a 21 s clip | 15 s |
| Indexing a 104 s clip with speech | 78 s |
| Indexing a 5:47 clip with speech | 157 s |
| Quick search | 19 s |
| Text-reading search | 39 s |
| Verified search | 40-130 s, depending on how many candidates survive |

Indexing cost is dominated by scene descriptions - roughly one vision call per 30 s of video, about
7 s each here - plus one pass of frame embeddings (five views per frame, so five times the embedding
work of a single whole-frame pass) and one Whisper pass. Search cost is dominated by the vision calls
in verification, so it grows with the number of candidates you allow and how finely each one is
split, not with the length of the video.

## What runs locally

| Role in the pipeline | Model |
| --- | --- |
| Query planner | `qwen3.5:4b` via Ollama |
| Scene descriptions, first verification pass, boundary refinement, OCR | `qwen3.5:4b` (vision) |
| Second, stricter verification pass | `qwen3.5:4b` (configurable separately) |
| Frame embeddings | SigLIP 2 `google/siglip2-base-patch16-224`, whole frame plus 2x2 tiles |
| Scene and transcript embeddings | `nomic-embed-text` via Ollama |
| Speech transcription | faster-whisper `base`, with word timestamps |

One multimodal model fills every role by default, so the GPU never swaps models mid-search.
Any Ollama model with the `vision` capability can be substituted in settings; the planner slot
accepts text-only models too.

Vision models see sampled frames plus the transcript for the interval, not a native video stream.
Sparse frames and small local models reduce accuracy on brief actions, small text, and fine
temporal precision compared with large hosted models. No equivalence is claimed.

---

# How retrieval works

The pipeline follows a coarse-to-fine strategy. Instead of processing every frame for every
query, the video is indexed once into searchable chunks; a query then narrows broad regions down
to precise timestamps.

```
Video
  └─ hierarchical chunking (coarse 120 s · medium 30 s · fine 8 s, overlapping)
      └─ feature extraction: CLIP frame embeddings · speech transcript · VLM scene metadata
          └─ FAISS indexes + BM25
Natural-language query
  └─ query planner  → per-channel queries, atomic evidence predicates, weights
      └─ multimodal retrieval (visual · scene · speech-semantic · speech-keyword)
          └─ temporal evidence map → candidate regions → recursive refinement
              └─ verification pass 1 → verification pass 2 → boundary refinement → temporal NMS
                  └─ final timestamps, clips and frames (FFmpeg)
```

### 1. Hierarchical chunking
The video is divided at several temporal scales. Large chunks preserve the context needed to
recognize an event; small chunks localize it. Retrieval moves from coarse to fine.

### 2. Video embeddings
Frames are sampled once per second and embedded with SigLIP 2, which outperforms CLIP ViT-B/32 at a
similar size. Each frame is embedded five times over: once whole, and once per tile of a 2x2 grid.
The model resizes whatever it is given to 224 px, so a detail occupying a tenth of a 4K frame is a
handful of pixels by the time it is seen; the tiles give that detail a view of its own. A chunk keeps one vector per view, mean-pooled over its frames,
and scores against a query by its **best-matching view**, so a match in one corner is not averaged
away by three quiet ones. Each frame is still embedded only once and reused across all scales.

### 3. Transcript retrieval
Audio is transcribed once with word-level timestamps and split into overlapping windows. Both
semantic (embedding) and BM25 (lexical) search run over it: embeddings catch paraphrase, BM25
catches exact names and phrases. The semantic side uses a dedicated text embedder rather than the
image-text model, whose text tower is trained on short captions and capped well below the length of
a paragraph - a poor fit for sentences of speech.

### 4. Visual metadata
Each medium chunk is described by the vision model as structured JSON (actions, people, objects,
state changes, visible text, search terms). These descriptions are embedded with the same text
embedder as the transcript and indexed separately, giving a second, independent visual channel
that does not depend on the raw frame embedding.

### 5. OCR / visual text
When answering needs text that is visible on screen, a separate executor scans the video, then
re-reads the best original-resolution frames and crops, and takes a consensus across readings.

### 6. Query planning
One planner call routes the request (event search vs text reading) and decomposes it into
per-channel queries plus 4–12 atomic evidence predicates, each with a role, the channels that can
retrieve it, whether it is required, and how discriminative it is. It also returns negative
evidence, temporal constraints, an expected duration, and channel weights.

### 7. Evidence aggregation
Every retrieval hit is mapped back onto a shared timeline. Within a query, overlapping hits
combine with a noisy-OR so agreement creates a peak rather than a flat score; channels are then
combined using the planner's weights. Scores are calibrated per ranking first, because raw scores
from different modalities are not comparable. The planner also names confounders it wants rejected;
those are retrieved the same way and **subtracted** from the map, so a region that looks like the
wrong thing scores lower rather than merely failing to score higher.

### 8. Candidate ranking
Connected regions of the evidence map above a relative floor become candidates, ranked by peak
score, total evidence mass, and supporting bins. When the request needs one thing to happen before
another, the planner returns that as a pair of evidence-predicate ids, and candidates whose evidence
peaks in the required order are promoted over candidates where it appears reversed.

### 9. Recursive temporal refinement
Promising regions are recursively subdivided into overlapping child windows, following the
strongest evidence down to roughly the expected event duration. This avoids fine-grained search
across the whole video.

### 10. Verification
The highest-ranking candidates are inspected by the vision model, which finds every distinct
occurrence inside a region. A vision call spends a fixed frame budget on whatever span it is
handed, so candidates are split into short windows: at 12 frames, a 75-second window is one frame
every six seconds, while a 25-second window is one every two. a second, stricter pass then re-checks each proposed occurrence and
rejects near misses. Boundaries are refined by asking progressively shorter clips where the
transition happens, and overlapping detections are removed by temporal NMS that preserves
genuinely distinct actors.

### 11. Final clip extraction
FFmpeg cuts the matching interval from the original video and extracts frames for the UI.

---

## Measuring accuracy

Changes to prompts, models or thresholds trade recall against precision in ways that are easy to
feel and hard to see, so there is a small benchmark:

```powershell
.\.venv\Scripts\python.exe -m bench.run                                  # every row
.\.venv\Scripts\python.exe -m bench.run --compare bench\results\<file>  # against an earlier run
```

Rows live in `bench/dataset.json`: a video in the library, a query, and either a true interval
(scored by temporal IoU, with recall at 0.3 and 0.5) or the exact characters a text query should
return. Results are written to `bench/results/` so two runs can be diffed row by row.

The dataset ships with three starter rows and only one of them has boundaries checked by eye - it
is a smoke test, not a benchmark, until you add your own. To label a row: play the video in the app,
note when the event really starts and ends, and append an entry. Ten careful rows are worth more
than fifty careless ones.

### Labels without labeling

```powershell
.\.venv\Scripts\python.exe -m bench.synthesize
```

Each indexed chunk already carries a description written by the vision model. Asking the planner to
turn those into the queries a person might type makes the chunk itself the ground truth: three
videos here produced 83 pairs in a couple of minutes, written to `bench/synthetic.json`.

These measure whether retrieval can find **what the scene model saw**, not whether it saw correctly,
so they inherit its blind spots. Use them for tuning and for catching regressions, and keep the
hand-checked rows as the anchor - a model-derived label once quietly agreed with a hallucination
here, and only reading the frame caught it.

### Answers known by construction

```powershell
.\.venv\Scripts\python.exe -m bench.construct --ingest       # build timelines, add and index them
.\.venv\Scripts\python.exe -m bench.run --dataset bench\constructed.json --mode quick
```

Every label above traces back to the vision model, so none of them can catch it being wrong. This
builds the one kind of ground truth that does not: an edit list. Whole described chunks from indexed
videos are cut, normalised to one format and spliced into new timelines, and random codes
(`K7X-4629`) are drawn onto known frames. A row's interval is where the chunk was placed; a text
row's answer is the string that was drawn. Checked by extracting frames: the code is on screen
inside its window and absent a second before, and a two-minute timeline drifts 21 ms from its
placements.

What an edit list cannot make exact is written into the rows. Event queries are still worded from
the scene model's description of the chunk. Two chunks from one source can look alike, so chunks
from the same video must differ in what the scene model listed, and any that share a source record
each other as `confusable_with`. Every constructed boundary is also a hard cut, which real events
are not - see the boundary model below for what that does.

### Every run is recorded

```powershell
.\.venv\Scripts\python.exe -m bench.tracking list
.\.venv\Scripts\python.exe -m bench.tracking compare <run> <run>
```

Each benchmark and trainer writes a folder under `bench/runs/`: the command and every argument, the
seed, the git commit and whether the tree had uncommitted changes (with a hash of the diff), library
versions, the GPU, a SHA-256 of every input file, per-step metrics, and a hash of every model it
wrote. A run that crashes or stops early is recorded too. `compare` lists what two runs disagreed
on and says when their inputs differed, so "the model improved" gets checked against "the data
changed" first. Every trainer takes `--seed`; the first comparison showed that on this much data a
different seed moves the *baseline*, because it changes which searches are held out.

## Learning from the pipeline's own output

Optional models, trained from data the system produces as you use it. None exists until you train
it, and every path falls back to the previous behaviour when the file is absent.

### Candidate pre-filter, distilled from the verifier

Every verified search appends one row per candidate to `local_data/learning/candidates.jsonl`: the
features retrieval had already computed, and whether the vision model's verdict confirmed that
candidate. A logistic regression learns to predict the verdict, and the pipeline then skips
candidates the vision model would have rejected - which is where a verified search spends nearly all
its time. The 4B vision model is the teacher; nothing is labeled by hand.

```powershell
.\.venv\Scripts\python.exe -m bench.distill --dry-run   # describe what has been collected
.\.venv\Scripts\python.exe -m bench.distill             # train it
```

The filter always keeps the strongest candidates whatever it believes, so a badly fitted model can
cost time but can never empty the candidate list.

### Candidate ranking, and knowing how many to verify

The pre-filter above asks "is this candidate good?" one candidate at a time. Nothing consumes
that answer one candidate at a time - verification walks the list from the top, so what matters
is the **order**, and ordering is what a ranking loss optimises directly.

```powershell
.\.venv\Scripts\python.exe -m bench.rank --dry-run   # what has been collected
.\.venv\Scripts\python.exe -m bench.rank             # train every loss, keep the best
```

Three losses over the same small network, so the comparison is an ablation rather than a claim:
**pointwise** binary cross-entropy per candidate (what `bench.distill` does), **pairwise**
RankNet over positive/negative pairs weighted by the NDCG each swap would change (LambdaRank),
and **listwise** softmax cross-entropy over a whole search (ListNet). All three are scored
against the ordering retrieval already produces, and nothing is saved unless it beats that.

Then **split conformal prediction** answers the expensive question: how many candidates deserve
a vision call? On held-out searches it calibrates a score threshold such that the set it keeps
contains a confirmed candidate at least `1 - alpha` of the time - 90% by default, `--alpha` to
trade coverage for speed. The guarantee is about coverage across searches, not about any single
candidate being right, and it holds without the probabilities being well calibrated. It needs
three disjoint splits, whole searches on one side of each line:

```
fit         train the ranker
calibrate   measure how badly the best confirmed candidate tends to score
test        check the coverage actually delivered, and what it cost
```

The floor from the pre-filter still applies: whatever the model believes, the strongest few
candidates are always verified, so a badly fitted ranker can cost time but cannot empty the list.

**Exploration keeps the loop honest.** A ranker that only ever collects rows about candidates it
approved of cannot learn that it was wrong to reject something: it would narrow with every
generation while the benchmark stayed silent, because the candidates it wrongly killed never
appear anywhere. So one rejected candidate in ten is verified anyway. Those rows are marked
`explored` and carry the probability that selected them, and pointwise training divides by it -
a row sampled one time in ten stands for the ten like it that were not. It costs a little time
per search and is the only thing keeping the next generation's data from being a picture of the
last one's opinions.

Two questions worth asking of any of this:

```powershell
.\.venv\Scripts\python.exe -m bench.rank --learning-curve   # short of data, or short of model?
.\.venv\Scripts\python.exe -m bench.rank --ablate           # which features carry the signal
```

The curve retrains on a growing slice of the fit split; if held-out NDCG is still climbing at the
end, collecting more searches is worth more than a better model. The ablation drops each group of
features and reports what the ordering loses without it, which is also how you find out that a
feature you were proud of contributes nothing.

### Query adapter, self-supervised from your own videos

Each indexed chunk hands over free training pairs: the text the vision model wrote about a stretch
of video, and the frame embeddings for that same stretch. Scene summaries, the search terms and
actions the scene model listed, and the pseudo-queries `bench.synthesize` writes all describe the
same spans; speech can be added with `--sources ...,speech`. A low-rank residual on the identity is
fitted from query-text embeddings into the frame-embedding space by a contrastive loss, adapting
retrieval to your footage without a single hand-labeled example.

```powershell
.\.venv\Scripts\python.exe -m bench.adapt --dry-run   # pairs and the untrained baseline
.\.venv\Scripts\python.exe -m bench.adapt             # train it
```

Four details decide whether the number it reports means anything:

* **Negatives come from the same video.** Search never ranks one video's chunks against another's,
  so neither does training or evaluation. Every negative is a different moment in the same place -
  the only mistake retrieval can actually make.
* **Spans are held out, not rows.** A span contributes a dozen texts and several frames. Splitting
  by row would put a caption in training and its own twin in the holdout, and the score would
  measure memorisation.
* **Overlap and repeated wording are masked, not punished.** Medium chunks overlap by half, and the
  scene model lists the same search term against a stretch of neighbours. Those pairs leave the
  loss rather than being pushed apart; on the three videos here that is about a quarter of them.
* **Training maximises what search scores.** A chunk is ranked by its best-matching view - whole
  frame plus four tiles - so the loss takes the same maximum instead of the whole frame alone.

It applies to queries only, never to stored vectors, so training a new one never invalidates an
index. The script refuses to save an adapter that fails to beat the identity on held-out spans.
On the three videos indexed here - 430 texts across 30 spans - it moves held-out recall@1 from
42.5% to 46.0% with recall@5 unchanged, which on 87 held-out texts is three of them and well inside
the noise. Treat a number like that as "nothing broke", not as a gain, and confirm any adapter
against `bench.run` before trusting it.

### Clip boundaries, from the frame embeddings

```powershell
.\.venv\Scripts\python.exe -m bench.boundaries                     # train on constructed intervals
.\.venv\Scripts\python.exe -m bench.boundaries --proposals bench\results\<run>.json   # real proposals
```

A quick-mode answer is an evidence region padded by ten seconds either side, which is most of why its
IoU is low. The per-second frame embeddings already hold two curves that say where an event starts
and ends: how well each second matches the query, and how different it looks from the second
before. Each proposed boundary is scored at every second within a window by a linear model over
features of both curves - a softmax over positions - fitted on the exact constructed intervals. Quick
mode applies it when a model exists, in milliseconds and without a vision call.

Held out by whole timeline, across three seeds, from simulated proposals:

| Method | Mean IoU | Boundary error |
| --- | --- | --- |
| Proposal as given | 0.835-0.850 | ~2.9 s |
| Snap to the biggest cut | 0.817-0.841 | ~3.0 s |
| Sit on the similarity step | 0.877-0.895 | ~2.2 s |
| Learned, with cut features | 0.892-0.909 | ~1.7 s |
| **Learned, no cut features** | **0.916-0.935** | **~1.4 s** |

The cut features were expected to flatter the model, because every constructed boundary is a hard
cut. They did the opposite: the source footage has cuts of its own, and the two cut features came
out nearly equal and opposite. So the no-cuts model is saved by default and `--with-cuts` has to be
asked for. These proposals start close to the truth; real quick-mode ones do not, which is what
`--proposals` is for - `bench.run` records each row's match intervals so a result file can be used.

### When to stop verifying, learned by replaying searches

```powershell
.\.venv\Scripts\python.exe -m bench.replay                       # every match counts
.\.venv\Scripts\python.exe -m bench.replay --goal first          # only the first one does
.\.venv\Scripts\python.exe -m bench.replay --value-seconds 120 --ranker
```

Choosing when to stop asking the vision model about candidates is a sequential decision problem, and
learning one by trial costs a minute or more per attempt. But each verified search logs every
candidate's verdict and what it cost to verify, so a search can be replayed: a policy asks for
candidates in its own order, the replay answers from the log and charges the logged seconds, and an
episode costs microseconds.

The reward is `value-seconds` per confirmed match minus the seconds spent. On held-out searches it
compares verifying everything (today's behaviour), a fixed top-k, a static probability threshold,
**optimal stopping** by backward induction over the ordered list, and **fitted Q iteration**, which
learns the value of verifying from replayed outcomes instead of trusting the probabilities. When
fitted Q wins, the probabilities were not telling the whole story. A test checks the backward
induction against brute force over 300 random lists, and a sweep over what a match is worth shows the
policy moving from verifying nothing to verifying nearly everything.

The replay can only answer for candidates that were verified. That holds for every logged candidate
until a ranker is trained, and afterwards only for the explored rows - one more reason they exist.

## Project layout

| Path | What it holds |
| --- | --- |
| `video_retrieval/config.py` | Local model names, cache locations, Ollama URL |
| `video_retrieval/local_backend.py` | The only place models are called: Ollama chat, CLIP, Whisper, frame sampling, progress and cancellation |
| `video_retrieval/video.py` | Probing, hierarchical chunking, clip and frame extraction |
| `video_retrieval/embeddings.py` | Chunk embeddings, hierarchical multi-scale index, FAISS |
| `video_retrieval/metadata.py` | Scene description generation, embedding and search |
| `video_retrieval/transcript.py` | Transcription, transcript windows, semantic + BM25 search |
| `video_retrieval/retrieval.py` | Query planner, multimodal retrieval, evidence map, candidates |
| `video_retrieval/verification.py` | Both verification passes, boundary refinement, temporal NMS |
| `video_retrieval/visual_text.py` | Visual-text (OCR) executor |
| `video_retrieval/local_indexing.py` | Builds or loads a video's indexes |
| `video_retrieval/pipeline.py` | `RetrievalResources` and `VideoRetrievalPipeline` |
| `webapp/` | Local web app: Starlette API, job queue, library on disk, and the UI in `webapp/static/` |
| `video_retrieval/learning.py` | The optional learned pieces: candidate pre-filter, ranker with its conformal set, and query adapter |
| `video_retrieval/boundaries.py` | Learned clip boundaries from per-second similarity and visual change |
| `bench/` | Labeled, synthetic and constructed rows, the accuracy runner, and the trainers: `distill`, `rank`, `adapt`, `boundaries` |
| `bench/construct.py` | Benchmark timelines with answers known from the edit list |
| `bench/replay.py` | Search replay, optimal stopping and fitted Q |
| `bench/tracking.py` | Run records under `bench/runs/`, listing and comparison |
| `model_completed.ipynb` | Notebook walkthrough of the same pipeline |

### Where data lives

Everything generated stays under `local_data/` (git-ignored):

```
local_data/library/<video id>/       source video, thumbnail, video.json
                    index/<hash>/    chunks, embeddings, transcript, scene descriptions
                    searches/<id>/   saved results with clips and frames
local_data/cache/                    frame embeddings and transcripts, keyed by file identity
local_data/learning/                 collected candidate outcomes and any trained models
local_data/models/                   CLIP and Whisper weights
```

Videos are content-addressed, so uploading the same file twice reuses the existing entry. The index
key covers only what changes an index's contents, so swapping the planner or verifier does not
force a rebuild. Deleting a video in the UI removes all of it.

### Security

The server binds to `127.0.0.1` and has no authentication — it is a single-user local tool. Requests
that change anything must carry an `X-Moments` header, which another website cannot send without a
CORS preflight that this server never grants, and only loopback hostnames are accepted. Uploaded
filenames are never used as paths, and files under a search are served only from inside that search's
own directory.

## Using the library directly

```python
from video_retrieval import LocalModels, prepare_video

pipeline = prepare_video("videos/example.mp4", LocalModels(), reporter=print)
result = pipeline.retrieve("a person getting out of a car")
for match in result["matches"]:
    print(match["start_timestamp"], match["end_timestamp"], match["confidence"], match["clip_path"])
```

`prepare_video` builds the indexes or loads them if they already exist. `retrieve` accepts
`reporter=` for progress events, `cancel_event=` for cooperative cancellation, and
`run_verification=False` for a retrieval-only search. `model_completed.ipynb` walks through the same
steps one stage at a time.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers the model boundaries (nothing reaches the network except loopback), interval
embedding and speech slicing, resumable scene description, cancellation, the retrieval-only path,
and the web app's upload, indexing, search, settings and file-serving behavior.

---

# Design notes

These are the original architecture decisions for the retrieval pipeline, updated where the models
changed.

1. **Hierarchical chunking:** large chunks preserve the context needed to understand an event while
   smaller chunks provide accurate timestamps, giving both efficient search and precise localization.
2. **Separate retrieval channels:** video, transcript, metadata and OCR capture different
   information, so a query can be routed toward its strongest source instead of forcing everything
   through one model.
3. **FAISS semantic search:** searching a large collection of embedding vectors is far faster than
   comparing a query against every stored chunk.
4. **Semantic search + BM25:** embeddings find conceptually similar content; BM25 is more reliable
   for exact words, names or phrases.
5. **Evidence-based ranking:** agreement between video, transcript and metadata signals is more
   reliable than any single similarity score.
6. **Zero-shot pretrained models:** no labelled dataset is needed, and the same system handles many
   kinds of video and question.
7. **Recursive refinement:** refining high-scoring coarse intervals into child windows avoids
   expensive fine-grained search across the whole video.
8. **Cached preprocessing:** embeddings, metadata, transcripts and indexes do not change between
   queries, so repeated searches over one video stay fast.
9. **Query planning:** different parts of a question need visual, transcript, OCR or temporal
   evidence, so the planner decomposes it rather than embedding it whole.
10. **Temporal evidence aggregation:** mapping results back onto a shared timeline makes overlapping
    support from different sources visible as peaks.
11. **Final verification:** retrieval returns semantically related clips that do not contain the
    event, so checking the strongest candidates reduces false positives.
12. **Separate preprocessing and retrieval:** a video is indexed once, after which many queries can
    be answered against it.
13. **One local runtime boundary:** every model call goes through `local_backend`, so models are
    swapped, progress reported and work cancelled in one place rather than in each stage.

## What I tried

1. I initially tried single fixed-size chunks, but choosing one length that captured enough
   information was difficult, which led to hierarchical chunking.
2. I first relied only on video and text embeddings, which did not work well for general prompts or
   prompts needing reasoning, so I added more retrieval channels.
3. I experimented with combining semantic and BM25 scores for audio, since exact language behaves
   differently from semantic meaning.
4. I implemented OCR specifically for text-extraction queries, rather than returning frames and
   leaving the reading to the user.

## Trade-offs

1. **Accuracy vs computation:** more temporal scales and channels give more evidence but cost
   preprocessing time, disk and VRAM.
2. **Context vs temporal precision:** long clips carry more context; short ones localize better.
3. **Query latency:** precomputation per video keeps per-query latency down, at the cost of a
   one-time indexing pass.
4. **General vs specialized models:** pretrained models avoid annotation and retraining, but a
   task-specific model would likely do better on any single task.
5. **Local vs hosted models:** running locally removes API keys, costs and rate limits, and keeps
   video private, but a 4B local model is weaker than a large hosted one at reading small text and
   reasoning over long contexts.

## Future work

1. **Better OCR:** text detection, frame enhancement, multi-frame aggregation and tracking for small
   or blurry text such as plates and signs.
2. **Stronger temporal reasoning:** extend the planner to handle before/after/during and sequences
   of several events.
3. **Learned ranking:** tune or learn how much weight each channel deserves per query instead of
   fixed rules.
4. **An annotated benchmark:** ground-truth timestamps and query–event pairs to measure retrieval
   accuracy and temporal localization properly.
5. **Stronger local models where they pay off:** a larger CLIP for retrieval recall and a larger
   Whisper for transcript quality, both of which fit alongside the vision model on a 16 GB GPU.
