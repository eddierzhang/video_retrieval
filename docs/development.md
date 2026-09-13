# Development

- [Setup](#setup)
- [Running the app](#running-the-app)
- [The moments command](#the-moments-command)
- [Using the library](#using-the-library)
- [Project layout](#project-layout)
- [Tests and checks](#tests-and-checks)
- [Learned models on disk](#learned-models-on-disk)
- [Conventions](#conventions)

## Setup

Requirements: Python 3.11+, [FFmpeg](https://ffmpeg.org/download.html) with `ffmpeg` and `ffprobe` on
`PATH`, and [Ollama](https://ollama.com/download).

```bash
python -m venv .venv
source .venv/bin/activate                   # Windows: .\.venv\Scripts\Activate.ps1

# On an NVIDIA GPU, install the CUDA build of PyTorch first so SigLIP and Whisper use it:
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -e ".[dev]"
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

The first index downloads the SigLIP 2 and Whisper weights (about 1.5 GB) into `local_data/models/`.
After that everything works offline.

On Windows, `.\start.ps1` does the Ollama part for you: it starts the server if it is not running,
pulls the two models on first run, and opens the app.

## Running the app

```bash
moments serve                     # http://127.0.0.1:8765, opens a browser tab
moments serve --port 9000 --no-browser
```

## The moments command

Every tool is a subcommand, and `moments <command> --help` lists its options.

| Command | What it does |
| --- | --- |
| `moments serve` | Run the web app |
| `moments benchmark` | Score end-to-end retrieval against a labeled dataset (`--dataset`, `--mode quick`, `--compare`) |
| `moments routing` | Score how often the planner picks the right executor |
| `moments runs` | `list`, `show` and `compare` recorded experiment runs |
| `moments construct` | Build benchmark timelines whose answers are known (`--ingest`, `--append`, `--event-seconds`) |
| `moments synthesize` | Turn scene descriptions into pseudo-queries |
| `moments tune` | Cross-validated search over how quick mode picks a moment (`--evaluate` to score saved settings) |
| `moments boundaries` | Train the clip boundary model |
| `moments rank` | Train the candidate ranker and conformal set (`--learning-curve`, `--ablate`) |
| `moments distill` | Train the pointwise candidate pre-filter |
| `moments adapt` | Train the self-supervised query adapter |
| `moments replay` | Replay logged searches to compare stopping policies |

Anything that plans, describes or verifies needs Ollama running: `benchmark`, `routing`, `construct`,
`synthesize`, and `tune` on rows it has not cached yet. The others need only the files on disk.

A typical loop, from nothing to tuned settings:

```bash
moments construct --timelines 10 --ingest
moments construct --event-seconds 4 20 --segments 8 --timelines 8 --out bench/constructed_short.json --ingest
moments boundaries
moments tune --dataset bench/constructed.json --dataset bench/constructed_short.json
moments benchmark --dataset bench/constructed.json --mode quick
moments runs list
```

## Using the library

```python
from video_retrieval import LocalModels, prepare_video

pipeline = prepare_video("videos/example.mp4", LocalModels(), reporter=print)
result = pipeline.retrieve("a person getting out of a car", run_verification=False)
for match in result["matches"]:
    print(match["start_timestamp"], match["end_timestamp"], match["confidence"], match["clip_path"])
```

`prepare_video` builds a video's indexes, or loads them if they already exist. `retrieve` accepts
`reporter=` for progress events, `cancel_event=` for cooperative cancellation, and
`run_verification=False` for a quick search. `notebooks/walkthrough.ipynb` runs the same stages one
cell at a time.

## Project layout

```
moments/            the moments command
video_retrieval/    the retrieval pipeline
webapp/             the local web app: Starlette API, job queue, video library, UI in static/
bench/              datasets, benchmarks and trainers
notebooks/          a stage-by-stage walkthrough
tests/              unit tests, no models or network needed
docs/               architecture, machine learning, evaluation, development
```

| Module | Responsibility |
| --- | --- |
| `video_retrieval/local_backend.py` | The only place models are called: Ollama chat, SigLIP, Whisper, frame sampling, progress, cancellation |
| `video_retrieval/config.py` | Model names, cache locations, the Ollama URL |
| `video_retrieval/video.py` | Probing, hierarchical chunking, clip and frame extraction |
| `video_retrieval/embeddings.py` | Chunk embeddings, the multi-scale index, FAISS |
| `video_retrieval/metadata.py` | Scene descriptions: generation, embedding, search |
| `video_retrieval/transcript.py` | Transcription, transcript windows, semantic and BM25 search |
| `video_retrieval/retrieval.py` | Planning and routing, multimodal retrieval, the evidence timeline, candidates, `DEFAULT_SCORING` |
| `video_retrieval/verification.py` | Both verification passes, boundary refinement, temporal NMS |
| `video_retrieval/visual_text.py` | The visual-text (OCR) executor |
| `video_retrieval/pipeline.py` | `VideoRetrievalPipeline`, `retrieve_video`, the replayable `locate_candidates` stage |
| `video_retrieval/learning.py` | Candidate logging, pre-filter, ranker, conformal selection with exploration, query adapter |
| `video_retrieval/boundaries.py` | Learned clip boundaries |
| `video_retrieval/local_indexing.py` | Building and loading a video's indexes |
| `bench/run.py`, `routing.py` | End-to-end and routing benchmarks |
| `bench/construct.py`, `synthesize.py` | Benchmark data with known and model-derived answers |
| `bench/tune.py`, `boundaries.py`, `rank.py`, `distill.py`, `adapt.py`, `replay.py` | Trainers |
| `bench/tracking.py` | Experiment records under `bench/runs/` |

## Tests and checks

```bash
python -m unittest discover -s tests
python -m pyflakes moments video_retrieval webapp bench tests
```

The suite runs in a few seconds with no GPU, models or network - model calls are mocked at the
`local_backend` boundary, and a test asserts nothing but loopback is ever contacted. GitHub Actions runs
both commands on every push and pull request.

| File | Covers |
| --- | --- |
| `tests/test_runtime.py` | The model boundary, embeddings and indexes, model names, progress and cancellation |
| `tests/test_retrieval.py` | Planning and routing, the evidence timeline, ordering, quick search, tuned settings, learned boundaries |
| `tests/test_learning.py` | Candidate logging, pre-filter, ranker, conformal selection, exploration, query adapter |
| `tests/test_bench.py` | Benchmark construction, tuning replay fidelity, stopping policies, experiment tracking, trainer logic |
| `tests/test_cli.py` | The `moments` command |
| `tests/test_webapp.py` | Upload, indexing, search, settings and file serving in the web app |

A few tests check properties rather than examples: optimal stopping against brute force over 300 random
lists, boundary refinement never returning an inverted interval under 200 random models, and the tuning
replay returning exactly what `retrieve_video` returns under both default and unusual settings.

## Learned models on disk

Trained models live in `local_data/learning/`. Deleting a file returns that part of the pipeline to its
hand-written behaviour.

| File | Effect when present |
| --- | --- |
| `retrieval_settings.json` | Tuned scoring and candidate settings, quick searches only |
| `boundary_model.json` | Learned boundaries on quick-search answers |
| `candidate_ranker.json` | Ranks candidates and keeps a conformal set before verification |
| `candidate_prefilter.json` | Drops unlikely candidates before verification, when there is no ranker |
| `query_adapter.npz` | Adapts frame-channel query embeddings |
| `candidates.jsonl` | Not a model: the training log every verified search appends to |

## Conventions

- **Every model call goes through `local_backend`**, so tests can mock one boundary and models can be
  swapped in one place.
- **Learned components are inert until trained**, and each falls back to the behaviour it replaces.
- **Trainers refuse to save a model that does not beat its baseline on held-out data**, and split by
  the unit that is actually independent - whole searches, spans, timelines - never by row.
- **Every benchmark and trainer records its run** through `bench.tracking`, including input hashes.
- **Commit messages explain why**, including results and caveats; `git log` is the project's history
  of what was tried and what it showed.
