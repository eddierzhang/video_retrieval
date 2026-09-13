# Evaluation

Changes to prompts, models and thresholds trade recall against precision in ways that are easy to feel
and hard to see. This page covers what Moments is measured against, how those datasets are built, what
they have shown, and how every run is recorded.

- [Datasets](#datasets)
- [Metrics](#metrics)
- [Benchmarks with known answers](#benchmarks-with-known-answers)
- [Results](#results)
- [Routing](#routing)
- [Recording and comparing runs](#recording-and-comparing-runs)
- [Known gaps](#known-gaps)

## Datasets

| File | Rows | Where the answers come from |
| --- | --- | --- |
| `bench/dataset.json` | 3 | Hand-labeled starter rows; a smoke test, not a benchmark |
| `bench/synthetic.json` | 83 | The planner rephrasing each chunk's scene description as a query |
| `bench/constructed.json` | 60 events, 20 text | Edit lists: 10 timelines of whole 30-second chunks, codes drawn onto known frames |
| `bench/constructed_short.json` | 64 events | Edit lists: 8 timelines of 4-20 second slices |
| `bench/routing.json` | 41 | Queries labeled by the kind of answer wanted: a moment, or text to read back |

The constructed files contain absolute paths into the local video library, so they are git-ignored
and rebuilt with `moments construct`; the others are committed.

Each kind of label has a different blind spot. **Synthetic** rows measure whether retrieval can find
what the scene model saw, not whether it saw correctly, so they inherit its mistakes - a model-derived
label once quietly agreed with a hallucination, and only reading the frame caught it. **Constructed**
rows are the only ones whose answers do not trace back to a model.

## Metrics

| Metric | Meaning |
| --- | --- |
| Top-1 IoU | Temporal intersection over union between the true interval and the answer the system is most confident in |
| Top-1 accuracy | Share of rows whose top-1 IoU is at least 0.3 |
| Best-of-5 IoU | The best IoU among the five most confident answers |
| Recall@0.3, @0.5 | Share of rows with any answer at that IoU or better |
| Exact text | For text rows, whether the normalised reading equals the drawn string |

"Top-1" is the most confident answer, not the earliest; results come back in time order, and scoring
the earliest would mark a correct but late answer as a miss.

## Benchmarks with known answers

```bash
moments construct --timelines 10 --ingest
moments construct --event-seconds 4 20 --segments 8 --timelines 8 --out bench/constructed_short.json --ingest
moments construct --append --timelines 5 --ingest       # add to an existing set without rebuilding it
```

Every other label traces back to the vision model, so none of them can catch that model being wrong.
`moments construct` builds the one kind of ground truth that does not: an edit list.

- **Splicing.** Described chunks from indexed videos are cut, normalised to one resolution and frame
  rate, and concatenated into new timelines. Each row's interval is where its chunk was placed.
- **Short events.** With `--event-seconds`, random slices of chunks are spliced instead, and the
  vision model writes a query for each slice, because the chunk's own description covers 30 seconds.
- **Drawn text.** Random codes (`K7X-4629`, avoiding glyphs that are easy to confuse) are drawn onto
  known frames; a text row's answer is the string and the window it was on screen.
- **Indexing.** `--ingest` adds each timeline to the library and indexes it exactly as the app does.

The edit list was checked by extracting frames: a drawn code is on screen inside its window and absent
a second before, and a two-minute timeline drifts 21 ms from its placements.

What an edit list cannot make exact is recorded in the rows rather than ignored:

- **Query wording** still comes from a model - the scene description, or the slice description.
- **Look-alikes.** Chunks from the same source video must differ in what the scene model listed for
  them, and rows sharing a source list each other as `confusable_with`. Short timelines draw
  round-robin across source videos, and rows whose query nearly repeats another in the same timeline
  are marked `ambiguous_with` and left out of tuning, since no setting can tell them apart.
- **Hard cuts.** Every constructed boundary is a cut, which real events are not; the boundary model was
  checked with and without cut features for exactly this reason.

## Results

### Quick mode on 30-second events

The first end-to-end run on 18 constructed events, before any tuning:

| | Mean IoU | Top-1 accuracy |
| --- | --- | --- |
| Hand-picked defaults | 0.266 | 50% |
| + learned boundaries | 0.292 | 50% |

The main finding was not the boundaries: **half the time quick mode picked the wrong 30-second
segment**, which no boundary adjustment can fix. That result - invisible to the synthetic benchmark,
which only checked agreement with the scene model - is what motivated tuning how a moment is picked.

### After tuning, across event lengths

Cross-validated by timeline over 113 constructed events from 18 timelines:

| Event length | Held-out rows | Defaults | Tuned |
| --- | --- | --- | --- |
| under 8 s | 11 | 0.151 | 0.260 |
| 8-15 s | 19 | 0.190 | 0.187 |
| 15-25 s | 31 | 0.420 | 0.517 |
| 25 s and over | 52 | 0.381 | 0.507 |
| **all** | 113 | **0.337** | **0.432** |

See [machine learning](machine-learning.md#tuned-retrieval-settings) for how the search works, why the
save gate checks each event length, and the caveats on these numbers.

## Routing

```bash
moments routing
```

The first decision a search makes is which executor answers it. The costly mistake is sending a moment
to the text reader - a minute or more of OCR that returns nothing. On the constructed benchmark the
planner did that whenever a scene was described by the text in it: "tennis court with NY text on wall",
"standing in front of HAB prediction poster", "Thanks For Watching title card".

| Planner | Labeled queries routed correctly |
| --- | --- |
| Original prompt | 16 of 22 (every miss a moment described by its text) |
| + "does the user already state the text?" | 22 of 22, but 8 of 10 on queries written afterwards |
| + a narrow confirmation question on every text route | 40 of 41, including 18 of 18 held out |

Two batches of queries were written after the changes, one deliberately avoiding every object the
prompts mention, so the result is not an artifact of the prompt's own examples. One keyword-soup query
from the benchmark ("MQ2 MQ7 gas sensor cost $10 patent document") is kept in the file as a hard case.

## Recording and comparing runs

```bash
moments runs list
moments runs show <run>
moments runs compare <run> <run>
```

Every benchmark and trainer writes a folder under `bench/runs/` with the command and every argument,
the seed, the git commit and whether the working tree had uncommitted changes (with a hash of the
diff), library versions, the GPU, a SHA-256 of every input file, per-step metrics, and a hash of every
model or result it wrote. Runs that crash or stop early are recorded too. `compare` lists what two runs
disagreed on and says when their inputs differed, so "the model improved" is checked against "the data
changed" first. Every trainer takes `--seed`; on data this size, a different seed moves the baseline,
not just the model, because it changes what is held out.

## Known gaps

- **Little source footage.** Every constructed timeline is cut from the same 33 described chunks,
  23 of them from one tennis video. More, and more varied, indexed video would make every number here
  more trustworthy.
- **Text rows are built but not yet reported.** The 20 drawn-code rows exist for measuring OCR, and
  OCR accuracy on them has not been written up.
- **Verified mode is not tuned or benchmarked at scale.** The constructed results above are quick
  mode; the learned selection components for verified mode are waiting on logged searches.
- **Three hand-labeled rows** remain the only answers checked by a person.
