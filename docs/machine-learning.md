# Machine learning

The pretrained models - SigLIP 2, Whisper, Qwen 3.5 and nomic-embed-text - are used as they are.
On top of them, Moments learns seven small components of its own, all from data the system produces
itself. None needs a hand-labeled example, each is inert until trained, and every path falls back to
the hand-written behaviour when its model file is absent.

- [Where the training signal comes from](#where-the-training-signal-comes-from)
- [Status at a glance](#status-at-a-glance)
- [Tuned retrieval settings](#tuned-retrieval-settings)
- [Learned clip boundaries](#learned-clip-boundaries)
- [Candidate ranking and conformal prediction](#candidate-ranking-and-conformal-prediction)
- [Candidate pre-filter](#candidate-pre-filter)
- [Self-supervised query adapter](#self-supervised-query-adapter)
- [When to stop verifying](#when-to-stop-verifying)

## Where the training signal comes from

| Signal | What produces it | Trains |
| --- | --- | --- |
| **The verifier's verdicts** | Every verified search logs each candidate's features and whether the vision model confirmed it | Pre-filter, ranker, conformal threshold, stopping policies |
| **Co-occurrence in time** | Text the scene model wrote about a stretch of video, paired with that stretch's frames | Query adapter |
| **Edit lists** | Benchmark videos spliced together by `moments construct`, so every answer is known exactly | Boundary model, tuned retrieval settings |

The first two trace back to the vision model: the models they train can become as good as it is,
never better, and they inherit its mistakes. Edit lists are the one source that does not - an event
is at 2:30 because it was placed there - which is why the components with the clearest measured
gains are the ones trained on them. See [evaluation](evaluation.md) for how the benchmarks are built.

## Status at a glance

| Component | Technique | Trained on this machine | Result |
| --- | --- | --- | --- |
| Tuned retrieval settings | Random search + hill climbing, cross-validated by timeline | Yes | Held-out top-1 IoU **0.337 → 0.432** on 113 events |
| Learned boundaries | Softmax over positions, linear features | Yes | Boundary error **2.9 s → 1.4 s** on simulated proposals |
| Candidate ranker | Pointwise / LambdaRank / ListNet ablation | Waiting on data | Needs 20+ logged verified searches |
| Conformal set | Split conformal prediction | Waiting on data | Trained alongside the ranker |
| Candidate pre-filter | Logistic regression distilled from the verifier | Waiting on data | Superseded by the ranker when both exist |
| Query adapter | Contrastive low-rank residual (InfoNCE) | Yes | Held-out recall@1 42.5% → 46.0%, within noise |
| Stopping policies | Optimal stopping, fitted Q iteration | Waiting on data | Validated on a synthetic fixture |

Every verified search run in the app logs training rows for the ranker, pre-filter and stopping
policies, so they become trainable with use.

## Tuned retrieval settings

```bash
moments tune --dataset bench/constructed.json --dataset bench/constructed_short.json
moments tune --dataset bench/constructed_short.json --evaluate     # score saved settings, no search
```

Which window wins a quick search is decided by eight scoring weights (`retrieval.DEFAULT_SCORING`)
and twelve evidence-map, candidate, refinement and NMS settings, all originally chosen by hand.

Searching them through the real pipeline would cost a planner call per row per trial. Everything
after retrieval calls no model, though, so each benchmark row's plan and retrieval hits are cached
once and a trial replays only the model-free stages - `pipeline.locate_candidates`, NMS and the
learned boundaries - in about half a second for every row. A unit test, and a live check on real
rows, confirm the replay returns exactly what the pipeline returns.

Twenty settings and a hundred rows will happily fit a benchmark instead of a task, so:

- **Cross-validation by timeline.** Each fold is searched on the other timelines and scored on its own.
- **A save gate.** Settings are saved only if they beat the defaults on held-out timelines in most
  folds *and* no event-length bucket loses more than 0.05 IoU.
- **One mode.** They apply only to quick searches, so padding that suits a quick answer cannot starve
  the verifier of context.

### Why the gate checks event lengths

The first tuning used only whole 30-second events and looked good: held-out top-1 IoU 0.406 to 0.523,
better in 5 of 5 folds. Scored afterwards on 4-20 second events it had never seen, it was better on
average and much worse for one length - which the average hid:

| Event length | Rows | Defaults | Tuned on 30 s events only |
| --- | --- | --- | --- |
| under 8 s | 11 | 0.151 | 0.376 |
| 8-15 s | 19 | 0.190 | **0.093** |
| 15-25 s | 24 | 0.370 | 0.428 |

Re-tuned on both sets together - 113 events from 18 timelines - with the stricter gate:

| Event length | Held-out rows | Defaults | Tuned |
| --- | --- | --- | --- |
| under 8 s | 11 | 0.151 | 0.260 |
| 8-15 s | 19 | 0.190 | 0.187 |
| 15-25 s | 31 | 0.420 | 0.517 |
| 25 s and over | 52 | 0.381 | 0.507 |
| **all** | 113 | **0.337** | **0.432** — better in 5 of 5 folds |

The search turned recursive refinement off, cut candidate padding from 10 s to 3.9 s and the gap that
joins evidence from 10 s to 2.2 s, and raised the floor a region must clear to become a candidate.
With recursion off, the window-score weights and most recursive settings in the saved file do nothing.

**Caveats.** Timelines draw from a small pool of source footage - 23 of its 33 chunks are one tennis
video - so folds separate timelines but not footage, and the held-out numbers are probably somewhat
optimistic. Short-event queries are worded by the vision model. 8-15 second events are unchanged
rather than improved.

## Learned clip boundaries

```bash
moments boundaries                                           # train on constructed intervals
moments boundaries --proposals bench/results/<run>.json      # real proposals instead of simulated
```

A quick-mode answer is an evidence region with padding, which is much of why its IoU is low. The
per-second frame embeddings already hold two curves that say where an event starts and ends: how
well each second matches the query, and how different it looks from the second before. Each proposed
boundary is scored at every second within a window by a linear model over features of both curves -
a softmax over positions, trained with cross-entropy against the true boundary from an edit list.

Held out by whole timeline, across three seeds, from simulated proposals:

| Method | Mean IoU | Boundary error |
| --- | --- | --- |
| Proposal as given | 0.835-0.850 | ~2.9 s |
| Snap to the biggest visual cut | 0.817-0.841 | ~3.0 s |
| Sit on the biggest similarity step | 0.877-0.895 | ~2.2 s |
| Learned, with cut features | 0.892-0.909 | ~1.7 s |
| **Learned, without cut features** | **0.916-0.935** | **~1.4 s** |

The cut features were expected to flatter the model, since every constructed boundary is a hard cut.
They did the opposite - the source footage has cuts of its own, and the two cut features came out
nearly equal and opposite - so the saved model leaves them out. End to end on 18 constructed events in
quick mode, the model raised mean IoU from 0.266 to 0.292; it only helps where retrieval already found
the right segment, since a boundary moves at most a few seconds.

Real proposals are usually further from the truth than the search window, so `--proposals` has too few
reachable boundaries to train on until there are more benchmark rows.

## Candidate ranking and conformal prediction

```bash
moments rank --learning-curve --ablate
```

A verified search spends nearly all its time asking the vision model about candidates, and walks them
from the top. What matters is therefore the **order**, so the ranker optimises the order directly.
The same small network is trained under three losses, making the comparison an ablation rather than
a claim:

| Loss | Objective |
| --- | --- |
| Pointwise | Binary cross-entropy per candidate |
| Pairwise | RankNet over confirmed/rejected pairs, weighted by the NDCG each swap changes (LambdaRank) |
| Listwise | Softmax cross-entropy over a whole search (ListNet) |

All three are scored against retrieval's own ordering, and nothing is saved unless one beats it.

**Split conformal prediction** then decides how many candidates deserve a vision call. On a separate
calibration split it finds the score threshold that keeps a confirmed candidate in the set for at
least `1 - alpha` of searches (90% by default), and a third split reports the coverage actually
delivered and the fraction of candidates it cost. The guarantee holds whether or not the scores are
calibrated, and it is about coverage across searches, not about any single candidate. Searches, not
candidates, are the unit of every split, because a search's candidates are not independent.

**Exploration keeps the training data honest.** A ranker that only sends its favourite candidates to
the verifier only ever collects data about its favourites: it could learn that candidates like X are
worthless and never see evidence it was wrong. So one rejected candidate in ten is verified anyway,
logged as `explored` with the probability that sampled it, and pointwise training divides by that
probability. `--learning-curve` shows whether more data or a better model would help more, and
`--ablate` shows which feature groups carry the signal.

On a synthetic fixture the ablation recovered exactly the two feature groups the fixture's labels
were built from, and conformal coverage landed at or above its target for every loss.

## Candidate pre-filter

```bash
moments distill --dry-run
moments distill
```

The first learned component: a logistic regression that predicts the verifier's verdict from the
twelve features retrieval already computed, distilling the 4B vision model into a model that runs in
microseconds. The strongest few candidates are always kept, so a badly fitted filter can cost time but
never empty the list. When a ranker is trained, the pipeline uses the ranker instead.

## Self-supervised query adapter

```bash
moments adapt --dry-run
moments adapt
```

Each indexed chunk provides free training pairs: the text the scene model wrote about a stretch of
video and the frame embeddings for that stretch. A low-rank residual on the identity is fitted from
query-text embeddings into the frame-embedding space with a symmetric InfoNCE loss, adapting retrieval
to the user's own footage. It applies to queries only, never to stored vectors, so training a new one
never invalidates an index.

Four details decide whether its score means anything:

- **Negatives come from the same video**, because search never ranks one video's chunks against
  another's; cross-video negatives would teach a distinction retrieval never has to make.
- **Spans are held out, not rows.** A span contributes a dozen texts and several frames; splitting by
  row would score captions against frames their own twins trained on.
- **Overlap and repeated wording are masked, not punished.** Chunks overlap by half and the scene
  model reuses search terms across neighbours, so those pairs leave the loss instead of being pushed
  apart.
- **Training maximises what search scores**: the best of five views, not the whole frame alone.

On three videos (430 texts, 30 spans) held-out recall@1 moved from 42.5% to 46.0% with recall@5
unchanged - three texts out of 87, inside the noise. The adapter is kept because it does no harm and
improves with footage, not because it has shown a gain.

## When to stop verifying

```bash
moments replay                          # every confirmed match counts
moments replay --goal first             # only the first one does
moments replay --value-seconds 120 --ranker
```

Choosing when to stop asking the vision model is a sequential decision problem, and learning one by
trial would cost a minute or more per episode. Each verified search logs every candidate's verdict and
how long it took, so a search can be **replayed**: a policy asks for candidates in its own order, the
replay answers from the log and charges the logged seconds, and an episode costs microseconds.

The reward is `value-seconds` per confirmed match minus the seconds spent. On held-out searches it
compares verifying everything, a fixed top-k, a static threshold, **optimal stopping** by backward
induction over the ordered list, and **fitted Q iteration**, which learns the value of verifying from
replayed outcomes instead of trusting the probabilities. A unit test checks the backward induction
against brute force on 300 random lists. When fitted Q beats optimal stopping, the probabilities were
miscalibrated or candidates were not independent - which is exactly what it showed on a fixture built
with deliberately weak probabilities.

The replay can only answer for candidates that were verified: every logged candidate until a ranker
is trained, and afterwards the explored ones.
