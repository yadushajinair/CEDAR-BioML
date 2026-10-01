# CEDAR evaluation protocol (OAEI Bio-ML 2026, local equivalence ranking)

Fixed on 2026-09-30, before any fine-tuned SNOMED–FMA or SNOMED–NCIT VALID result existed. It
fixes every choice that could otherwise be tuned on VALID and applies identically to all three
pairs (NCIT–DOID, SNOMED–FMA, SNOMED–NCIT).

## Data use

| Split | Used for |
|---|---|
| `local.train.cands.tsv` minus train-dev | gradient updates only (fine-tuning items, 1 positive : 5 negatives) |
| **train-dev** (10% of TRAIN queries, split by source entity, data seed 0) | epoch selection (first ≤400 train-dev queries); ensemble (mode, α); choice of the training seed |
| `local.valid.cands.tsv` (official VALID) | **reporting only**: scored once per system after every choice above is made; never used for any choice |
| `local.test.cands.tsv` | ranked once with the frozen system; no labels exist locally |

## Fixed system

* Entity representation **D**: preferred label + synonyms + definition + one-hop parent labels,
  each concept fitted to its own token budget so that no pair state is truncated.
* Pair-specific pairwise equivalence model: the base model is fine-tuned separately for each pair,
  on that pair's TRAIN only, with the base model's official RL fine-tuning recipe (see README,
  [Implementation](README.md#implementation)): 4 epochs, micro-batch 8, AdamW (encoder 2.5e-5 /
  head 1e-4, weight decay 0.01), cosine schedule, fp16 autocast.
* **Negatives 1:5** from the query's own candidate pool: the ⌈5/2⌉ = 3 lexically closest plus 2
  random; another TRAIN gold of the same source is never a negative.
* **Seeds**: data seed 0 for every run; training seeds 0, 1, 2 for every pair.
* Final score per candidate: `src/ensemble.py` combination of the lexical reference score and the
  model's logit margin, with **(mode, α) tuned on the full train-dev split** over the fixed grid
  {minmax, prob} × {0.0, 0.1, …, 1.0}.
* **Seed selection**: per pair, the training seed whose train-dev ensemble MRR (with its own
  train-dev-tuned α) is highest; ties go to the lower seed. The other seeds are reported on VALID
  as the seed spread.

## Reported on VALID (per pair, official `score_local.py`)

Lexical reference · zero-shot base model (representation D) · fine-tuned model per seed · lexical +
fine-tuned model per seed with train-dev α · the selected system. Plus a hard-subset analysis (gold
has no exact normalised-name match with its source) with a paired bootstrap against the lexical
reference. Macro-MRR over the three pairs is computed from the selected systems.

## Not used

UMLS, Mondo, any cross-reference or mapping annotation (the loaders read a role whitelist; the
SNOMED CT file contains no such property), and anything about TEST beyond its candidate pools.

## Amendment 1 — a training seed that diverges (2026-09-30, 22:35 EDT)

Written while one SNOMED–NCIT run showed accelerating non-finite fp16 forward passes, and before
any SNOMED VALID result of a fine-tuned model existed (only the lexical and zero-shot VALID
baselines had been scored).

* A training run stopped by the fine-tuning driver's numerical guard (non-finite loss on more than
  20 consecutive steps or on more than 1% of an epoch's steps, or non-finite weights) is reported
  as **diverged**. It is excluded from seed selection; none of its checkpoints is used for VALID or
  TEST.
* Seed selection is then over the seeds that completed all 4 epochs (at least one is required).
  No replacement seed is trained.
* A seed that completed training but lacks a downstream file is a hard error (never silently
  excluded).
