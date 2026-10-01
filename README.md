# CEDAR — biomedical ontology equivalence ranking (OAEI Bio-ML 2026)

CEDAR is a system for the **local equivalence ranking** task of the
[OAEI Bio-ML 2026](https://huggingface.co/datasets/OAEI-ML/bio-ml) track developed at  **Massachusetts Institute of Technology CTL** : for every source class,
rank the 100 candidate target classes the organisers provide so that the equivalent class comes
first. It covers all three ontology pairs — NCIT–DOID, SNOMED CT–FMA and SNOMED CT–NCIT.

CEDAR combines two scorers per (source, candidate) pair:

1. a transparent **lexical reference** (exact and fuzzy matching of normalised labels and
   synonyms, fixed weights, no learning), and
2. a **pairwise equivalence model** fine-tuned separately for each ontology pair on that pair's
   TRAIN split, reading a compact text description of both classes (label, synonyms, definition,
   one-hop parents).

The two scores are combined per query with a weight chosen on a held-out slice of TRAIN. Every
model choice — checkpoint epoch, training seed and combination weight — is made on that TRAIN
slice (**train-dev**), never on VALID; the rules were fixed in advance in
[PROTOCOL.md](PROTOCOL.md). No external mapping resource (UMLS, Mondo) and no cross-reference
annotation is used.

## Results — official VALID

All numbers are produced by the organisers' `score_local.py`, and every ranking passes their
`validate_ranking.py`. Only VALID results are reported; TEST gold is not available.

| Pair (VALID queries) | System | MRR | Hits@1 | Hits@5 | Hits@10 |
|---|---|---|---|---|---|
| NCIT-DOID (634) | Lexical reference | 0.9412 | 0.9227 | 0.9590 | 0.9732 |
| | Zero-shot base model | 0.0693 | 0.0205 | 0.0726 | 0.1246 |
| | **CEDAR** (seed 0; prob, α=0.9) | **0.9682** | 0.9590 | 0.9748 | 0.9858 |
| SNOMED-FMA (1,178) | Lexical reference | 0.9109 | 0.8854 | 0.9389 | 0.9576 |
| | Zero-shot base model | 0.0479 | 0.0119 | 0.0467 | 0.0798 |
| | **CEDAR** (seed 1; minmax, α=0.5) | **0.9598** | 0.9465 | 0.9745 | 0.9822 |
| SNOMED-NCIT (3,614) | Lexical reference | 0.8936 | 0.8533 | 0.9419 | 0.9582 |
| | Zero-shot base model | 0.1084 | 0.0346 | 0.1428 | 0.2402 |
| | **CEDAR** (seed 1; minmax, α=0.6) | **0.9636** | 0.9496 | 0.9801 | 0.9851 |
| **Macro over the 3 pairs** | Lexical reference | 0.9152 | 0.8872 | 0.9466 | 0.9630 |
| | Zero-shot base model | 0.0752 | 0.0223 | 0.0873 | 0.1482 |
| | **CEDAR** | **0.9639** | 0.9517 | 0.9765 | 0.9843 |

Macro-MRR over the three pairs: **0.9639** (lexical reference 0.9152). A uniformly random ranking
of 100 candidates scores MRR ≈ 0.052.

### Training seeds and selection

| Pair | Seed | Training | Best epoch | train-dev MRR: model alone / with lexical (selection) | VALID MRR: model alone | VALID MRR: with lexical (train-dev α) |
|---|---|---|---|---|---|---|
| NCIT-DOID | 0 ← selected | 4/4 epochs, 0 skipped steps¹ | 3 | 0.9550 / **0.9679** (prob, α=0.9) | 0.9642 | 0.9682 |
| NCIT-DOID | 1 | 4/4 epochs, 0 skipped steps¹ | 2 | 0.9604 / **0.9667** (minmax, α=0.6) | 0.9541 | 0.9646 |
| NCIT-DOID | 2 | 4/4 epochs, 0 skipped steps | 4 | 0.9566 / **0.9665** (minmax, α=0.6) | 0.9542 | 0.9674 |
| SNOMED-FMA | 0 | 4/4 epochs, 0 skipped steps | 3 | 0.9567 / **0.9583** (minmax, α=0.9) | 0.9588 | 0.9627 |
| SNOMED-FMA | 1 ← selected | 4/4 epochs, 0 skipped steps | 3 | 0.9512 / **0.9623** (minmax, α=0.5) | 0.9523 | 0.9598 |
| SNOMED-FMA | 2 | **diverged** — training stopped after 1/4 epochs (fp16 non-finite forward passes; guard stop) | – | – | – | – |
| SNOMED-NCIT | 0 | 4/4 epochs, 0 skipped steps | 3 | 0.9530 / **0.9595** (minmax, α=0.5) | 0.9512 | 0.9609 |
| SNOMED-NCIT | 1 ← selected | 4/4 epochs, 0 skipped steps | 3 | 0.9605 / **0.9637** (minmax, α=0.6) | 0.9559 | 0.9636 |
| SNOMED-NCIT | 2 | **diverged** — training stopped after 2/4 epochs (fp16 non-finite forward passes; guard stop) | – | – | – | – |

¹ Trained with an earlier version of the numerical guard that stopped on any non-finite step, so
0 is exact.

* **Two of nine runs diverged**, both on SNOMED pairs: the fp16 forward pass began to produce
  non-finite logits, the loss scale collapsed and the driver's guard stopped the run. Following
  PROTOCOL.md amendment 1 (written before any fine-tuned SNOMED VALID result existed) they are
  reported as diverged and excluded; no replacement seed was trained. The same instability is not
  deterministic per seed: two earlier SNOMED–FMA attempts that stopped in epoch 1 trained cleanly
  when re-run.
* **Seed spread** (completed seeds, VALID): model alone 0.954–0.964 (NCIT–DOID), 0.952–0.959
  (SNOMED–FMA), 0.951–0.956 (SNOMED–NCIT); with the lexical combination 0.965–0.968, 0.960–0.963,
  0.961–0.964.
* **Selection on train-dev, not VALID.** The train-dev rule picked the VALID-best seed on two
  pairs; on SNOMED–FMA it picked seed 1 (0.9598) over seed 0 (0.9627), a gap inside the seed
  spread. Selecting on VALID would have added about 0.001 macro-MRR and is not done.
* Fine-tuning is what makes the pairwise model useful: zero-shot it is at chance (macro-MRR
  0.075); fine-tuned it reaches 0.95–0.96 alone, and the lexical combination adds 0.004–0.013.

### Hard subset

Queries whose gold has no exact normalised-name match with the source (VALID):

| Pair | Hard queries (of VALID) | Lexical MRR | CEDAR MRR | Δ MRR [95% bootstrap CI] | CEDAR better / equal / worse (queries) | Easy subset: lexical → CEDAR |
|---|---|---|---|---|---|---|
| NCIT-DOID | 102 (16.1%) | 0.6641 | 0.8074 | +0.143 [+0.084, +0.208] | 31 / 69 / 2 | 0.9944 → 0.9991 |
| SNOMED-FMA | 238 (20.2%) | 0.5715 | 0.8136 | +0.242 [+0.193, +0.291] | 104 / 116 / 18 | 0.9968 → 0.9968 |
| SNOMED-NCIT | 912 (25.2%) | 0.6175 | 0.8708 | +0.253 [+0.229, +0.278] | 407 / 460 / 45 | 0.9868 → 0.9950 |


Paired bootstrap over hard queries (10,000 resamples, seed 0) against the lexical reference.
Where an exact name match exists, both systems are near ceiling; CEDAR's gain is concentrated on
the hard queries. On SNOMED–NCIT the largest group is SNOMED *medicinal product* sources (307 of
308 are hard; lexical MRR 0.65 → 0.98), whose names carry a product wording around the ingredient
while the NCIT gold is the substance itself. When a *wrong* candidate carries the source's exact
name (4, 36 and 6 queries on SNOMED–FMA, SNOMED–NCIT and NCIT–DOID) CEDAR mostly cannot overrule
it. The hard-subset analysis script is not part of this repository.

### Development ablations (NCIT–DOID, VALID)

Run on NCIT–DOID before the SNOMED pairs; each fine-tuned row is a single run (seed spread ≈ 0.01).

| Positives : negatives | train-dev MRR (best epoch) | VALID MRR (model alone) | Note |
|---|---|---|---|
| 1 : 1 | 0.9400 (ep. 3) | 0.9384 | works, weakest |
| **1 : 5** | **0.9550** (ep. 3) | **0.9642** | used |
| 1 : 10 | 0.0529 | 0.0573 | training collapsed to the class prior (twice, two seeds) |

| Representation | Zero-shot MRR | Fine-tuned (1:5) MRR |
|---|---|---|
| A label | 0.2481 | 0.8248 |
| B + synonyms | 0.1815 | 0.9542 |
| C + definition | 0.1402 | 0.9576 |
| **D** + one-hop parent labels | 0.0693 | 0.9642 |

Zero-shot, longer descriptions make the base model worse; after fine-tuning the order reverses.
Synonyms are the big step; B, C and D differ by less than the seed spread. A 100-way variant
(all candidates as options of one question) is ~10× faster zero-shot but far weaker (MRR ≤ 0.29),
and pairwise fine-tuning does not transfer to it.

## How CEDAR works

For every source class S and each of its 100 candidates T:

1. **Entity text** (`src/ontology_loader.py`, `src/ofn_loader.py`, `src/build_entity_text.py`).
   Ontologies are streamed once and cached: preferred label, synonyms, definition, one-hop parent
   labels. Annotation properties are read through a role **whitelist**, so cross-reference and
   mapping properties (e.g. `oboInOwl:hasDbXref`, `skos:*Match`, NCIT `P207`/`P208`/`P375`) never
   reach a scorer.
2. **Lexical score** (`src/lexical_baseline.py`): exact label match, exact name match, best token
   Jaccard and best character similarity over all name pairs, averaged with fixed weights.
3. **Pair state** (`src/build_pair_states.py`): `SOURCE CONCEPT … / TARGET CONCEPT …`, each
   concept fitted to its own token budget (label always whole, then synonyms, parents and the
   definition cut at a word boundary) so the target is never truncated away.
4. **Model score** (`src/score_laya.py`): one shared binary question (EQUIVALENT /
   NOT_EQUIVALENT) asked of every pair state; the score is the full-precision logit margin.
5. **Combination** (`src/ensemble.py`): `α · L + (1 − α) · X`, with L and X either min-max
   normalised per query or (L = P(EQUIVALENT), X raw); (mode, α) tuned on train-dev.
6. **Ranking**: candidates sorted by descending score, ties kept in pool order; the output is
   always an exact permutation of the given pool, in the official LIST form.

### Ontology roles

| Role | NCIT | DOID | FMA | SNOMED CT |
|---|---|---|---|---|
| label | `rdfs:label`, `P108` | `rdfs:label` | `rdfs:label`, `fma:preferred_name` | `skos:prefLabel` (en-us, then en, then en-gb) |
| synonyms | `P90` | `oboInOwl:has{Exact,Related,Narrow,Broad}Synonym` | `fma:synonym` | FSN without its semantic tag, `skos:altLabel` |
| definition | `P97`, `P325` | `obo:IAO_0000115` | `fma:definition` | `skos:definition` |
| parents | named superclasses + named members of an equivalent-class intersection | same | same | same; attribute concepts (declared as OWL properties): super-property |

Coverage over the entities of the BioML candidate files: label 100% for all four ontologies;
synonyms NCIT 70%, DOID 69%, FMA 35%, SNOMED 67%; definition NCIT 87%, DOID 81%, FMA 2%, SNOMED
3%; parent context ≥ 99.99%.

**SNOMED CT** is read from its OWL Functional Syntax serialisation by `src/ofn_loader.py`: in the
RDF/XML serialisation class expressions are flattened into blank nodes, and parents that sit inside
`SubClassOf(:X ObjectIntersectionOf(:Parent …))` would otherwise be lost. `src/verify_snomed.py`
cross-checks the extraction against the RDF/XML file with an independent blank-node-resolving
parser (for the release used: 0 annotation and 0 parent mismatches over all 42,429 SNOMED task
entities).

## Implementation

The pairwise equivalence model is **LAYA**
([model card](https://huggingface.co/convaiinnovations/laya),
[code](https://github.com/NandhaKishorM/laya)), a non-autoregressive typed-decision model, used
through its Python API (`laya==0.3.22`) with the English checkpoint at a pinned revision
(`environment.txt`). Code and run names keep the `laya` prefix.

* **Question.** One `choice` question with neutral option labels, asked of every pair state:
  "Determine whether the source ontology class and target ontology class denote the same
  biomedical concept. Equivalence means the same concept, not merely related, broader, narrower,
  parent, child, or frequently associated." Options EQUIVALENT / NOT_EQUIVALENT.
* **Scores.** LAYA rounds probabilities to 4 decimals, so `src/score_laya.py` also captures the two
  option logits at full precision and ranks by their margin (monotone in P(EQUIVALENT)); it asserts
  that the API probability equals sigmoid(margin / T) on every pair.
* **Fine-tuning** (`src/finetune_laya.py`): a driver around LAYA's official single-device
  fine-tuning script, whose item builder, collate, forward and temperature fit are used unchanged;
  the optimisation step is that script's RLCD step (Gaussian logit exploration σ 0.4→0.1, proper
  scoring-rule reward, group-mean-baseline policy gradient, plus cross-entropy). Changes: the
  English checkpoint with its own `max_len` 512 / `head_max_len` 192; per-epoch checkpoints ranked
  on train-dev; a numerical guard — a non-finite loss is skipped exactly as the official fp16
  GradScaler does, but counted, logged and bounded (abort after 20 in a row or >1% of an epoch).
* **Training data** (`src/build_finetune_data.py`): official JSONL schema, TRAIN only, 1 positive
  : 5 negatives from the query's own pool (3 lexically hardest + 2 random).
* **Throughput** on one NVIDIA L40S: ~360–530 pairs/s scoring (batch size 16, bf16 autocast), ~2.6 GB
  GPU memory; fine-tuning 6–7 steps/s (micro-batch 8), 9.3 GB; 4 epochs take ~36 min (NCIT–DOID), ~48 min
  (SNOMED–FMA) and ~2.6 h (SNOMED–NCIT).

## Reproducing

Python 3.12, one GPU per job; `scripts/*.slurm` are SLURM jobs whose partition names are
site-specific placeholders (`gpu` / `cpu`). Submit them from the repository root after
`mkdir -p logs`. `scripts/env.sh` (sourced by every job) sets the venv, `HF_HOME=cache/hf_cache`
and offline mode.

```bash
# environment
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # full freeze: environment.txt
mkdir -p logs

# official data (revision 2026), scoring kit, the base model's official fine-tuning script, base model
git clone https://github.com/liseda-lab/OAEI-Bio-ML external/OAEI-Bio-ML
git clone https://github.com/NandhaKishorM/laya external/laya
HF_HOME=cache/hf_cache .venv/bin/hf download OAEI-ML/bio-ml --repo-type dataset --revision 2026 --local-dir data/bio-ml
HF_HOME=cache/hf_cache .venv/bin/hf download convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
# SNOMED CT: licence-restricted and NOT included. Holders of a SNOMED CT licence can obtain the
# release used by Bio-ML from the organisers; place the OFN file under data/private/ (git-ignored)
mkdir -p data/private && ln -s /path/to/<snomed release>.owl data/private/
sbatch --export=ALL,SNOMED_RDFXML=/path/to/<snomed release>.rdfxml.owl scripts/snomed_cache.slurm
for O in NCIT DOID FMA; do .venv/bin/python src/ontology_loader.py --ontology $O; done   # on a compute node (NCIT: several GB RAM)
bash scripts/smoke_test.sh                     # CPU sanity check on NCIT-DOID

# per pair (CPU): lexical VALID / train-dev, internal splits, rep-D 1:5 fine-tuning data
for P in NCIT-DOID SNOMED-FMA SNOMED-NCIT; do sbatch --export=ALL,PAIR=$P scripts/prep_pair.slurm; done
# zero-shot base model on VALID
sbatch --export=ALL,PAIR=<pair>,SKIP_LEXICAL=1 scripts/run_valid.slurm
# fine-tune, training seeds 0/1/2 (SNOMED-NCIT: add -t 06:00:00 and SCORE_VALID=0, then score
# VALID with run_valid.slurm MODEL=checkpoints/<run>/best METHOD=laya_bioml_neg5[_seedN])
sbatch --export=ALL,PAIR=<pair>,TRAIN_SEED=<s>,SKIP_DATA=1 scripts/finetune.slurm
# full train-dev scores per checkpoint
sbatch --export=ALL,PAIR=<pair>,TRAIN_SEED=<s>,SKIP_PREP=1,FP32_VALID=0 scripts/posthoc.slurm
# freeze (CPU): train-dev alpha per seed, seed selection, final VALID ranking
sbatch --export=ALL,PAIRS_TO_RUN=<pair> scripts/finalize_valid.slurm
# TEST rankings with the frozen system, validation and packaging (nothing is uploaded)
sbatch --export=ALL,PAIRS_TO_RUN=<pair> scripts/run_final_test.slurm
sbatch scripts/package_final.slurm
```

Validation and scoring always go through the official kit:

```bash
python external/OAEI-Bio-ML/scoring_kit/validate_ranking.py data/bio-ml/<pair>/local.valid.cands.tsv outputs/<pair>/valid/<run>/ranking.tsv
python external/OAEI-Bio-ML/scoring_kit/score_local.py      outputs/<pair>/valid/<run>/ranking.tsv data/bio-ml/<pair>/local.valid.cands.tsv
```

## Competition-rule compliance

* No UMLS, no Mondo; no cross-reference or mapping annotation (role whitelist; the SNOMED CT file
  has no such property).
* Fine-tuning reads `local.train.cands.tsv` only; epoch, seed and combination weight are chosen on
  train-dev; VALID is used for reporting only; TEST pools are only ranked, never used to tune
  anything, and no attempt is made to infer TEST gold.
* Each query is ranked independently from its own 100 candidates.
* SNOMED CT content is not redistributed: no ontology file and no SNOMED-derived text is in this
  repository.

## Layout

```
src/        ontology_loader, ofn_loader, build_entity_text, build_pair_states, lexical_baseline,
            score_laya, finetune_laya, build_finetune_data, ensemble, select_final, evaluate,
            make_internal_splits, make_submission, inspect_data, verify_snomed,
            verify_loader_rdflib, score_laya_100way, smoke_laya, bench_batch, utils
scripts/    env.sh, smoke_test.sh, smoke_gpu.slurm, snomed_cache.slurm, prep_pair.slurm,
            run_valid.slurm, finetune.slurm, posthoc.slurm, finalize_valid.slurm,
            run_final_test.slurm, package_final.slurm, ablation_100way.slurm
PROTOCOL.md requirements.txt environment.txt
```

Generated at run time (git-ignored): `data/`, `external/`, `cache/`, `checkpoints/`, `outputs/`, `logs/`.

## References

* OAEI Bio-ML 2026 dataset: <https://huggingface.co/datasets/OAEI-ML/bio-ml>; scoring kit:
  <https://github.com/liseda-lab/OAEI-Bio-ML>.
* LAYA: <https://huggingface.co/convaiinnovations/laya>, <https://github.com/NandhaKishorM/laya>.
* Ontologies and their licences: NCIT (CC BY 4.0), DOID (CC0 1.0), FMA (FMA licence), SNOMED CT
  (SNOMED CT Affiliate Licence; not redistributed here).
