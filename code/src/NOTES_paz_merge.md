# NOTES_paz_merge.md

Validation record for two pipeline changes made 2026-10-10, both triggered by discrepancies found
while answering reviewer questions Q1-Q7 on `is360_Breast_manuscript_v2.docx`. Referenced from
comments in `models.py`, `config.py`, and `evalkit.py` — read this before trusting those comments
blindly, and before citing either change as "resolved" to anyone outside this thread.

---

## 1. `select_epsilon()` / entropy-based PAZ gating — merged into the live pipeline

**The gap (Q1).** The manuscript states the Protocol v1.1 per-task epsilons were chosen by "max
accuracy_confident subject to >=50% coverage" on DEV. That is true, but the function that did the
choosing (`select_epsilon()`) lived only in a standalone script (`epsilon_sweep.py`), not in the
shipped pipeline. `models.py` also still used a single global margin-epsilon for every task,
including `density_img` (3-class), where margin scoring is the documented weak point (ignores all
posterior mass below the top two classes).

**What changed.** `models.py` gained `margin_score()`, `entropy_score()`, `paz_gate(method=...)`
(auto-resolves margin for 2-class, entropy for 3+-class), `select_epsilon()`, and `paz_task_validity()`,
merged from `PAZ_Entropy_Margin_Patch.zip` — but NOT that patch's `ComBatTransformer`, which is unused
in this pipeline. `make_model()` and `NeuroSyncSLDA` were left untouched; the patch bundle's own copy
of `models.py` is explicitly self-labeled "reconstructed... do not overwrite blind" and is missing
`make_model()` entirely, so a wholesale replacement would have broken the pipeline. `config.py` gained
`PAZ_EPSILON_ENTROPY = 0.90`, the Protocol v1.1 value `select_epsilon()` selects for `density_img` on DEV.

**Validation performed:**

1. Called the merged, native `models.select_epsilon()` against the real `exp_20261009_002_DEV`
   out-of-fold posteriors for all 4 tasks, and compared to `epsilon_sweep_full.json` (the standalone
   script's saved output). Result: **exact match to 4 decimal places**, all 4 tasks
   (epsilon, accuracy_confident, coverage) — severity_roi (eps=0.15), lesion_vs_normal_roi (eps=0.20),
   abnormal_img (eps=0.05), density_img (eps=0.90, entropy).
2. `bc.py check`: 0 FAIL, 2 WARN (both pre-existing: optional `h5py`, 2 CPU cores) — no regression.
3. **End-to-end smoke test** (`bc.py experiments --partition DEV --repeats 1 --boot 100`) caught a real
   call-site break that step 1-2 did not: `s4_experiments.py` line 72 did `mg, amb = paz_gate(Pbar)`,
   assuming the pre-merge 2-value return. The merged `paz_gate()` returns 3 values
   `(score, is_ambiguous, method_used)`. Fixed by unpacking all 3, and by renaming the oof CSV column
   from `paz_margin` to `paz_score` + `paz_method` — the old fixed name would have silently mislabeled
   `density_img`'s entropy scores as "margin" once auto-resolution took effect.
4. Re-ran the smoke test clean through all 4 tasks x 5 variants x 4 models. Confirmed directly in the
   generated oof CSVs (not inferred) that auto-resolution behaves correctly in a live run:
   `density_img` -> `paz_method=entropy` (scores spread across 0-1, consistent with normalized entropy),
   `severity_roi` -> `paz_method=margin`.

**Lesson:** a call-site audit by reading code (done once already, before this merge) is not the same
guarantee as actually running the pipeline. The audit missed this break; the smoke test caught it.

**Known, deliberately unfixed interaction:** `s5_report.py` line 87 sweeps epsilon over
`np.linspace(0, 0.6, 13)` for the PRIMARY_TASK figure only. Since `PRIMARY_TASK = severity_roi`
(2-class, margin), this range is appropriate and the figure is unaffected. It would NOT be appropriate
if `PRIMARY_TASK` were ever changed to a 3+-class task — flagged for whoever touches that constant next.

---

## 2. Risk-coverage AURC / AURC gain — reconstructed, NOT recovered

**The gap (Q5).** "AURC" / "AURC gain" appear in the manuscript (Sections 3.2-3.6) and in a legacy
results file (`results_MS_003.csv`, columns `aurc`, `aurc_baseline`, `aurc_gain`), but an exhaustive
grep across `code/src/` found **zero code presence** for either term, anywhere, at any point in this
project's history as sandboxed. `results_MS_003.csv` is tied to `exp_20261008_003_MS`, run under the
RETIRED protocol hash (`538d5f1f...`) whose own artifacts are separately known to be irreproducible
(Appendix item 1). The per-case predictions and whatever script generated those three columns are gone.

**This cannot be "resolved" in the sense of recovering the original formula — there is nothing left to
recover.** What follows is a new, standard, literature-grounded implementation added to close the
pipeline gap going forward, explicitly NOT represented as a reproduction of the legacy numbers.

**Implementation** (`evalkit.py`: `confidence_score`, `risk_coverage_curve`, `flat_baseline_risk`,
`aurc`, `aurc_summary`) follows the standard selective-classification "risk-coverage AURC"
(El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017):
- Rank cases by descending confidence (margin for 2-class, `1 - normalized entropy` for 3+-class —
  same auto-resolution rule as `paz_gate`).
- At each coverage level (fraction kept, most-confident-first), risk = error rate among kept cases.
- `aurc` = trapezoidal area under that risk-coverage curve (lower = better).
- `aurc_random` (the flat/random-rejection baseline) = this model's own overall error rate — rejecting
  independently of the score leaves risk constant at every coverage level.
- `aurc_gain = aurc_random - aurc` (positive = the score concentrates errors into the rejected region
  better than chance). **This is the opposite sign from the legacy `aurc_gain` column**, which computes
  as `aurc - aurc_baseline` by direct arithmetic check against `results_MS_003.csv` — that file's sign
  convention could not be independently verified or explained, so this implementation instead follows
  the convention standard in the literature (positive = improvement), and that difference must not be
  elided if these numbers are ever shown next to the legacy table.

**Partial corroboration found, not full validation:** the legacy `aurc_baseline` column, for every
`severity_roi` row (n=81 in that MS cohort), is exactly `integer / 81` (verified: 20/20 rows land on
an exact integer when multiplied by 81). That is consistent with "baseline = this model's own overall
error count / n" — exactly what `flat_baseline_risk()` computes here — and is reasonable evidence the
general shape of the original definition matches. It is NOT proof the two implementations agree on
every detail (confidence-ranking tie-breaking, interpolation method, which score was ranked), because
the per-case artifacts needed to check `aurc` itself no longer exist.

**Fresh numbers computed** under Protocol v1.1, from the real confirmatory `exp_20261009_003_MS`
posteriors, primary variant P4, primary model slda:

| task | aurc | aurc_random (baseline) | aurc_gain | method |
|---|---|---|---|---|
| severity_roi | 0.5432 | 0.4198 | **-0.1234** | margin |
| lesion_vs_normal_roi | 0.2215 | 0.3190 | 0.0976 | margin |
| abnormal_img | 0.4259 | 0.4865 | 0.0606 | margin |
| density_img | 0.2014 | 0.3288 | 0.1274 | entropy |

These are internally consistent with everything else this session has found: `severity_roi` is the
one task with a negative gain — a PAZ score cannot beat random rejection on a task where the underlying
classifier has no real decision boundary (AUC 0.502, indistinguishable from chance per the permutation
null). The three tasks with genuine signal all show positive gain.

**Status: the code gap is closed going forward. The historical manuscript AURC values are not
reproduced, are not expected to match these, and remain tied to the retired, irreproducible protocol
run exactly as already disclosed in Appendix item 1.** Whether to replace the manuscript's AURC
reporting with this new, version-tagged metric is a separate decision, not taken here.
