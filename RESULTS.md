# Every run

Regenerate any row with `scripts/analyze_truthlens_auditor_eval_v2.py`.

**Read the `model?` column first.** Runs with `dry_run: true` in their
`run_manifest.json` never called the model — they exercise the harness. They are
not results.

| run | model? | examples | located | exact | precision | trap FP |
|---|---|---:|---:|---:|---:|---:|
| `truthlens_auditor_eval_v2_1_unit_novita` | **yes** | 165 | 104/110 (94.5%) | 93/110 (84.5%) | 100.0% | 0/55 |
| `truthlens_auditor_eval_v2_unit` | **yes** | 165 | 106/110 (96.4%) | 75/110 (68.2%) | 100.0% | 0/55 |
| `truthlens_auditor_eval_v2_1_packed_novita_2` | **yes** | 2 | 19/22 (86.4%) | 16/22 (72.7%) | 61.5% | 2/8 |
| `truthlens_auditor_eval_v2_1_packed_strict_novita_2` | **yes** | 2 | 19/22 (86.4%) | 17/22 (77.3%) | 100.0% | 0/8 |
| `truthlens_auditor_eval_v2_1_packed_dry` | no — dry | 1 | 0/11 (0.0%) | 0/11 (0.0%) | — | 0/4 |
| `truthlens_auditor_eval_v2_1_unit_dry` | no — dry | 1 | 0/1 (0.0%) | 0/1 (0.0%) | — | 0/0 |

**located** = pointed at the correct turn. **exact** = correct turn *and*
correctly named mechanism. The gap between them is the taxonomy finding in the
[README](README.md#finding-most-of-the-error-was-taxonomy-not-detection).

The two 165-example rows are the paired comparison: identical examples, identical
model, identical pipeline, `dry_run: false` on both. Only the label set differs.

Raw per-chunk LLM dumps live under each run's `raw/` and are gitignored — bulky
and regenerable.
