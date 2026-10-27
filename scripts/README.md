# `scripts/`

The pipeline, in order. Run from the project root.

```bash
python scripts/01_prepare_data.py     # raw CSV  -> data/processed/clean.csv + profile
python scripts/02_train_models.py     # clean    -> models/*.pkl + metrics + figures
python scripts/03_run_evaluation.py   # clean+models -> policy comparison + buffer analysis
```

Each script is a thin wrapper around the corresponding module in `src/qms/`, so the
logic is importable and testable. They share `_bootstrap.py`, which puts `src/` on
the path so the scripts work from a plain checkout **without installing** the package
(`pip install -e .` also works and makes the bootstrap a no-op).

**Outputs**

| Script | Produces |
|---|---|
| `01_prepare_data.py` | `data/processed/clean.csv`, `data/processed/data_profile.json` |
| `02_train_models.py` | `models/{no_show,arrival_delay,waiting_time}.pkl`, `reports/metrics/model_metrics.json`, `reports/figures/*` |
| `03_run_evaluation.py` | `reports/metrics/policy_comparison.{json,csv}`, `reports/metrics/buffer_analysis.json`, `reports/figures/{policy_comparison,composite_score,buffer_analysis}.png` |

Re-running is idempotent — outputs are overwritten.
