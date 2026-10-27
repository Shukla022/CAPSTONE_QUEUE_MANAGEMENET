# `reports/`

Generated outputs from training and evaluation (git-ignored — regenerate with the
pipeline scripts).

```
reports/
├── metrics/
│   ├── model_metrics.json        # test metrics + feature importances for all 3 models
│   ├── policy_comparison.json     # FCFS vs Static-Priority vs Multi-Objective
│   ├── policy_comparison.csv       # same, tabular
│   └── buffer_analysis.json        # no-show-aware overbooking summary
└── figures/
    ├── no_show_feature_importance.png / no_show_calibration.png
    ├── arrival_delay_feature_importance.png / arrival_delay_pred_vs_actual.png
    ├── waiting_time_feature_importance.png / waiting_time_pred_vs_actual.png
    ├── policy_comparison.png       # the four objectives, per policy
    ├── composite_score.png         # single normalised score, per policy
    └── buffer_analysis.png         # expected no-shows & recommended overbooking
```

Produced by:

```bash
python scripts/02_train_models.py     # model_metrics.json + model figures
python scripts/03_run_evaluation.py   # policy_comparison.* + buffer_analysis.json + figures
```

The Admin Dashboard and Model Insights pages read these files directly, so refresh
them by re-running the scripts.
