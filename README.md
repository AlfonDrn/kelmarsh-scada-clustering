# Kelmarsh SCADA 2022 — Multivariate Time-Series Clustering

Unsupervised clustering of wind-turbine **operating regimes** from multivariate SCADA data
(six turbines, 10-minute cadence, year 2022). The pipeline covers preprocessing benchmarks,
1-hour non-overlapping window features, four clustering families (K-Means++, FCM, Ward,
HDBSCAN), internal validity, physical profiling, and controlled ablations.

Paper: *Klasterisasi Kondisi Operasi Turbin Angin Kelmarsh Menggunakan K-Means++ pada
Representasi Temporal SCADA Multivariat*.

## Dataset

- **Kelmarsh wind farm data**, Zenodo, DOI: [`10.5281/zenodo.16807551`](https://doi.org/10.5281/zenodo.16807551) (ver. 4).
- 6 Senvion MM92 turbines (2.050 kW), 10-minute interval, UTC calendar year 2022, unlabeled.
- **Raw CSV is not included** (≈1.27 GB). The loader `kelmarsh_2022.py` expects the
  Greenbyte exports under a folder named:

  ```
  Kelmarsh_SCADA_2022_4457/
      Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv   # SCADA
      Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv         # events
      ... (six turbines)
  ```

## Repository structure

```
.
├─ README.md
├─ requirements.txt          # top-level dependencies
├─ data/README.md            # dataset placement instructions
├─ paper/                    # IEEE LaTeX source (main.tex, methodology.tex, gap_analysis.tex) + main.pdf
├─ kelmarsh_2022.py          # raw loader + preprocessing helpers
├─ kelmarsh_stage3.py        # clustering engine (Experiment)
├─ stage3_structural.py      # feature diagnosis + HDBSCAN challenger
├─ run_local_notebook.py     # execute a notebook with a real kernel
├─ prepare_local_preprocessing.py
├─ verify_stage3_results.py
├─ requirements_stage3_local.txt   # exact pinned versions
├─ notebooks/
│   ├─ 01_preprocessing.ipynb    # Tahap 2 (preprocessing benchmark, executed)
│   └─ 02_clustering.ipynb       # Tahap 3 (clustering, executed)
├─ experiments/              # controlled ablations (RQ1, RQ2, 12-mean)
│   ├─ run_rq1_ablation.py   run_rq2.py   run_means12.py
│   └─ results/                  # outputs
└─ results/
    ├─ preprocessing/        # summary CSVs (skewness, correlation, benchmarks)
    └─ clustering/           # candidate metrics, profiles, persistence, etc.
```

## Requirements

```powershell
python -m pip install -r requirements.txt
```

Key versions used: Python 3.14, numpy 2.5, pandas 3.0, scikit-learn 1.9, scikit-fuzzy 0.5,
hdbscan 0.8.44, matplotlib 3.10.

## How to run

Run from the repository root (so the loader/notebooks resolve paths).

```powershell
# 1) Preprocessing (writes artifacts/kelmarsh_2022/*)
python prepare_local_preprocessing.py
python run_local_notebook.py notebooks/01_preprocessing.ipynb

# 2) Clustering (reads artifacts/kelmarsh_2022/windows_2022.csv.gz)
python run_local_notebook.py notebooks/02_clustering.ipynb
python verify_stage3_results.py
```

> The clustering notebook reads the frozen preprocessing artifacts
> (`artifacts/kelmarsh_2022/`), not the raw CSV. Raw input is only needed to reproduce
> preprocessing.

## Controlled ablations

One-factor controls around a frozen reference (StandardScaler, 60 features), measuring
Silhouette/DBI/CHI and agreement (ARI) versus the reference partition:

```powershell
python experiments/run_rq1_ablation.py   # scale, redundancy, noise, missing-value policies
python experiments/run_rq2.py            # pointwise 10-min vs 1-hour window
python experiments/run_means12.py        # 12-mean representation sweep
```

## Key results (selection period, July–December)

- 52.241 one-hour windows × 60 features (25.881 training / 26.360 selection).
- **K-Means++ k=2**: Silhouette 0.3295, DBI 1.856, CHI 4071 — two broad regimes
  (idle vs production), persistence 0.93.
- HDBSCAN core: Silhouette 0.438 on ~67% coverage (noise ≈33%).
- **Scale disparity** is the dominant structural lever; removing redundant features
  (12-mean) improves geometry; temporal aggregation mainly stabilizes labels.

## Paper

The manuscript (IEEE conference, two columns) is in `paper/`:
`main.pdf` is the compiled result, and the sources are `main.tex` (+ `methodology.tex`,
`gap_analysis.tex`, `IEEEtran.cls`) with figures in `paper/assets/`. Compile with pdfLaTeX
(e.g., Overleaf) — no BibTeX needed (bibliography is inline).

## Notes

- The dataset has **no ground-truth labels**; clusters are descriptive operating summaries,
  not fault/shutdown labels.
- July–December was reused for model selection, so it is not an independent holdout.

## License

MIT (see `LICENSE`).
