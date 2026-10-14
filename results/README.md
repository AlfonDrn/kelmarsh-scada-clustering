# Results

Curated summary tables produced by the pipeline (full artifacts are larger and are not
stored here). All numbers use the frozen preprocessing configuration.

## `preprocessing/`

| File | Shows |
|---|---|
| `selected_scale_skewness.csv` | Descriptive stats (mean, std, min/median/max, skewness, % missing) per signal |
| `selected_units.csv` | Signal → 2022 SCADA column header and unit mapping |
| `selected_missing_pct.csv` | Conditional sensor missingness per turbine |
| `raw_valid_coverage.csv` | Scheduled slots vs available (`Data Availability=1`) records per turbine |
| `coverage_gap_runs.csv` | Longest missing-gap runs |
| `sensor_pearson_correlation.csv` / `sensor_spearman_correlation.csv` | Inter-sensor correlation matrices |
| `calendar_lag_autocorrelation.csv` | Temporal autocorrelation at several lags |
| `benchmark_missing.csv` | Missing-value imputation benchmark (gate: nRMSE, coverage) |
| `benchmark_outliers.csv` | Outlier handling benchmark (IQR / z-score / winsor / DBSCAN vs flag-only) |
| `benchmark_scaling.csv` | Scaler comparison (max-distance share) |
| `benchmark_scaler_single_extreme_stress.csv` | Scaler robustness to one synthetic extreme |
| `benchmark_windows.csv` | Window duration × completeness retention |
| `manual_preprocessing_decision.csv` | Chosen preprocessing decision per step |
| `preprocessing_selection.json` | Frozen preprocessing configuration |
| `full_field_quality_audit.csv` | Per-turbine × field quality audit |

## `clustering/`

| File | Shows |
|---|---|
| `model_comparison.csv` | All candidates (K-Means/FCM) with Silhouette, DBI, CHI, sizes, stability, persistence |
| `kmeans_sweep.csv` | K-Means $k=2..10$ sweep (incl. inertia, seed ARI, persistence) |
| `fcm_comparison.csv` | Fuzzy C-Means challenger |
| `ward_sensitivity.csv` | Ward vs K-Means on the same training sample |
| `hdbscan_comparison.csv` | Six HDBSCAN configurations (coverage, noise, conditional metrics) |
| `hdbscan_matched_coverage_kmeans.csv` | K-Means on the identical assigned/non-noise population |
| `cluster_profiles_median.csv` / `q25` / `q75` | Physical profiles of the final clusters |
| `cluster_sizes.csv` | Final cluster sizes |
| `temporal_persistence_by_turbine.csv` | Label persistence per turbine |
| `cluster_distribution_by_turbine.csv` / `by_month.csv` | Cluster distribution across turbines / months |
| `quality_burden_by_cluster.csv` | Quality flags per cluster |
| `candidate_quality_association.csv` | NMI of labels vs quality flags |
| `high_correlation_pairs.csv` | Feature pairs with \|r\| ≥ 0.95 |
| `feature_space_variance.csv` | Per-feature variance (train vs selection) |
| `redundancy_sensitivity.csv` | Rotor/generator re-weighting sensitivity |
| `final_model_configuration.json` | Final model configuration and label mapping |
| `final_verification.json` | Reproducibility / consistency checks |
| `input_manifest.json` / `experiment_configuration.json` / `feature_columns.json` | Input checksums, configuration, feature order |

Ablation outputs (preprocessing scale/redundancy/noise/missing; temporal pointwise-vs-window;
12-mean) are in `../experiments/results/`.
