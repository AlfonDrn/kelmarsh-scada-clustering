"""Level 2 experiments: re-transform the frozen window features.

Reads existing preprocessing artifacts (windows_2022.csv.gz, preprocessed_2022.csv.gz)
and writes ALL outputs to _experiments/level2/. It never touches or overwrites the
original artifacts, notebooks or modules.

Baseline = reproduce the published KMeans_k2 (StandardScaler on the 60 coordinate
features, fit on Jan-Jun, evaluated on Jul-Dec) to validate the harness, then compare
alternative representations by silhouette / DBI / CHI and ARI against the baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   PowerTransformer, QuantileTransformer)
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / 'artifacts' / 'kelmarsh_2022'
OUT = Path(__file__).resolve().parent
SEED = 2022
KM = dict(init='k-means++', n_init=20, max_iter=500, tol=1e-4, algorithm='lloyd')
SIGNALS = ['wind', 'power', 'reactive', 'ambient', 'bearing', 'oil',
           'rotor', 'generator', 'pitch', 'frequency']
COORDS = SIGNALS + ['direction_sin', 'direction_cos']
STATS = ['mean', 'std', 'min', 'max', 'endpoint_change']
FEATS = [f'{c}__{s}' for c in COORDS for s in STATS]
assert len(FEATS) == 60


def audit(msg):
    print(msg, flush=True)


def load_windows():
    df = pd.read_csv(CP / 'windows_2022.csv.gz')
    sil = pd.read_csv(CP / 'clustering' / 'silhouette_sample.csv')
    return df, sil


def one_hour_matrix(df):
    tr = df.split.eq('train').to_numpy()
    va = df.split.eq('validation').to_numpy()
    med = df.loc[tr, FEATS].median()
    X = df[FEATS].fillna(med).to_numpy(float)
    return X, tr, va


def sil_positions(va_idx, sil):
    pos = {r: i for i, r in enumerate(va_idx)}
    return np.array([pos[r] for r in sil.window_row.to_numpy() if r in pos])


def score(name, Xtr, Xva, y_base, sil_pos, results, y_tr=None):
    km = KMeans(n_clusters=2, random_state=SEED, **KM).fit(Xtr)
    lab = km.predict(Xva)
    rec = dict(
        variant=name,
        silhouette=round(float(silhouette_score(Xva[sil_pos], lab[sil_pos])), 5),
        dbi=round(float(davies_bouldin_score(Xva, lab)), 5),
        chi=round(float(calinski_harabasz_score(Xva, lab)), 3),
        ari_vs_baseline=1.0 if y_base is None else round(float(adjusted_rand_score(y_base, lab)), 5),
        min_cluster_fraction=round(float(np.bincount(lab, minlength=2).min() / len(lab)), 5),
    )
    audit(rec)
    results.append(rec)
    return lab


def six_hour_matrix():
    """Rebuild 6-hour non-overlapping windows from the cleaned 10-minute series."""
    keep = ['turbine', 'timestamp', 'split'] + COORDS
    pre = pd.read_csv(CP / 'preprocessed_2022.csv.gz', usecols=keep)
    pre['timestamp'] = pd.to_datetime(pre.timestamp, utc=True)
    origin = pd.Timestamp('2022-01-01', tz='UTC')
    secs = (pre.timestamp - origin).dt.total_seconds()
    pre['bin'] = (secs // (6 * 3600)).astype(int)
    rows = []
    for (turbine, b), g in pre.groupby(['turbine', 'bin'], sort=True):
        if len(g) != 36:
            continue
        rec = dict(turbine=turbine, window_start=origin + pd.to_timedelta(b * 6, unit='h'))
        for c in COORDS:
            v = g[c].to_numpy(float)
            rec[f'{c}__mean'] = v.mean()
            rec[f'{c}__std'] = v.std(ddof=1)
            rec[f'{c}__min'] = v.min()
            rec[f'{c}__max'] = v.max()
            rec[f'{c}__endpoint_change'] = v[-1] - v[0]
        rows.append(rec)
    w = pd.DataFrame(rows)
    cut = pd.Timestamp('2022-07-01', tz='UTC')
    tr = (w.window_start < cut).to_numpy()
    va = ~tr
    med = w.loc[tr, FEATS].median()
    w[FEATS] = w[FEATS].fillna(med)
    ok = w[FEATS].notna().all(axis=1).to_numpy()
    tr = tr & ok
    va = va & ok
    X = w[FEATS].to_numpy(float)
    audit(f'6h windows: {len(w)} total, usable {ok.sum()}, train {tr.sum()}, validation {va.sum()}')
    return w, X, tr, va


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    audit('== 1h frozen window features ==')
    df, sil = load_windows()
    Xraw, tr, va = one_hour_matrix(df)
    va_idx = np.where(va)[0]
    sp = sil_positions(va_idx, sil)
    Xtr, Xva = Xraw[tr], Xraw[va]
    audit(f'rows {len(df)} | train {tr.sum()} | validation {va.sum()} | silhouette n {len(sp)}')

    sc = StandardScaler().fit(Xtr)
    base = score('baseline_standard', sc.transform(Xtr), sc.transform(Xva), None, sp, results)

    pt = PowerTransformer(method='yeo-johnson').fit(Xtr)
    score('yeo_johnson', pt.transform(Xtr), pt.transform(Xva), base, sp, results)

    rb = RobustScaler().fit(Xtr)
    score('robust', rb.transform(Xtr), rb.transform(Xva), base, sp, results)

    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                             random_state=SEED).fit(Xtr)
    score('quantile_normal', qt.transform(Xtr), qt.transform(Xva), base, sp, results)

    score('yj_then_robust', rb.transform(pt.transform(Xtr)),
          rb.transform(pt.transform(Xva)), base, sp, results)

    # V6: drop the extreme-tail frequency family (grid frequency spikes)
    keep = [i for i, f in enumerate(FEATS) if not f.startswith('frequency__')]
    scd = StandardScaler().fit(Xtr[:, keep])
    score('drop_frequency', scd.transform(Xtr[:, keep]),
          scd.transform(Xva[:, keep]), base, sp, results)

    # V7: winsorize each feature at train 1st/99th percentile, then standardize
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    scw = StandardScaler().fit(np.clip(Xtr, lo, hi))
    score('winsor_1_99', scw.transform(np.clip(Xtr, lo, hi)),
          scw.transform(np.clip(Xva, lo, hi)), base, sp, results)

    # V8: winsorize + drop frequency (combine both fixes)
    scdw = StandardScaler().fit(np.clip(Xtr[:, keep], lo[keep], hi[keep]))
    score('winsor_drop_frequency',
          scdw.transform(np.clip(Xtr[:, keep], lo[keep], hi[keep])),
          scdw.transform(np.clip(Xva[:, keep], lo[keep], hi[keep])), base, sp, results)

    # V9/V10: drop redundant std/min/max (level + dynamics, then level only)
    keep2 = [i for i, f in enumerate(FEATS)
             if f.endswith('__mean') or f.endswith('__endpoint_change')]
    sc2 = StandardScaler().fit(Xtr[:, keep2])
    score(f'level_plus_dynamics_{len(keep2)}', sc2.transform(Xtr[:, keep2]),
          sc2.transform(Xva[:, keep2]), base, sp, results)

    keep3 = [i for i, f in enumerate(FEATS) if f.endswith('__mean')]
    sc3 = StandardScaler().fit(Xtr[:, keep3])
    score(f'means_only_{len(keep3)}', sc3.transform(Xtr[:, keep3]),
          sc3.transform(Xva[:, keep3]), base, sp, results)

    audit('== 6h windows rebuilt from cleaned 10-min series ==')
    results6 = []
    w6, X6, tr6, va6 = six_hour_matrix()
    sc6 = StandardScaler().fit(X6[tr6])
    score('six_hour_standard', sc6.transform(X6[tr6]), sc6.transform(X6[va6]),
          None, np.arange(va6.sum()), results6)
    lo6, hi6 = np.percentile(X6[tr6], 1, axis=0), np.percentile(X6[tr6], 99, axis=0)
    sc6w = StandardScaler().fit(np.clip(X6[tr6], lo6, hi6))
    score('six_hour_winsor', sc6w.transform(np.clip(X6[tr6], lo6, hi6)),
          sc6w.transform(np.clip(X6[va6], lo6, hi6)), None, np.arange(va6.sum()), results6)

    out = pd.DataFrame(results)
    out.to_csv(OUT / 'level2_results.csv', index=False)
    (OUT / 'level2_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame(results6).to_csv(OUT / 'level2_results_6h.csv', index=False)
    audit('\n1h frozen features:\n' + out.to_string(index=False))
    audit('\n6h rebuilt windows:\n' + pd.DataFrame(results6).to_string(index=False))
    audit(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
