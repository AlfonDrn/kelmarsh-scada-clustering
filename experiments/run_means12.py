"""Full evaluation of the '12 means only' representation (Level 2 winner).

Same 52,241 windows; each window described by the 12 coordinate MEANS only
(10 scalar signals + direction sin/cos), instead of all 60 statistics.
StandardScaler fit on Jan-Jun; KMeans sweep k=2..10 on the Jul-Dec selection
period; plus seed stability and temporal persistence for k=2.

Outputs go to _experiments/level2/. Original artifacts are never touched.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / 'artifacts' / 'kelmarsh_2022'
OUT = Path(__file__).resolve().parent
SEEDS = [2022, 2023, 2024, 2025, 2026]
KM = dict(init='k-means++', n_init=20, max_iter=500, tol=1e-4, algorithm='lloyd')
SIGNALS = ['wind', 'power', 'reactive', 'ambient', 'bearing', 'oil',
           'rotor', 'generator', 'pitch', 'frequency']
COORDS = SIGNALS + ['direction_sin', 'direction_cos']
MEANS = [f'{c}__mean' for c in COORDS]
assert len(MEANS) == 12


def audit(msg):
    print(msg, flush=True)


def persistence(frame, labels):
    d = frame[['turbine', 'split', 'window_start']].copy()
    d['window_start'] = pd.to_datetime(d.window_start, utc=True)
    d['cluster'] = labels
    d = d.sort_values(['turbine', 'window_start'])
    g = d.groupby('turbine')
    d['prev_cluster'] = g.cluster.shift()
    d['prev_time'] = g.window_start.shift()
    valid = d.prev_cluster.notna() & (
        (d.window_start - d.prev_time) == pd.Timedelta(hours=1))
    d = d[valid]
    return float((d.cluster == d.prev_cluster).mean()), int(len(d))


def main():
    df = pd.read_csv(CP / 'windows_2022.csv.gz')
    sil = pd.read_csv(CP / 'clustering' / 'silhouette_sample.csv')
    tr = df.split.eq('train').to_numpy()
    va = df.split.eq('validation').to_numpy()

    med = df.loc[tr, MEANS].median()
    X = df[MEANS].fillna(med).to_numpy(float)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])

    va_idx = np.where(va)[0]
    pos = {r: i for i, r in enumerate(va_idx)}
    sp = np.array([pos[r] for r in sil.window_row.to_numpy() if r in pos])
    audit(f'rows {len(df)} | train {tr.sum()} | val {va.sum()} | silhouette n {len(sp)}')

    frame_va = df.loc[va].reset_index(drop=True)

    # ---- k sweep (seed 2022) ----
    rows = []
    labels_by_k = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=2022, **KM).fit(Xtr)
        lab = km.predict(Xva)
        labels_by_k[k] = lab
        frac = np.bincount(lab, minlength=k) / len(lab)
        persist, npairs = persistence(frame_va, lab)
        rows.append(dict(
            k=k,
            silhouette=round(float(silhouette_score(Xva[sp], lab[sp])), 5),
            dbi=round(float(davies_bouldin_score(Xva, lab)), 5),
            chi=round(float(calinski_harabasz_score(Xva, lab)), 3),
            min_cluster_fraction=round(float(frac.min()), 5),
            persistence=round(persist, 5),
            turbine_nmi=round(float(normalized_mutual_info_score(
                frame_va.turbine, lab)), 6),
        ))
        audit(rows[-1])
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / 'means12_sweep.csv', index=False)

    # ---- seed stability at k=2 (and chosen best k) ----
    stability = []
    for k in [2, 3]:
        labs = []
        for s in SEEDS:
            km = KMeans(n_clusters=k, random_state=s, **KM).fit(Xtr)
            labs.append(km.predict(Xva))
        aris = [adjusted_rand_score(labs[i], labs[j])
                for i in range(len(labs)) for j in range(i + 1, len(labs))]
        stability.append(dict(k=k, seed_ari_mean=round(float(np.mean(aris)), 5),
                              seed_ari_min=round(float(np.min(aris)), 5),
                              n_seeds=len(SEEDS)))
        audit(stability[-1])
    pd.DataFrame(stability).to_csv(OUT / 'means12_seed_stability.csv', index=False)

    # ---- k=2 profiles (physical interpretation check) ----
    lab2 = labels_by_k[2]
    prof = frame_va.groupby(lab2)[MEANS].median().round(3)
    prof['n'] = np.bincount(lab2, minlength=2)
    prof.to_csv(OUT / 'means12_k2_profiles.csv')
    audit('\nk=2 median profiles (original units, validation):\n' + prof.to_string())

    audit('\nsweep:\n' + sweep.to_string(index=False))
    (OUT / 'means12_summary.json').write_text(json.dumps(
        dict(sweep=rows, seed_stability=stability), indent=2), encoding='utf-8')
    audit(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
