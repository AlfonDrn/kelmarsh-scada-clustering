"""Preprocessing ablation: how do preprocessing choices affect clustering structure?

One-factor-at-a-time around a frozen reference (StandardScaler, 60 features,
no clipping, ambient-only missing policy). For every variant we report the
internal validity metrics, the agreement (ARI) with the reference partition,
cluster sizes, seed stability, temporal persistence and the physical profile,
so the effect can be measured by magnitude-of-structure-change rather than by score.

Windows-based factors (scale, redundancy, noise) reuse the artifact window table.
The missing-value factor rebuilds 1-hour windows directly from the cleaned
10-minute series with different fill policies.

All outputs go to _experiments/level2/. Original artifacts are never touched.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / 'artifacts' / 'kelmarsh_2022'
OUT = Path(__file__).resolve().parent
SEED = 2022
SEEDS = [2022, 2023, 2024, 2025, 2026]
KM = dict(init='k-means++', n_init=20, max_iter=500, tol=1e-4, algorithm='lloyd')
SIGNALS = ['wind', 'power', 'reactive', 'ambient', 'bearing', 'oil',
           'rotor', 'generator', 'pitch', 'frequency']
COORDS = SIGNALS + ['direction_sin', 'direction_cos']
STATS = ['mean', 'std', 'min', 'max', 'endpoint_change']
FEATS = [f'{c}__{s}' for c in COORDS for s in STATS]
MEANS = [f'{c}__mean' for c in COORDS]


def audit(m):
    print(m, flush=True)


def persistence(frame, labels):
    d = frame[['turbine', 'window_start']].copy()
    d['window_start'] = pd.to_datetime(d.window_start, utc=True)
    d['cluster'] = labels
    d = d.sort_values(['turbine', 'window_start'])
    g = d.groupby('turbine')
    prev = g.cluster.shift()
    prev_t = g.window_start.shift()
    valid = prev.notna() & ((d.window_start - prev_t) == pd.Timedelta(hours=1))
    d, prev = d[valid], prev[valid]
    return round(float((d.cluster.to_numpy() == prev.to_numpy()).mean()), 5)


def evaluate(name, factor, level, Xtr, Xva, frame_va, ref_lab, sil_pos, results,
             ref_ari=True):
    km = KMeans(n_clusters=2, random_state=SEED, **KM).fit(Xtr)
    lab = km.predict(Xva)
    sizes = np.bincount(lab, minlength=2) / len(lab)
    pw = frame_va['power__mean'].to_numpy(float)
    m0, m1 = np.median(pw[lab == 0]), np.median(pw[lab == 1])
    idle = 0 if m0 < m1 else 1
    seed_labs = [KMeans(n_clusters=2, random_state=s, **KM).fit(Xtr).predict(Xva)
                 for s in SEEDS]
    aris = [adjusted_rand_score(seed_labs[i], seed_labs[j])
            for i in range(len(seed_labs)) for j in range(i + 1, len(seed_labs))]
    rec = dict(
        factor=factor, level=level,
        silhouette=round(float(silhouette_score(Xva[sil_pos], lab[sil_pos])), 5),
        dbi=round(float(davies_bouldin_score(Xva, lab)), 5),
        chi=round(float(calinski_harabasz_score(Xva, lab)), 3),
        ari_vs_ref=(round(float(adjusted_rand_score(ref_lab, lab)), 5)
                    if ref_ari and ref_lab is not None else None),
        min_cluster_fraction=round(float(sizes.min()), 5),
        idle_fraction=round(float(sizes[idle]), 5),
        idle_power_median=round(float(np.median(pw[lab == idle])), 2),
        active_power_median=round(float(np.median(pw[lab == 1 - idle])), 2),
        seed_ari_mean=round(float(np.mean(aris)), 5),
        seed_ari_min=round(float(np.min(aris)), 5),
        persistence=persistence(frame_va, lab),
    )
    audit(rec)
    results.append(rec)
    return lab


def scale_fit(kind, Xtr, Xva, turb_tr, turb_va):
    if kind == 'none':
        return Xtr, Xva
    if kind == 'standard':
        s = StandardScaler().fit(Xtr)
        return s.transform(Xtr), s.transform(Xva)
    if kind == 'robust':
        s = RobustScaler().fit(Xtr)
        return s.transform(Xtr), s.transform(Xva)
    if kind == 'minmax':
        s = MinMaxScaler().fit(Xtr)
        return s.transform(Xtr), s.transform(Xva)
    if kind == 'per_turbine_standard':
        At, Av = np.empty_like(Xtr), np.empty_like(Xva)
        for t in np.unique(turb_tr):
            trm = turb_tr == t
            vam = turb_va == t
            s = StandardScaler().fit(Xtr[trm])
            At[trm] = s.transform(Xtr[trm])
            Av[vam] = s.transform(Xva[vam])
        return At, Av
    raise ValueError(kind)


def clip_fit(lo_q, hi_q, Xtr, Xva):
    lo = np.percentile(Xtr, lo_q, axis=0)
    hi = np.percentile(Xtr, hi_q, axis=0)
    return np.clip(Xtr, lo, hi), np.clip(Xva, lo, hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CP / 'windows_2022.csv.gz')
    sil = pd.read_csv(CP / 'clustering' / 'silhouette_sample.csv')
    turb = df.turbine.to_numpy()
    tr = df.split.eq('train').to_numpy()
    va = df.split.eq('validation').to_numpy()
    med = df.loc[tr, FEATS].median()
    Xraw = df[FEATS].fillna(med).to_numpy(float)
    frame_va = df.loc[va].reset_index(drop=True)
    va_idx = np.where(va)[0]
    pos = {r: i for i, r in enumerate(va_idx)}
    sp = np.array([pos[r] for r in sil.window_row.to_numpy() if r in pos])
    results = []

    # ---- reference ----
    audit('== reference: standard | full 60 | no clip ==')
    Xtr, Xva = Xraw[tr], Xraw[va]
    Rtr, Rva = scale_fit('standard', Xtr, Xva, turb[tr], turb[va])
    ref = evaluate('reference', 'reference', 'standard_full60', Rtr, Rva,
                   frame_va, None, sp, results, ref_ari=False)

    # ---- factor 1: scale disparity ----
    audit('== factor: scale disparity ==')
    for kind in ['none', 'robust', 'minmax', 'per_turbine_standard']:
        A, B = scale_fit(kind, Xtr, Xva, turb[tr], turb[va])
        evaluate('scale', kind, kind, A, B, frame_va, ref, sp, results)

    # ---- factor 2: redundant features ----
    audit('== factor: redundant features ==')
    subsets = {
        'means_only_12': [i for i, f in enumerate(FEATS) if f in MEANS],
        'drop_generator': [i for i, f in enumerate(FEATS)
                           if not f.startswith('generator__')],
        'level_plus_dynamics_24': [i for i, f in enumerate(FEATS)
                                   if f.endswith('__mean') or f.endswith('__endpoint_change')],
        'drop_frequency': [i for i, f in enumerate(FEATS)
                           if not f.startswith('frequency__')],
    }
    for nm, idx in subsets.items():
        s = StandardScaler().fit(Xtr[:, idx])
        evaluate('redundancy', nm, nm, s.transform(Xtr[:, idx]),
                 s.transform(Xva[:, idx]), frame_va, ref, sp, results)

    # ---- factor 3: noisy / extreme values ----
    audit('== factor: noisy / extreme values ==')
    for q in [(1, 99), (5, 95)]:
        C, D = clip_fit(q[0], q[1], Xtr, Xva)
        s = StandardScaler().fit(C)
        evaluate('noise', f'winsor_{q[0]}_{q[1]}', f'winsor_{q[0]}_{q[1]}',
                 s.transform(C), s.transform(D), frame_va, ref, sp, results)

    pd.DataFrame(results).to_csv(OUT / 'ablation_preprocessing.csv', index=False)
    audit('\n' + pd.DataFrame(results).to_string(index=False))

    # ---- factor 4: missing-value policy (rebuild 1h windows from 10-min) ----
    audit('\n== factor: missing-value policy (rebuilt from 10-min series) ==')
    miss = pd.read_csv(CP / 'preprocessed_2022.csv.gz',
                       usecols=['turbine', 'timestamp'] + COORDS +
                       [f'missing_{s}' for s in SIGNALS] + ['missing_direction'])
    miss['timestamp'] = pd.to_datetime(miss.timestamp, utc=True)
    miss = miss.sort_values(['turbine', 'timestamp']).reset_index(drop=True)
    origin = pd.Timestamp('2022-01-01', tz='UTC')
    miss['bin'] = ((miss.timestamp - origin).dt.total_seconds() // 3600).astype(int)
    tr_time = (miss.timestamp < pd.Timestamp('2022-07-01', tz='UTC')).to_numpy()
    mmed = {c: np.nanmedian(miss.loc[tr_time, c]) for c in COORDS}
    missing_any = miss[[f'missing_{s}' for s in SIGNALS] + ['missing_direction']].any(axis=1).to_numpy()

    policies = {}
    policies['ambient_only_rebuilt'] = miss[COORDS].copy()
    policies['median_fill'] = miss[COORDS].fillna(pd.Series(mmed))
    ff = miss.groupby('turbine')[COORDS].ffill()
    policies['forward_fill'] = ff
    policies['listwise_drop'] = miss.loc[~missing_any, COORDS].copy()

    def run_policy(name, coord_df):
        d = pd.concat([miss[['turbine', 'bin', 'timestamp']], coord_df], axis=1).dropna(subset=COORDS)
        g = d.groupby(['turbine', 'bin'])
        keep = g.size()
        keep = keep[keep == 6].index
        d = d.set_index(['turbine', 'bin'])
        gg = d.groupby(level=[0, 1])
        out = pd.DataFrame(index=keep)
        for c in COORDS:
            out[f'{c}__mean'] = gg[c].mean()
            out[f'{c}__std'] = gg[c].std()
            out[f'{c}__min'] = gg[c].min()
            out[f'{c}__max'] = gg[c].max()
            out[f'{c}__endpoint_change'] = gg[c].last() - gg[c].first()
        out = out.reset_index()
        out['window_start'] = origin + pd.to_timedelta(out['bin'] * 3600, unit='s')
        return out

    mres = []
    for nm, cd in policies.items():
        w = run_policy(nm, cd)
        wt = (w.window_start < pd.Timestamp('2022-07-01', tz='UTC')).to_numpy()
        wv = ~wt
        if wt.sum() < 100 or wv.sum() < 100:
            audit(f'skip {nm} (too few windows)')
            continue
        s = StandardScaler().fit(w.loc[wt, FEATS].to_numpy(float))
        evaluate(f'missing_{nm}', 'missing', nm,
                 s.transform(w.loc[wt, FEATS].to_numpy(float)),
                 s.transform(w.loc[wv, FEATS].to_numpy(float)),
                 w.loc[wv].reset_index(drop=True), None,
                 np.arange(int(wv.sum())), mres, ref_ari=False)


    pd.DataFrame(mres).to_csv(OUT / 'ablation_preprocessing_missing.csv', index=False)
    audit('\n' + pd.DataFrame(mres).to_string(index=False))
    (OUT / 'ablation_preprocessing.json').write_text(json.dumps(
        dict(windows=results, missing=mres), indent=2), encoding='utf-8')
    audit(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
