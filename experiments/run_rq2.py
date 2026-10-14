"""RQ2: does temporal aggregation (1-hour windows) matter versus clustering
individual 10-minute SCADA observations?

Both representations use the SAME 12 signals (10 scalars + direction sin/cos),
the SAME chronological split, scaler and KMeans k=2 setting. We compare:
  * internal structure (indication only; object granularity differs),
  * physical regime profile (idle vs producing),
  * temporal coherence: mean run length in HOURS and switching rate per hour,
  * cross-representation agreement (ARI) after mapping window labels to their
    six 10-minute member observations.

Outputs go to _experiments/level2/. Original artifacts are never touched.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
MEANS = [f'{c}__mean' for c in COORDS]
CUT = pd.Timestamp('2022-07-01', tz='UTC')
ORIGIN = pd.Timestamp('2022-01-01', tz='UTC')


def audit(m):
    print(m, flush=True)


def runs_in_hours(frame, labels, native_hours):
    d = frame[['turbine', 'timestamp']].copy()
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
    d['cluster'] = labels
    d = d.sort_values(['turbine', 'timestamp']).reset_index(drop=True)
    g = d.groupby('turbine')
    changed = d.cluster.ne(d.cluster.shift()) | (d.turbine.ne(d.turbine.shift()))
    gap = (d.timestamp - g.timestamp.shift()) != pd.Timedelta(hours=native_hours)
    d['run'] = (changed | gap.fillna(True)).cumsum()
    lengths = d.groupby('run').size() * native_hours
    hours = (d.timestamp.max() - d.timestamp.min()).total_seconds() / 3600
    switches = int(changed.sum()) - d.turbine.nunique()
    return dict(mean_run_hours=round(float(lengths.mean()), 3),
                median_run_hours=round(float(lengths.median()), 3),
                switches_per_hour=round(float(switches / hours), 4),
                n_runs=int(len(lengths)))


def profile(frame, labels, power_col):
    pw = frame[power_col].to_numpy(float)
    m0, m1 = np.nanmedian(pw[labels == 0]), np.nanmedian(pw[labels == 1])
    idle = 0 if m0 < m1 else 1
    frac = np.bincount(labels, minlength=2) / len(labels)
    return idle, round(float(frac[idle]), 5), round(float(np.nanmedian(pw[labels == idle])), 2), \
        round(float(np.nanmedian(pw[labels == 1 - idle])), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []

    # ---------- pointwise: individual 10-minute observations ----------
    pre = pd.read_csv(CP / 'preprocessed_2022.csv.gz',
                      usecols=['turbine', 'timestamp', 'observed_row'] + COORDS)
    pre['timestamp'] = pd.to_datetime(pre.timestamp, utc=True)
    pre = pre[pre.observed_row].drop(columns='observed_row')
    pre = pre.sort_values(['turbine', 'timestamp']).reset_index(drop=True)
    trm = (pre.timestamp < CUT).to_numpy()
    vam = ~trm
    med = pre.loc[trm, COORDS].median()
    Xp = pre[COORDS].fillna(med).to_numpy(float)
    scp = StandardScaler().fit(Xp[trm])
    Ztr, Zva = scp.transform(Xp[trm]), scp.transform(Xp[vam])
    kmp = KMeans(n_clusters=2, random_state=SEED, **KM).fit(Ztr)
    labp = kmp.predict(Zva)
    frame_p = pre.loc[vam].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sp_p = rng.choice(len(Zva), size=3000, replace=False)
    idle_p = profile(frame_p, labp, 'power')
    rp = runs_in_hours(frame_p, labp, native_hours=1 / 6)
    row_p = dict(representation='pointwise_10min', n_objects=int(len(pre)),
                 n_train=int(trm.sum()), n_val=int(vam.sum()),
                 silhouette=round(float(silhouette_score(Zva[sp_p], labp[sp_p])), 5),
                 dbi=round(float(davies_bouldin_score(Zva, labp)), 5),
                 chi=round(float(calinski_harabasz_score(Zva, labp)), 3),
                 idle_fraction=idle_p[1], idle_power_median=idle_p[2],
                 active_power_median=idle_p[3], **rp)
    audit(row_p)
    summary.append(row_p)

    # ---------- windowed: 1-hour means ----------
    df = pd.read_csv(CP / 'windows_2022.csv.gz')
    trw = df.split.eq('train').to_numpy()
    vaw = ~trw
    medw = df.loc[trw, MEANS].median()
    Xw = df[MEANS].fillna(medw).to_numpy(float)
    scw = StandardScaler().fit(Xw[trw])
    Wtr, Wva = scw.transform(Xw[trw]), scw.transform(Xw[vaw])
    kmw = KMeans(n_clusters=2, random_state=SEED, **KM).fit(Wtr)
    labw = kmw.predict(Wva)
    frame_w = df.loc[vaw].reset_index(drop=True)
    frame_w['timestamp'] = pd.to_datetime(frame_w.window_start, utc=True)
    rngw = np.random.default_rng(SEED)
    sp_w = rngw.choice(len(Wva), size=min(3000, len(Wva)), replace=False)
    idle_w = profile(frame_w, labw, 'power__mean')
    rw = runs_in_hours(frame_w, labw, native_hours=1)
    row_w = dict(representation='windowed_1h', n_objects=int(len(df)),
                 n_train=int(trw.sum()), n_val=int(vaw.sum()),
                 silhouette=round(float(silhouette_score(Wva[sp_w], labw[sp_w])), 5),
                 dbi=round(float(davies_bouldin_score(Wva, labw)), 5),
                 chi=round(float(calinski_harabasz_score(Wva, labw)), 3),
                 idle_fraction=idle_w[1], idle_power_median=idle_w[2],
                 active_power_median=idle_w[3], **rw)
    audit(row_w)
    summary.append(row_w)

    # ---------- cross-representation agreement on 10-minute rows ----------
    # map each window label to its 6 member 10-min rows
    wp = frame_w[['turbine', 'timestamp']].copy()
    wp['bin'] = ((wp.timestamp - ORIGIN).dt.total_seconds() // 3600).astype(int)
    wp['win_label'] = labw
    pre_va = frame_p[['turbine', 'timestamp']].copy()
    pre_va['bin'] = ((pre_va.timestamp - ORIGIN).dt.total_seconds() // 3600).astype(int)
    pre_va['point_label'] = labp
    merged = pre_va.merge(wp[['turbine', 'bin', 'win_label']], on=['turbine', 'bin'],
                          how='inner')
    both = merged.dropna(subset=['point_label', 'win_label'])
    ari_cross = float(adjusted_rand_score(both.point_label, both.win_label))
    agree = float((both.point_label == both.win_label).mean())
    cross = dict(matched_10min_rows=int(len(both)), ari_pointwise_vs_window=round(ari_cross, 5),
                 raw_agreement=round(agree, 5))
    audit(cross)

    out = pd.DataFrame(summary)
    out.to_csv(OUT / 'rq2_pointwise_vs_window.csv', index=False)
    (OUT / 'rq2_cross_agreement.json').write_text(json.dumps(cross, indent=2), encoding='utf-8')
    audit('\n' + out.to_string(index=False))
    audit(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
