"""Stage 3 clustering on frozen 60-coordinate windows. No preprocessing refit."""
from __future__ import annotations
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import pickle
import platform
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import skfuzzy

SEED = 2022
SEEDS = [2022, 2023, 2024, 2025, 2026]
KM = dict(init='k-means++', n_init=20, max_iter=500, tol=1e-4, algorithm='lloyd')
FCM = dict(m=2.0, error=1e-5, maxiter=500)
UNITS = {'wind':'m/s','power':'kW','reactive':'kvar','rotor':'RPM','generator':'RPM',
         'pitch':'degrees','ambient':'°C','bearing':'°C','oil':'°C','frequency':'Hz'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_out(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   default=lambda v: v.item() if hasattr(v,'item') else str(v)), encoding='utf-8')


def membership(X, centers, m=2.0):
    # Same Euclidean FCM membership update as cmeans with fixed trained centers.
    distance = np.fmax(cdist(X, centers), np.finfo(float).eps)
    weights = distance ** (-2/(m-1))
    return weights / weights.sum(axis=1, keepdims=True)


def fit_fcm(X, k, seed):
    """Instrument library updates to retain the actual stopping norm, without altering updates."""
    backend=importlib.import_module('skfuzzy.cluster._cmeans')
    original=backend._cmeans0
    last={}
    def traced(data,u_old,*args,**kwargs):
        result=original(data,u_old,*args,**kwargs)
        last['delta']=float(np.linalg.norm(result[1]-u_old))
        return result
    backend._cmeans0=traced
    try:
        result=skfuzzy.cluster.cmeans(X.T,k,seed=seed,**FCM)
    finally:
        backend._cmeans0=original
    return result,last['delta']


def validity(X, labels, sample, k):
    unique = np.unique(labels)
    counts = np.bincount(labels, minlength=k)
    valid = 1 < len(unique) < len(X)
    sub_labels = labels[sample]
    sil_ok = 1 < len(np.unique(sub_labels)) < len(sample)
    return dict(silhouette=float(silhouette_score(X[sample],sub_labels)) if sil_ok else np.nan,
        dbi=float(davies_bouldin_score(X,labels)) if valid else np.nan,
        chi=float(calinski_harabasz_score(X,labels)) if valid else np.nan,
        occupied_clusters=len(unique), min_cluster_n=int(counts.min()),
        min_cluster_fraction=float(counts.min()/len(X)),
        imbalance_ratio=float(counts.max()/counts.min()) if counts.min() else np.inf,
        very_small_cluster_warning=bool(counts.min()/len(X)<0.01))


def temporal_summary(frame, labels):
    data = frame[['turbine','split','window_start']].copy()
    data['cluster'] = labels
    data = data.sort_values(['turbine','split','window_start'])
    group = data.groupby(['turbine','split'], sort=True)
    data['previous_time'] = group.window_start.shift()
    data['previous_cluster'] = group.cluster.shift()
    consecutive = (data.window_start-data.previous_time).eq(pd.Timedelta(hours=1))
    pairs = data[consecutive].copy()
    pairs['previous_cluster'] = pairs.previous_cluster.astype(int)
    pairs['changed'] = pairs.cluster.ne(pairs.previous_cluster)
    rows = []
    for (t, split), part in pairs.groupby(['turbine','split']):
        rows.append({'turbine':t,'split':split,'consecutive_pairs':len(part),
                     'transition_rate':float(part.changed.mean()),'persistence':float(1-part.changed.mean())})
    # Runs reset at gaps and split/turbine boundaries, including one-window runs.
    same = consecutive & data.cluster.eq(data.previous_cluster)
    data['run_id'] = (~same).cumsum()
    runs = data.groupby('run_id').agg(turbine=('turbine','first'),split=('split','first'),
        cluster=('cluster','first'),start=('window_start','first'),length_hours=('cluster','size'))
    transitions = pairs.groupby(['split','turbine','previous_cluster','cluster']).size().rename('count').reset_index()
    return pairs, pd.DataFrame(rows), runs, transitions


def synthetic_checks():
    """Tests never enter research result files."""
    from sklearn.datasets import make_blobs
    X, truth = make_blobs(n_samples=180, centers=[[-5,-5],[5,5]], cluster_std=.25, random_state=SEED)
    km = KMeans(n_clusters=2,random_state=SEED,**KM).fit(X)
    repeat = KMeans(n_clusters=2,random_state=SEED,**KM).fit(X)
    assert adjusted_rand_score(truth, km.labels_) == 1
    assert np.array_equal(km.labels_,repeat.labels_)
    centers, u, _, _, _, iterations, _ = skfuzzy.cluster.cmeans(X.T, 2, seed=SEED, **FCM)
    hard = u.argmax(axis=0)
    predicted = membership(X,centers)
    assert np.allclose(predicted.sum(axis=1),1)
    assert adjusted_rand_score(truth,hard)==1 and iterations<FCM['maxiter']-1
    assert adjusted_rand_score(predicted.argmax(axis=1),hard)==1
    traced,delta=fit_fcm(X,2,SEED)
    assert np.array_equal(traced[0],centers) and np.array_equal(traced[1],u)
    assert delta<FCM['error'], 'Instrumentation preserves exact updates and measures stopping criterion'
    metrics=validity(X,km.labels_,np.arange(len(X)),2)
    assert metrics['silhouette'] > .8 and metrics['dbi'] < .2
    # Consecutive windows only: no pairs across a gap, turbine, or split.
    frame=pd.DataFrame({'turbine':['a']*4+['b'],'split':['train']*3+['validation']*2,
        'window_start':pd.to_datetime(['2022-01-01 00:00','2022-01-01 01:00','2022-01-01 03:00',
                                     '2022-07-01 00:00','2022-07-01 01:00'],utc=True)})
    pairs, _, runs, _ = temporal_summary(frame,np.array([0,1,1,0,0]))
    assert len(pairs)==1 and pairs.changed.iloc[0] and runs.length_hours.sum()==5
    print('Synthetic clustering and temporal checks passed; excluded from research evidence.')


class Experiment:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.input_dir = self.root/'artifacts/kelmarsh_2022'
        self.out = self.input_dir/'clustering'
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/'figures').mkdir(exist_ok=True)
        (self.out/'models').mkdir(exist_ok=True)
        self.models, self.labels, self.rows = {}, {}, []
        self.stability_rows, self.failures = [], []
        self.started = time.perf_counter()
        threadpool_limits(limits=4)

    def table(self, name, rows):
        frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(self.out/name,index=False)
        return frame

    def log_failure(self, stage, config, error):
        self.failures.append({'stage':stage,'configuration':config,'error':str(error)})
        json_out(self.out/'experiment_failures.json',self.failures)

    def environment(self):
        versions = {name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scipy','scikit-learn','matplotlib','scikit-fuzzy','hdbscan','nbformat','nbclient','ipykernel']}
        json_out(self.out/'environment_versions.json',dict(python=platform.python_version(),
            executable=os.sys.executable,packages=versions,threads=4,
            memory_total_bytes=psutil.virtual_memory().total, memory_available_bytes=psutil.virtual_memory().available))
        json_out(self.out/'experiment_configuration.json',dict(seed=SEED,seeds=SEEDS,kmeans=KM,fcm=FCM,
            kmeans_initial_k_range=[2,10],fcm_k_range=list(range(2,11)),ward_k_range=list(range(2,11)), silhouette_sample_size=3000,ward_target_sample=4000,
            tiny_cluster_threshold=.01,ambiguity_membership_threshold=.60,
            selection_period='July–December reused in preprocessing, not an unbiased test',
            feature_scaler_refit=False))
        json_out(self.out/'execution_status.json',dict(status='RUNNING_CANDIDATES',
            scope='KMeans/FCM/Ward k=2–10; six HDBSCAN configurations; frozen preprocessing',
            preprocessing_rerun=False))

    def load(self):
        required = ['windows_2022.csv.gz','preprocessing_selection.json','fitted_preprocessing.pkl',
                    'manual_preprocessing_decision.csv','local_preprocessing_changes.json']
        absent = [name for name in required if not (self.input_dir/name).is_file()]
        if absent: raise FileNotFoundError(f'Preprocessing artifacts missing: {absent}')
        original_manifest = json.loads((self.out/'input_manifest.json').read_text(encoding='utf-8'))
        raw_files = [p for p in original_manifest['files'] if p['role'] in ['raw_scada','status_context_not_ground_truth','original_preprocessing_notebook']]
        for record in raw_files:
            if digest(record['path']) != record['sha256']:
                raise ValueError(f'Original input changed: {record["path"]}')
        self.frame = pd.read_csv(self.input_dir/'windows_2022.csv.gz')
        df = self.frame
        self.features = [c for c in df if c.startswith('coordinate__')]
        expected = {f'coordinate__{signal}__{stat}' for signal in list(UNITS)+['direction_sin','direction_cos']
                    for stat in ['mean','std','min','max','endpoint_change']}
        assert len(self.features)==60 and set(self.features)==expected, 'No silent feature dropping or adding'
        assert len(df)==52241, f'Unexpected window count: {len(df)}; investigate before modeling'
        assert len(df.turbine.unique())==6
        assert not df[['turbine','window_start']].duplicated().any()
        # CSV timezone must be explicit, then normalized to UTC.
        assert df.window_start.str.contains(r'(?:\+00:00|Z)$',regex=True).all()
        df['window_start']=pd.to_datetime(df.window_start,utc=True)
        assert df.window_start.eq(df.window_start.dt.floor('h')).all()
        assert df.window_start.min()>=pd.Timestamp('2022-01-01',tz='UTC')
        assert df.window_start.max()<pd.Timestamp('2023-01-01',tz='UTC')
        assert df.completeness.eq(1).all()
        assert set(df.split)=={'train','validation'}
        cut=pd.Timestamp('2022-07-01',tz='UTC')
        assert df.loc[df.split.eq('train'),'window_start'].lt(cut).all()
        assert df.loc[df.split.eq('validation'),'window_start'].ge(cut).all()
        self.tr=df.split.eq('train').to_numpy()
        self.va=~self.tr
        self.X=df[self.features].to_numpy(dtype=np.float64)
        assert np.isfinite(self.X).all()
        self.Xtr=self.X[self.tr];self.Xva=self.X[self.va]
        self.selection=json.loads((self.input_dir/'preprocessing_selection.json').read_text(encoding='utf-8'))
        for key,value in {'scalar_scaler':'standard','gap_fill_signals':['ambient'],
                          'max_whole_gap_steps':2,'window_hours':1,'window_completeness':1.0,'smoothing':'none'}.items():
            assert self.selection[key]==value,(key,self.selection[key])
        # Locally generated pickle with verified provenance; never refit its scaler.
        with (self.input_dir/'fitted_preprocessing.pkl').open('rb') as stream:
            fitted=pickle.load(stream)
        assert self.features==['coordinate__'+c for c in fitted['window_features']]
        stats=df[fitted['window_features']].fillna(fitted['window_training_medians'])
        reconstructed=fitted['window_scaler'].transform(stats)
        assert np.allclose(self.X,reconstructed,rtol=1e-10,atol=1e-10)
        assert np.allclose(fitted['window_scaler'].mean_,stats.loc[self.tr].mean().to_numpy())
        for stat in ['mean','std','min','max','endpoint_change']:
            pair=[fitted['window_features'].index('direction_'+s+'__'+stat) for s in ['sin','cos']]
            assert fitted['window_scaler'].scale_[pair[0]]==fitted['window_scaler'].scale_[pair[1]]
        self.sil_sample=np.sort(np.random.default_rng(SEED).choice(len(self.Xva),min(3000,len(self.Xva)),replace=False))
        self.table('silhouette_sample.csv',df.loc[self.va,['turbine','split','window_start']].iloc[self.sil_sample].reset_index().rename(columns={'index':'window_row'}))
        manifest_files=raw_files+[dict(path=str(self.input_dir/name),bytes=(self.input_dir/name).stat().st_size,
                                     sha256=digest(self.input_dir/name),role='preprocessing_artifact') for name in required]
        manifest_files += [dict(path=str(self.root/name),sha256=digest(self.root/name),role='implementation')
                           for name in ['kelmarsh_2022.py','kelmarsh_stage3.py','stage3_structural.py','prepare_local_preprocessing.py']]
        json_out(self.out/'input_manifest.json',dict(status='VERIFIED',files=manifest_files,
            clustering_input=dict(path=str(self.input_dir/required[0]),sha256=digest(self.input_dir/required[0]),
            shape=list(df.shape),feature_shape=list(self.X.shape),train_windows=int(self.tr.sum()),
            selection_windows=int(self.va.sum()),first_timestamp=df.window_start.min(),last_timestamp=df.window_start.max()),
            checks='finite, unique, UTC-aligned, complete, 60 declared coordinates, frozen training scaler reproduced'))
        json_out(self.out/'feature_columns.json',self.features)
        json_out(self.out/'physical_profile_units.json',{f'{key}__mean':val for key,val in UNITS.items()})
        print(f'Verified {self.X.shape}; train={self.tr.sum()}, selection={self.va.sum()}')

    def diagnostics(self, key, k, labels, centers):
        scores=validity(self.Xva,labels[self.va],self.sil_sample,k)
        pairs, _, _, _=temporal_summary(self.frame.loc[self.va],labels[self.va])
        scores.update(method=key.split('_')[0],model=key,k=k,evaluation_population='full_selection_windows',
            evaluation_n=int(self.va.sum()),silhouette_n=len(self.sil_sample),
            turbine_nmi=float(normalized_mutual_info_score(self.frame.loc[self.va,'turbine'],labels[self.va])),
            transition_rate=float(pairs.changed.mean()),persistence=float(1-pairs.changed.mean()),
            centroid_min_separation=float(pdist(centers).min()))
        return scores

    def kmeans_sweep(self):
        rows=[]
        for k in range(2,11):
            started=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model=KMeans(n_clusters=k,random_state=SEED,**KM).fit(self.Xtr)
            key=f'KMeans_k{k}'
            labels=model.predict(self.X)
            row=self.diagnostics(key,k,labels,model.cluster_centers_)
            row.update(inertia_train=float(model.inertia_),n_iter=int(model.n_iter_),
                converged=bool(model.n_iter_<KM['max_iter'] and not caught),runtime_seconds=time.perf_counter()-started)
            self.models[key]=model;self.labels[key]=labels
            rows.append(row)
            self.table('kmeans_sweep.csv',rows)
            print(f'{key}: silhouette={row["silhouette"]:.4f}, DBI={row["dbi"]:.4f}, CHI={row["chi"]:.1f}',flush=True)
        self.kmeans_rows=rows
        self.rows.extend(rows)
        sweep=pd.DataFrame(rows)
        # Shortlist spans distinct internal-metric optima and a near-optimal richer partition.
        ks=set(sweep.nlargest(2,'silhouette').k.astype(int))
        ks.add(int(sweep.loc[sweep.dbi.idxmin(),'k']))
        ks.add(int(sweep.loc[sweep.chi.idxmax(),'k']))
        self.shortlist=list(range(2,11))
        boundary=[]
        for metric,maximum in [('silhouette',True),('dbi',False),('chi',True)]:
            best=sweep.loc[sweep[metric].idxmax() if maximum else sweep[metric].idxmin()]
            if int(best.k) in [2,10]:
                boundary.append({'metric':metric,'best_k':int(best.k),'response':
                    'k=1 has undefined internal validity and no regime partition' if int(best.k)==2 else
                    'Extend k to 11 and 12 as a separate, logged boundary sensitivity'})
        if any(item['best_k']==10 for item in boundary):
            for k in [11,12]:
                started=time.perf_counter()
                model=KMeans(n_clusters=k,random_state=SEED,**KM).fit(self.Xtr)
                key=f'KMeans_k{k}';labels=model.predict(self.X)
                row=self.diagnostics(key,k,labels,model.cluster_centers_)
                row.update(inertia_train=float(model.inertia_),n_iter=int(model.n_iter_),
                    converged=model.n_iter_<KM['max_iter'],runtime_seconds=time.perf_counter()-started)
                self.models[key]=model;self.labels[key]=labels;self.rows.append(row)
            self.table('kmeans_boundary_sensitivity.csv',[r for r in self.rows if r['k']>10])
        json_out(self.out/'boundary_review.json',boundary)
        json_out(self.out/'fcm_candidate_selection.json',dict(candidate_k=self.shortlist,
            metric_diagnostic_shortlist=sorted(ks),ward_candidate_k=self.shortlist,
            rationale='User requested the identical full k=2–10 range for K-Means, FCM and Ward; metric optima remain diagnostics only'))

    def fcm_challenger(self):
        rows=[];initial=[];histories={}
        for k in self.shortlist:
            best=None;fallback=None;run_labels=[]
            for seed in SEEDS:
                started=time.perf_counter()
                result,delta=fit_fcm(self.Xtr,k,seed)
                centers,u,u0,d,jm,p,fpc=result
                converged=delta<FCM['error']
                histories[f'k{k}_seed{seed}']=jm.tolist()
                record=dict(method='FCM',k=k,seed=seed,objective=float(jm[-1]),fpc_train=float(fpc),
                    iterations=int(p),final_membership_delta=delta,converged=converged,runtime_seconds=time.perf_counter()-started)
                initial.append(record)
                if not converged:
                    self.log_failure('FCM initialization',{'k':k,'seed':seed,**FCM},
                                     f'Reached iteration cap ({p}); excluded from best-converged initialization selection')
                run_labels.append(membership(self.Xva,centers).argmax(axis=1))
                candidate=dict(centers=centers,objective=float(jm[-1]),fpc_train=float(fpc),
                               iterations=int(p),seed=seed,converged=bool(converged))
                if fallback is None or jm[-1]<fallback['objective']:fallback=candidate
                if converged and (best is None or jm[-1]<best['objective']):best=candidate
            for a,b in itertools.combinations(range(len(SEEDS)),2):
                self.stability_rows.append(dict(method='FCM',k=k,seed_a=SEEDS[a],seed_b=SEEDS[b],
                    evaluation_population='same_full_selection_windows',ari=adjusted_rand_score(run_labels[a],run_labels[b])))
            self.table('fcm_initializations.csv',initial)
            json_out(self.out/'fcm_objective_histories.json',histories)
            if best is None:
                self.log_failure('FCM',{'k':k,**FCM},'No initialization converged; excluded from selection')
                best=fallback  # Keep failed k visible diagnostically; final_refit rejects it.
            key=f'FCM_k{k}'
            U=membership(self.X,best['centers'])
            labels=U.argmax(axis=1)
            entropy=-(U*np.log(np.fmax(U,np.finfo(float).eps))).sum(axis=1)/np.log(k)
            strength=U.max(axis=1)
            row=self.diagnostics(key,k,labels,best['centers'])
            vaU=U[self.va]
            row.update(objective_train=best['objective'],fpc_train=best['fpc_train'],
                fpc_selection=float(np.mean(np.sum(vaU**2,axis=1))),
                mean_strength_selection=float(strength[self.va].mean()),
                normalized_entropy_selection=float(entropy[self.va].mean()),
                ambiguous_fraction_selection=float((strength[self.va]<.60).mean()),
                iterations=best['iterations'],best_seed=best['seed'],converged=best['converged'],
                runtime_seconds=sum(r['runtime_seconds'] for r in initial if r['k']==k))
            row['membership_uniformity_warning']=bool(row['normalized_entropy_selection']>0.999
                and abs(row['fpc_selection']-1/k)<1e-6)
            diagnostic=self.frame.loc[self.va,['turbine','split','window_start']].copy()
            diagnostic['cluster']=labels[self.va]
            diagnostic['entropy']=entropy[self.va];diagnostic['strength']=strength[self.va]
            diagnostic=diagnostic.sort_values(['turbine','window_start'])
            group=diagnostic.groupby('turbine')
            consecutive=diagnostic.window_start.sub(group.window_start.shift()).eq(pd.Timedelta(hours=1))
            diagnostic['changed']=diagnostic.cluster.ne(group.cluster.shift())
            for changed in [False,True]:
                part=diagnostic[consecutive & diagnostic.changed.eq(changed)]
                row['entropy_after_transition' if changed else 'entropy_after_persistence']=float(part.entropy.mean())
                row['ambiguity_after_transition' if changed else 'ambiguity_after_persistence']=float(part.strength.lt(.60).mean())
            diagnostic.to_csv(self.out/f'fcm_membership_diagnostics_k{k}.csv.gz',index=False,compression='gzip')
            self.models[key]=best;self.labels[key]=labels;rows.append(row)
            self.table('fcm_comparison.csv',rows)
            print(f'{key}: silhouette={row["silhouette"]:.4f}, ambiguous={row["ambiguous_fraction_selection"]:.3f}',flush=True)
        self.fcm_rows=rows;self.rows.extend(rows)
        self.table('fcm_comparison.csv',rows)

    def ward_sensitivity(self):
        free=psutil.virtual_memory().available
        n=min(4000 if free>2*1024**3 else 2000,len(self.Xtr))
        estimated=4*n*(n-1)//2*8
        json_out(self.out/'ward_memory_plan.json',dict(sample_n=n,available_bytes=free,
            estimated_condensed_and_workspace_bytes=estimated,full_year_hierarchy_run=False))
        idx=np.sort(np.random.default_rng(SEED).choice(len(self.Xtr),n,replace=False))
        X=self.Xtr[idx]
        sample=np.sort(np.random.default_rng(SEED+1).choice(n,min(3000,n),replace=False))
        self.table('ward_sample.csv',self.frame.loc[self.tr,['turbine','split','window_start']].iloc[idx].reset_index().rename(columns={'index':'window_row'}))
        rows=[]
        for k in self.shortlist:
            print(f'Ward sensitivity k={k}, sample_n={n}',flush=True)
            try:
                started=time.perf_counter()
                ward=AgglomerativeClustering(n_clusters=k,linkage='ward',metric='euclidean').fit_predict(X)
                km=KMeans(n_clusters=k,random_state=SEED,**KM).fit_predict(X)
                for method,labels in [('Ward',ward),('KMeans_same_sample',km)]:
                    row=validity(X,labels,sample,k)
                    row.update(method=method,k=k,evaluation_population='same_deterministic_training_sample',
                        evaluation_n=n,silhouette_n=len(sample),
                        ward_vs_kmeans_ari=float(adjusted_rand_score(ward,km)),
                        ward_vs_full_training_kmeans_ari=float(adjusted_rand_score(ward,self.labels[f'KMeans_k{k}'][self.tr][idx])),
                        runtime_seconds=time.perf_counter()-started)
                    rows.append(row)
            except Exception as error:
                self.log_failure('Ward',{'k':k,'sample_n':n},error)
            self.table('ward_sensitivity.csv',rows)

    def initialization_stability(self):
        for row in [r for r in self.rows if r['method']=='KMeans']:
            k=row['k'];runs=[]
            for seed in SEEDS:
                if seed==SEED: labels=self.labels[f'KMeans_k{k}'][self.va]
                else:
                    model=KMeans(n_clusters=k,random_state=seed,**KM).fit(self.Xtr)
                    labels=model.predict(self.Xva)
                runs.append(labels)
            for a,b in itertools.combinations(range(len(SEEDS)),2):
                self.stability_rows.append(dict(method='KMeans',k=k,seed_a=SEEDS[a],seed_b=SEEDS[b],
                    evaluation_population='same_full_selection_windows',ari=float(adjusted_rand_score(runs[a],runs[b]))))
        stability=self.table('initialization_stability.csv',self.stability_rows)
        for row in self.rows:
            scope=stability[(stability.method==row['method']) & (stability.k==row['k'])].ari
            row['seed_ari_mean']=float(scope.mean());row['seed_ari_min']=float(scope.min())
        self.table('model_comparison.csv',self.rows)

    def candidate_profiles(self):
        profiles=[];sizes=[];qualities=[];distribution=[];monthly=[];temporal=[];transitions=[];runs=[]
        physical=[f'{name}__mean' for name in UNITS]
        flags=[c for c in self.frame if c.endswith('__fraction') or c.startswith('coordinate_imputed__')]
        for row in self.rows:
            key=row['model'];labels=self.labels[key]
            data=self.frame.copy();data['cluster']=labels
            for split in ['train','validation']:
                part=data[data.split.eq(split)]
                for quantile,label in [(.25,'q25'),(.5,'median'),(.75,'q75')]:
                    profile=part.groupby('cluster')[physical].quantile(quantile).reset_index()
                    profile.insert(0,'model',key);profile.insert(1,'split',split);profile['quantile']=label
                    profiles.append(profile)
                counts=part.groupby('cluster').size().reindex(range(row['k']),fill_value=0)
                for cluster,count in counts.items():
                    sizes.append(dict(model=key,split=split,cluster=cluster,count=int(count),fraction=float(count/len(part))))
                if flags:
                    quality=part.groupby('cluster')[flags].mean().reset_index()
                    quality.insert(0,'model',key);quality.insert(1,'split',split);qualities.append(quality)
                dist=part.groupby(['turbine','cluster']).size().rename('count').reset_index()
                dist['fraction_within_turbine']=dist['count']/dist.groupby('turbine')['count'].transform('sum')
                dist.insert(0,'model',key);dist.insert(1,'split',split);distribution.append(dist)
                month=part.assign(month=part.window_start.dt.strftime('%Y-%m')).groupby(['month','cluster']).size().rename('count').reset_index()
                month['fraction_within_month']=month['count']/month.groupby('month')['count'].transform('sum')
                month.insert(0,'model',key);monthly.append(month)
            _,t,r,tmat=temporal_summary(self.frame,labels)
            for table in [t,r,tmat]:table.insert(0,'model',key)
            temporal.append(t);runs.append(r);transitions.append(tmat)
        self.table('candidate_physical_profiles.csv',pd.concat(profiles,ignore_index=True))
        self.table('candidate_cluster_sizes.csv',sizes)
        self.table('candidate_quality_burden.csv',pd.concat(qualities,ignore_index=True))
        self.table('candidate_distribution_by_turbine.csv',pd.concat(distribution,ignore_index=True))
        self.table('candidate_distribution_by_month.csv',pd.concat(monthly,ignore_index=True))
        self.table('candidate_temporal_persistence.csv',pd.concat(temporal,ignore_index=True))
        self.table('candidate_temporal_transitions.csv',pd.concat(transitions,ignore_index=True))
        self.table('candidate_temporal_runs.csv',pd.concat(runs,ignore_index=True))
        # Quality association: group indicator burden remains outside distance space.
        quality_rows=[]
        for row in self.rows:
            key=row['model'];labels=self.labels[key][self.va]
            flags_data=self.frame.loc[self.va,flags].astype(float)
            for family in ['physical_','missing_','filled_','empirical_iqr_','empirical_z_','coordinate_imputed__']:
                cols=[c for c in flags if c.startswith(family)]
                if not cols:continue
                any_burden=flags_data[cols].gt(0).any(axis=1)
                quality_rows.append(dict(model=key,family=family,burden_prevalence=float(any_burden.mean()),
                    cluster_burden_nmi=float(normalized_mutual_info_score(any_burden,labels))))
        self.table('candidate_quality_association.csv',quality_rows)

    def selection_evidence(self):
        comparison=pd.DataFrame(self.rows)
        # A review table only: no automatic rank voting or hidden final decision.
        print(comparison[['model','silhouette','dbi','chi','min_cluster_fraction','seed_ari_min','persistence','turbine_nmi']].to_string(index=False))
        self.comparison=comparison
        lines=['# Candidate selection evidence\n',
            'STATUS: numerical experiments complete; physical and quality profiles require evidence review before final refit.\n',
            'Fit: January–June. Comparison: July–December reused during preprocessing; not an independent holdout.\n',
            'All candidates use the same 60 coordinates and the same saved 3,000-window silhouette subset. DBI/CHI use the entire selection period.\n',
            'Ward results use a separate shared training sample, and are not ranked against selection-period values.\n',
            comparison.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else comparison.to_csv(index=False),
            '\nFuzzy validity uses argmax assignments; FPC varies with k. Initialization ARI and temporal persistence measure distinct phenomena.\n',
            '\nSee candidate_physical_profiles.csv, candidate_quality_burden.csv, candidate_quality_association.csv and candidate_temporal_persistence.csv for review.\n']
        (self.out/'candidate_selection_report.md').write_text('\n'.join(lines),encoding='utf-8')
        json_out(self.out/'candidate_execution_status.json',dict(status='CANDIDATES_EXECUTED_REVIEW_PENDING',
            elapsed_seconds=time.perf_counter()-self.started,models=len(self.rows),failures=self.failures))

    def final_refit(self, selected_model, rationale):
        """Called only after measured candidate evidence has been reviewed."""
        if selected_model not in self.models:raise ValueError('Unknown selected candidate')
        row=next(r for r in self.rows if r['model']==selected_model)
        if not row['converged']:raise ValueError('Cannot select a nonconverged candidate')
        k=row['k'];started=time.perf_counter()
        if row['method']=='KMeans':
            model=KMeans(n_clusters=k,random_state=SEED,**KM).fit(self.X)
            labels=model.labels_;centers=model.cluster_centers_
            if model.n_iter_>=KM['max_iter']:raise RuntimeError('Final KMeans did not converge')
            artifact=model
        else:
            result,delta=fit_fcm(self.X,k,self.models[selected_model]['seed'])
            centers,u,_,_,jm,p,fpc=result
            if delta>=FCM['error']:raise RuntimeError('Final FCM did not converge')
            labels=u.argmax(axis=0)
            artifact=dict(centers=centers,m=2.0,objective=jm[-1],fpc=fpc)
        # Keep raw estimator labels and use an explicit canonical descriptive ordering.
        power_means=pd.Series(self.frame['power__mean'].to_numpy()).groupby(labels).median()
        order=power_means.sort_values().index.tolist();mapping={int(old):new for new,old in enumerate(order)}
        canonical=np.array([mapping[int(x)] for x in labels])
        self.final_labels=canonical;self.final_model=selected_model
        with (self.out/'models/final_model.pkl').open('wb') as stream:pickle.dump(artifact,stream)
        config=dict(selected_candidate=selected_model,method=row['method'],k=k,
            candidate_fit_period='January–June',final_fit_period='full year descriptive refit',
            preprocessing_scalers='frozen January–June fitted parameters; not refit',
            seed=SEED if row['method']=='KMeans' else self.models[selected_model]['seed'],
            kmeans=KM if row['method']=='KMeans' else None,fcm=FCM if row['method']=='FCM' else None,
            raw_to_canonical_labels=mapping,rationale=rationale,runtime_seconds=time.perf_counter()-started,
            candidate_to_refit_ari=float(adjusted_rand_score(self.labels[selected_model],labels)),
            feature_columns=self.features,input_sha256=digest(self.input_dir/'windows_2022.csv.gz'))
        json_out(self.out/'final_model_configuration.json',config)
        assignments=self.frame[['turbine','split','window_start','completeness']].copy()
        assignments['candidate_cluster']=self.labels[selected_model]
        assignments['final_estimator_cluster']=labels
        assignments['cluster']=canonical
        assignments.to_csv(self.out/'final_cluster_assignments.csv.gz',index=False,compression='gzip')
        data=self.frame.copy();data['cluster']=canonical
        physical=[c for c in data if c in [f'{s}__{stat}' for s in UNITS for stat in ['mean','std','min','max','endpoint_change']]]
        for q,name in [(.25,'q25'),(.5,'median'),(.75,'q75')]:
            self.table(f'cluster_profiles_{name}.csv',data.groupby('cluster')[physical].quantile(q).reset_index())
        counts=data.groupby('cluster').size().rename('count').reset_index();counts['fraction']=counts['count']/len(data)
        self.table('cluster_sizes.csv',counts)
        for col,name in [('turbine','turbine'),('month','month')]:
            if col=='month':data['month']=data.window_start.dt.strftime('%Y-%m')
            dist=data.groupby([col,'cluster']).size().rename('count').reset_index()
            dist['fraction']=dist['count']/dist.groupby(col)['count'].transform('sum')
            self.table(f'cluster_distribution_by_{name}.csv',dist)
        flags=[c for c in data if c.endswith('__fraction') or c.startswith('coordinate_imputed__')]
        self.table('quality_burden_by_cluster.csv',data.groupby('cluster')[flags].mean().reset_index())
        pairs,persistence,runs,transitions=temporal_summary(self.frame,canonical)
        self.table('temporal_transitions.csv',transitions)
        self.table('temporal_persistence_by_turbine.csv',persistence)
        self.table('temporal_runs.csv',runs.reset_index(drop=True))
        # Setpoint is joined for physical context only; it never enters the 60 coordinates.
        context=pd.read_csv(self.input_dir/'preprocessed_2022.csv.gz',usecols=['turbine','timestamp','setpoint'])
        context['window_start']=pd.to_datetime(context.timestamp,utc=True).dt.floor('h')
        context=context.groupby(['turbine','window_start']).setpoint.mean().reset_index()
        joined=data[['turbine','window_start','cluster']].merge(context,on=['turbine','window_start'],how='left',validate='one_to_one')
        self.table('cluster_setpoint_context.csv',joined.groupby('cluster').setpoint.agg(['count','median','min','max']).reset_index())
        self.final_config=config
        print(f'Final descriptive refit: {selected_model}; ARI to candidate={config["candidate_to_refit_ari"]:.5f}')

    def figures(self):
        plt.rcParams.update({'font.size':10,'figure.dpi':120})
        def save(fig,name):
            fig.tight_layout();fig.savefig(self.out/'figures'/name,dpi=300,bbox_inches='tight');plt.close(fig)
        km=pd.DataFrame(self.kmeans_rows)
        for metric,ylabel,name in [('silhouette','Silhouette (dimensionless)','k_silhouette.png'),
                                  ('dbi','Davies–Bouldin index (dimensionless)','k_dbi.png'),
                                  ('chi','Calinski–Harabasz index (dimensionless)','k_chi.png'),
                                  ('inertia_train','Training SSE (standardized coordinate units²)','k_inertia.png')]:
            fig,ax=plt.subplots(figsize=(7,4));ax.plot(km.k,km[metric],'-o',label='K-Means++')
            ax.set(xlabel='Number of clusters k',ylabel=ylabel,title='K-Means candidate sweep');ax.legend();save(fig,name)
        comp=pd.DataFrame(self.rows)
        fig,ax=plt.subplots(figsize=(9,4))
        for method,g in comp.groupby('method'):ax.plot(g.k,g.silhouette,'-o',label=method)
        ax.set(xlabel='Number of clusters k',ylabel='Selection silhouette (same sample)',title='Candidate method comparison');ax.legend();save(fig,'method_comparison.png')
        if not hasattr(self,'final_labels'):return
        data=self.frame.copy();data['cluster']=self.final_labels;k=len(np.unique(self.final_labels))
        counts=data.cluster.value_counts().sort_index()
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(counts.index,counts.values)
        ax.set(xlabel='Descriptive cluster',ylabel='One-hour windows (count)',title='Full-year descriptive cluster sizes',xticks=counts.index);save(fig,'cluster_sizes.png')
        fig,axes=plt.subplots(2,5,figsize=(17,7))
        for ax,(signal,unit) in zip(axes.flat,UNITS.items()):
            groups=data.groupby('cluster')[signal+'__mean']
            med=groups.median();q25=groups.quantile(.25);q75=groups.quantile(.75)
            ax.errorbar(med.index,med.values,yerr=[med-q25,q75-med],fmt='o',capsize=3,label='Median and IQR')
            ax.set(xlabel='Cluster',ylabel=f'{signal} ({unit})',title=signal,xticks=range(k));ax.legend(fontsize=7)
        fig.suptitle('Physical profiles of hourly means in original units');save(fig,'physical_profiles.png')
        sample=data.sample(min(12000,len(data)),random_state=SEED)
        fig,ax=plt.subplots(figsize=(8,5))
        for label,part in sample.groupby('cluster'):
            ax.scatter(part.wind__mean,part.power__mean,s=4,alpha=.35,label=f'Cluster {label}')
        ax.set(xlabel='Hourly mean wind speed (m/s)',ylabel='Hourly mean active power (kW)',title='Full-year regimes: wind and production');ax.legend(markerscale=3);save(fig,'wind_power.png')
        for field,name,xlabel in [('turbine','turbine_distribution.png','Turbine'),('month','temporal_distribution.png','Month (2022 UTC)')]:
            data['month']=data.window_start.dt.strftime('%Y-%m')
            tab=pd.crosstab(data[field],data.cluster,normalize='index')
            fig,ax=plt.subplots(figsize=(10,5));tab.plot.bar(stacked=True,ax=ax)
            ax.set(xlabel=xlabel,ylabel='Fraction of retained hourly windows',title='Descriptive regime distribution');ax.legend(title='Cluster');save(fig,name)
        transitions=pd.read_csv(self.out/'temporal_transitions.csv')
        matrix=transitions.groupby(['previous_cluster','cluster'])['count'].sum().unstack(fill_value=0).reindex(index=range(k),columns=range(k),fill_value=0)
        rate=matrix.div(matrix.sum(axis=1),axis=0)
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(rate,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(k),yticks=range(k),xlabel='Next cluster',ylabel='Previous cluster',title='Consecutive-hour transition probabilities')
        for i in range(k):
            for j in range(k):ax.text(j,i,f'{rate.iloc[i,j]:.2f}',ha='center',color='white' if rate.iloc[i,j]>.6 else 'black')
        fig.colorbar(im,ax=ax,label='Conditional probability');save(fig,'transition_matrix.png')

    def save_candidates(self):
        for key,model in self.models.items():
            if getattr(self,'reused_centroid_cache',False) and (self.out/'models'/f'{key}.pkl').is_file():continue
            with (self.out/'models'/f'{key}.pkl').open('wb') as stream:pickle.dump(model,stream)
        json_out(self.out/'experiment_failures.json',self.failures)

    def paper_report(self):
        def markdown(frame,columns=None):
            table=frame[columns].copy() if columns else frame.copy()
            def value(x):
                return f'{x:.5g}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
            lines=['| '+' | '.join(table.columns)+' |','| '+' | '.join(['---']*len(table.columns))+' |']
            lines+=['| '+' | '.join(value(x) for x in row)+' |' for row in table.itertuples(index=False,name=None)]
            return '\n'.join(lines)
        chosen=next(r for r in self.rows if r['model']==self.final_model)
        profiles=pd.read_csv(self.out/'cluster_profiles_median.csv')
        sizes=pd.read_csv(self.out/'cluster_sizes.csv')
        quality=pd.read_csv(self.out/'quality_burden_by_cluster.csv')
        persistence=pd.read_csv(self.out/'temporal_persistence_by_turbine.csv')
        trans=pd.read_csv(self.out/'temporal_transitions.csv')
        p_same=trans.loc[trans.previous_cluster.eq(trans.cluster),'count'].sum()/trans['count'].sum()
        full_nmi=normalized_mutual_info_score(self.frame.turbine,self.final_labels)
        descriptions=[]
        for _,p in profiles.iterrows():
            cluster=int(p.cluster);size=sizes[sizes.cluster.eq(cluster)].iloc[0]
            q=quality[quality.cluster.eq(cluster)].iloc[0]
            descriptions.append(
                f"Klaster {cluster} memuat {int(size['count']):,} window ({100*size.fraction:.2f}%). "
                f"Median mean per jam: angin {p['wind__mean']:.3f} m/s, daya aktif {p['power__mean']:.2f} kW, "
                f"daya reaktif {p['reactive__mean']:.2f} kvar, rotor {p['rotor__mean']:.3f} RPM, "
                f"generator {p['generator__mean']:.2f} RPM, pitch {p['pitch__mean']:.3f}°, "
                f"temperatur ambien {p['ambient__mean']:.2f} °C, bearing generator {p['bearing__mean']:.2f} °C, "
                f"oli gearbox {p['oil__mean']:.2f} °C, dan frekuensi {p['frequency__mean']:.4f} Hz. "
                + (f"Rata-rata fraksi flag frekuensi ialah {q['physical_frequency__fraction']:.5f}. " if 'physical_frequency__fraction' in q else '')
                + 'Nama klaster merupakan deskripsi statistik, bukan label kondisi operasi terverifikasi.')
        comparison=pd.DataFrame(self.rows)
        columns=['model','k','silhouette','dbi','chi','min_cluster_fraction','seed_ari_mean','seed_ari_min','persistence','turbine_nmi','runtime_seconds']
        ward=pd.read_csv(self.out/'ward_sensitivity.csv')
        report=f"""# Hasil Eksperimen Clustering Tahap 3 Kelmarsh 2022

## A. Metode clustering

Eksperimen menggunakan {len(self.frame):,} window UTC non-overlap berdurasi satu jam dengan completeness 100%, dan 60 koordinat statistik terstandar. Mean, simpangan baku sampel, minimum, maksimum, dan perubahan endpoint dihitung untuk sepuluh sinyal skalar serta sin/cos arah angin. Fitur turbine, waktu, split, setpoint, dan flag kualitas tidak masuk ruang jarak. StandardScaler global dan scaler koordinat window tetap memakai parameter Januari–Juni. Median training hanya mengisi statistik window dengan provenance yang diekspor. Input dan checksum: `input_manifest.json`; urutan fitur: `feature_columns.json`.

Preprocessing merupakan **rekonstruksi lokal yang diotorisasi**, bukan replikasi identik helper asli yang tidak tersedia. Sel benchmark numerik, threshold, pemilihan, dan ekspor berasal dari notebook asli dan dipertahankan. Implementasi loader serta interpolasi whole-gap direkonstruksi; perubahan tercatat dalam `local_preprocessing_changes.json` pada folder artefak preprocessing. Uji dan pemeriksaan konsistensi menghasilkan 314.018 observasi tersedia, 315.360 slot terjadwal, 19 sel ambien terisi, 52.241 window dan 60 koordinat. Metadata eksternal, CSV site tambahan, serta `papers.json` tidak direkayasa. Audit full-text/literature terstruktur belum dapat direproduksi dari resource yang tersedia. Tidak dibuat klaim PRISMA final. StandardScaler mengurangi mean training dan membagi dengan simpangan baku training; pasangan sin/cos memakai skala bersama. IQR, atau rentang antarkuartil, menyatakan selisih kuartil 75% dan 25%.

K-Means++ mengelompokkan vektor berdasarkan jarak Euclidean ke centroid, dengan inisialisasi K-Means++, n_init=20, max_iter=500, tol=0,0001, algoritma Lloyd dan seed 2022. Ketiga metode diuji pada seluruh k=2–10 sesuai permintaan penyamaan rentang; pemeriksaan batas K-Means tercatat dalam `boundary_review.json`. Fuzzy C-Means (FCM) menghasilkan bobot keanggotaan yang jumlahnya satu; challenger memakai m=2, error=0,00001, maksimal 500 iterasi dan lima seed untuk setiap k={self.shortlist}. Optimum metrik hanya diagnostik, bukan pembatas rentang. Ward menggabungkan cluster dengan kenaikan SSE terkecil dan digunakan sebagai sensitivitas pada sampel training deterministik untuk setiap k yang sama. Penetapan tiga metode, ruang fitur, serta parameter merupakan keputusan benchmark Kelmarsh sesuai protokol penelitian; adaptasi ini tidak diklaim sebagai replikasi penuh sebelas paper kerja.

Sebanyak {int(self.tr.sum()):,} window Januari–Juni dipakai untuk fitting dan {int(self.va.sum()):,} window Juli–Desember untuk seleksi. Periode seleksi sudah digunakan saat preprocessing, sehingga bukan final holdout independen. Silhouette mengukur kohesi terhadap separasi pada subset deterministik yang sama ({len(self.sil_sample):,} window); DBI mengukur rasio sebaran terhadap jarak centroid; CHI membandingkan dispersi antarklaster dan intraklaster pada seluruh window seleksi. Adjusted Rand Index (ARI) mengukur kesepakatan partisi antarseed pada observasi yang sama. Persistence mengukur kesamaan label pada pasangan satu jam yang benar-benar berurutan; ini berbeda dari stabilitas inisialisasi. Normalized Mutual Information (NMI) mengukur asosiasi label dengan identitas turbin.

## B. Hasil eksperimen aktual

Seluruh kandidat yang dievaluasi pada periode seleksi:

{markdown(comparison,columns)}

Sumber: `model_comparison.csv`, `kmeans_sweep.csv`, `fcm_comparison.csv`. SSE, jumlah kuadrat jarak terhadap centroid, hanya menjadi diagnostik K-Means. Hasil tiap inisialisasi FCM beserta norm perubahan membership akhir dan objective tersedia dalam `fcm_initializations.csv`; riwayat objective dalam `fcm_objective_histories.json`; failure aktual dalam `experiment_failures.json`. Konvergensi diperiksa memakai norm aktual perubahan membership di bawah 0,00001, dengan instrumentasi update library yang tidak mengubah perhitungannya. Metrik FCM memakai argmax membership sehingga menyederhanakan informasi fuzzy. FPC (Fuzzy Partition Coefficient) ialah rata-rata jumlah kuadrat membership; baseline keanggotaan seragam adalah 1/k. Entropy ternormalisasi mengukur ketidakpastian membership, dengan nilai satu untuk keanggotaan seragam. FPC tidak dipakai sebagai pemilih tunggal. Runtime kandidat K-Means mencakup fitting dan diagnostik, sedangkan runtime FCM menjumlahkan fitting lima inisialisasi; runtime tahap lengkap dicatat dalam file execution notebook, sehingga kolom runtime kandidat tidak dipakai untuk perbandingan waktu dengan cakupan identik.

Sensitivitas Ward dan K-Means pada sampel training yang sama:

{markdown(ward,['method','k','evaluation_n','silhouette','dbi','chi','ward_vs_kmeans_ari'])}

Sumber: `ward_sensitivity.csv`; sampel: `ward_sample.csv`; perencanaan memori: `ward_memory_plan.json`. Angka ini berasal dari populasi berbeda dengan tabel seleksi dan tidak diperlakukan sebagai perbandingan langsung terhadap validation.

## C. Konfigurasi terpilih dan trade-off

Konfigurasi yang paling didukung untuk deskripsi akhir ialah **{self.final_model}**. {self.final_config['rationale']}

Metrik kandidat terpilih: Silhouette {chosen['silhouette']:.5f}, DBI {chosen['dbi']:.5f}, CHI {chosen['chi']:.2f}, minimum proporsi klaster {100*chosen['min_cluster_fraction']:.3f}%, ARI rata-rata antarseed {chosen['seed_ari_mean']:.5f}, ARI minimum {chosen['seed_ari_min']:.5f}, persistence seleksi {chosen['persistence']:.5f}, dan NMI turbine {chosen['turbine_nmi']:.5f}. Sumber: baris `{self.final_model}` pada `model_comparison.csv`. Keputusan tidak didasarkan pada voting ranking tiga metrik. CHI dibaca bersama ukuran kelompok, profil fisik, kualitas, stabilitas, serta kompleksitas.

Model kemudian di-refit secara deskriptif pada seluruh tahun dengan scaler preprocessing tetap. ARI antara assignment kandidat dan assignment refit pada seluruh observasi ialah {self.final_config['candidate_to_refit_ari']:.5f}; refit dapat mengubah partisi sehingga profil berikut berasal dari assignment final, bukan disamakan dengan metrik kandidat. Konfigurasi dan pemetaan label: `final_model_configuration.json`. Assignment: `final_cluster_assignments.csv.gz`.

## D. Interpretasi operating regimes

{'\n\n'.join(descriptions)}

Label diurutkan menurut median daya aktif mean per jam. Perbedaan daya, rotor, pitch, temperatur, dan distribusi waktu merupakan bukti deskriptif; tidak menetapkan fault, curtailment, shutdown, atau normal tanpa ground truth. Nilai median dan IQR lengkap tersedia dalam `cluster_profiles_median.csv`, `cluster_profiles_q25.csv`, dan `cluster_profiles_q75.csv`. Power setpoint digunakan hanya untuk konteks dalam `cluster_setpoint_context.csv`.

Persistence final gabungan ialah {p_same:.5f} pada {int(trans['count'].sum()):,} pasangan berurutan; NMI turbin final {full_nmi:.5f}. Rincian periode/turbin tersedia dalam `temporal_persistence_by_turbine.csv`, distribusi dalam `cluster_distribution_by_turbine.csv` dan `cluster_distribution_by_month.csv`. Missingness, flag fisik/empiris, serta coordinate-imputed provenance diringkas dalam `quality_burden_by_cluster.csv`. Asosiasi kualitas kandidat tersedia dalam `candidate_quality_association.csv`; flag bukan fitur clustering dan bukan ground truth kerusakan.

## E. Jawaban RQ3–RQ4

RQ3: konfigurasi {self.final_model} didukung oleh bukti internal, ukuran klaster, stabilitas inisialisasi, struktur waktu, profil fisik, serta audit kualitas sebagaimana dibahas pada bagian C. Kesimpulan terbatas pada representasi 60 koordinat dan kandidat yang diuji. Sensitivitas Ward menyediakan pemeriksaan struktur pada sampel bersama, bukan validasi eksternal.

RQ4: partisi dapat dijelaskan melalui profil SCADA aktual pada bagian D, dengan bukti ukuran serta median/IQR tiap kelompok. Interpretasi merupakan operating regimes deskriptif dan tidak mengukur ketepatan terhadap true operational state. Tidak diklaim menjawab RQ1 atau RQ2 secara kausal karena tidak ada perbandingan preprocessing/representasi terkendali pada eksperimen Tahap 3 ini.

## F. Keterbatasan

Tidak tersedia true operating-regime labels. Juli–Desember bukan independent final holdout. Window memiliki ketergantungan temporal; persistence tidak membuktikan independensi atau kebenaran label. Rare state dan outlier tetap ambigu. Mean/min/max serta rotor/generator berpotensi redundan sehingga bobot Euclidean dipengaruhi redundansi. Resolusi 10 menit tidak menangkap transien lebih cepat. Metrik internal tidak merupakan validasi fisik mutlak. Uji stabilitas hanya mencakup seed dan konfigurasi yang tercatat. Rekonstruksi helper tidak membuktikan kesetaraan kode dengan helper asli meskipun consistency checks cocok; resource metadata dan audit full-text asli masih merupakan gap provenance.

## G. Sumber bukti dan reproduksi

Semua tabel di atas berasal dari CSV yang disebutkan pada paragraf terkait. Grafik 300 dpi terdapat di `figures/`: sweep validitas, inertia, perbandingan kandidat, ukuran klaster, profil fisik, wind–power, distribusi bulanan/turbin dan transisi. Profil fisik memakai satuan asli; tidak ada PCA/UMAP dalam input atau pemilihan model. Notebook utama `Kelmarsh_Clustering_Stage3_Local.ipynb` menjalankan workflow melalui Run All pada kernel bersih dengan dependency lokal. Versi: `environment_versions.json`; konfigurasi: `experiment_configuration.json`; sumber input: `input_manifest.json`; model: `models/`. Laporan ini tidak mengedit paper LaTeX final.
"""
        # Keep the evidence discussion readable, and link each named existing artifact.
        for marker in ['DBI k=2','k=5 sampai','FCM m=2','Ward k=2','NMI turbine kandidat']:
            report=report.replace('. '+marker,'.\n\n'+marker)
        for directory in [self.out,self.input_dir,self.root]:
            for path in directory.iterdir():
                if path.is_file():
                    report=report.replace('`'+path.name+'`',f'[{path.name}](<{path.as_posix()}>)')
        (self.root/'STAGE3_PAPER_RESULTS.md').write_text(report,encoding='utf-8')
        (self.out/'candidate_selection_report.md').write_text(report[report.index('## C.'):report.index('## D.')],encoding='utf-8')
        json_out(self.out/'execution_status.json',dict(status='COMPLETED',
            runtime_seconds=time.perf_counter()-self.started,selected_configuration=self.final_model,
            candidate_models=len(self.rows),failed_experiments=self.failures,
            remaining_provenance_gaps=['unavailable original helper source','external asset metadata','original structured full-text literature audit']))
