"""Feature diagnosis and density challenger on the frozen main feature matrix."""
from pathlib import Path
import itertools
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from joblib import Memory
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from kelmarsh_stage3 import KM, SEED, FCM, SEEDS, UNITS, digest, json_out, validity, temporal_summary

class StructuralStudy:
    def __init__(self, experiment):
        self.e=experiment;self.out=experiment.out
        self.models={};self.labels={};self.rows=[]

    def feature_diagnosis(self):
        e=self.e;X=e.Xtr
        names=pd.Series(e.features)
        variance=pd.DataFrame({'feature':e.features,'variance_train':X.var(axis=0),
                              'variance_selection':e.Xva.var(axis=0),
                              'max_abs_train':np.abs(X).max(axis=0)})
        e.table('feature_space_variance.csv',variance)
        corr=np.corrcoef(X,rowvar=False)
        pd.DataFrame(corr,index=e.features,columns=e.features).to_csv(self.out/'feature_space_correlation.csv')
        pairs=[]
        for i,j in itertools.combinations(range(60),2):
            if abs(corr[i,j])>=.95:pairs.append(dict(feature_a=e.features[i],feature_b=e.features[j],correlation=corr[i,j]))
        e.table('high_correlation_pairs.csv',pd.DataFrame(pairs,columns=['feature_a','feature_b','correlation']))
        rng=np.random.default_rng(SEED)
        pairs_idx=rng.integers(len(X),size=(20000,2));pairs_idx=pairs_idx[pairs_idx[:,0]!=pairs_idx[:,1]]
        square=(X[pairs_idx[:,0]]-X[pairs_idx[:,1]])**2
        distances=np.sqrt(square.sum(axis=1))
        signal_groups=names.str.replace('coordinate__','',regex=False).str.rsplit('__',n=1).str[0]
        contribution=[]
        for signal in signal_groups.unique():
            ids=np.flatnonzero(signal_groups.eq(signal))
            contribution.append(dict(signal=signal,coordinates=len(ids),mean_squared_distance=square[:,ids].sum(axis=1).mean(),
                fraction_total_squared_distance=square[:,ids].sum()/square.sum()))
        e.table('feature_group_distance_contributions.csv',contribution)
        norm=np.linalg.norm(X,axis=1)
        threshold=np.quantile(norm,.99);keep=norm<=threshold
        ordinary=distances[keep[pairs_idx].all(axis=1)]
        extreme=distances[~keep[pairs_idx].all(axis=1)]
        stats=dict(sampled_pairs=len(distances),train_dimensions=60,
            distance_quantiles={str(q):float(np.quantile(distances,q)) for q in [.01,.1,.5,.9,.99]},
            distance_cv=float(distances.std()/distances.mean()),
            ordinary_pair_median=float(np.median(ordinary)),extreme_pair_median=float(np.median(extreme)),
            upper_one_percent_norm_threshold=float(threshold),extreme_observations_n=int((~keep).sum()),
            top_one_percent_fraction_total_squared_norm=float((norm[~keep]**2).sum()/(norm**2).sum()),
            rotor_generator_correlations={stat:float(corr[e.features.index('coordinate__rotor__'+stat),e.features.index('coordinate__generator__'+stat)]) for stat in ['mean','std','min','max','endpoint_change']},
            preprocessing_altered=False,main_input_sha256=digest(e.input_dir/'windows_2022.csv.gz'))
        json_out(self.out/'feature_space_diagnosis.json',stats)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].hist(distances,bins=80);axes[0].set(xlabel='Euclidean distance, frozen 60 coordinates',ylabel='Sampled pairs')
        table=pd.DataFrame(contribution).sort_values('fraction_total_squared_distance')
        axes[1].barh(table.signal,table.fraction_total_squared_distance)
        axes[1].set(xlabel='Fraction of total sampled squared distance')
        fig.tight_layout();fig.savefig(self.out/'figures/feature_space_diagnosis.png',dpi=300);plt.close(fig)
        print(json.dumps(stats,indent=2))

    def density_challenger(self):
        e=self.e
        if __import__('os').environ.get('KELMARSH_FORCE_RETRAIN')!='1' and self.restore_density_cache():
            self.profiles();self.density_figures();return
        configurations=[(c,s) for s in [10,30] for c in [100,300,800]]
        json_out(self.out/'hdbscan_protocol.json',dict(configurations=[dict(min_cluster_size=c,min_samples=s) for c,s in configurations],
            fit='all January–June windows',inference='hdbscan.approximate_predict on July–December; frozen training tree',
            main_dimensions=60,metric='euclidean',algorithm='prims_kdtree',cluster_selection_method='eom',
            prediction_data=True,alpha=1.0,cluster_selection_epsilon=0.0,allow_single_cluster=False,
            parameter_grid_rationale='Minimum group sizes ~0.39%, 1.16%, 3.09% of training; two conservativeness levels, bounded grid chosen before results.',
            cluster_count_not_controlled_by_k=True,preprocessing_changed=False,
            api_source='https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html'))
        memory=Memory(str(e.root/'.runtime/hdbscan_cache'),verbose=0)
        matched=[];timing=[]
        for c,s in configurations:
            key=f'HDBSCAN_c{c}_s{s}';started=time.perf_counter()
            print('Executing',key,'on',e.Xtr.shape,flush=True)
            try:
                model=hdbscan.HDBSCAN(min_cluster_size=c,min_samples=s,metric='euclidean',
                    algorithm='prims_kdtree',prediction_data=True,core_dist_n_jobs=1,
                    cluster_selection_method='eom',memory=memory).fit(e.Xtr)
                fit_seconds=time.perf_counter()-started;infer_start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    va_labels,strength=hdbscan.approximate_predict(model,e.Xva)
                infer_seconds=time.perf_counter()-infer_start
                labels=np.full(len(e.X),-1,dtype=int);labels[e.tr]=model.labels_;labels[e.va]=va_labels
                self.models[key]=model;self.labels[key]=labels
                train_clusters=len(set(model.labels_)-{-1});selection_clusters=len(set(va_labels)-{-1})
                assigned=va_labels>=0
                population=np.flatnonzero(assigned)
                # Restrict the SAME fixed 3000 global rows; never resample flattering assignments.
                sil_global=e.sil_sample[assigned[e.sil_sample]]
                local_lookup=np.full(len(e.Xva),-1,dtype=int);local_lookup[population]=np.arange(len(population))
                sample=local_lookup[sil_global]
                scores=validity(e.Xva[assigned],va_labels[assigned],sample,train_clusters) if assigned.any() else dict(silhouette=np.nan,dbi=np.nan,chi=np.nan,occupied_clusters=0,min_cluster_n=0,min_cluster_fraction=np.nan,imbalance_ratio=np.nan,very_small_cluster_warning=True)
                row=dict(model=key,method='HDBSCAN',min_cluster_size=c,min_samples=s,
                    train_clusters=train_clusters,selection_clusters=selection_clusters,
                    train_noise_fraction=float((model.labels_==-1).mean()),selection_noise_fraction=float((va_labels==-1).mean()),
                    assigned_coverage_selection=float(assigned.mean()),evaluation_n=int(assigned.sum()),silhouette_n=len(sample),
                    mean_strength_assigned=float(strength[assigned].mean()) if assigned.any() else np.nan,
                    fit_seconds=fit_seconds,approximate_predict_seconds=infer_seconds,runtime_seconds=time.perf_counter()-started,
                    prediction_procedure='approximate assignment into fixed training tree',
                    scores_population='selection non-noise only; partial coverage',
                    turbine_nmi_all_including_noise=float(normalized_mutual_info_score(e.frame.loc[e.va,'turbine'],va_labels)),
                    turbine_nmi_assigned=float(normalized_mutual_info_score(e.frame.loc[e.va,'turbine'].to_numpy()[assigned],va_labels[assigned])) if assigned.any() else np.nan,
                    warnings='; '.join(str(w.message) for w in caught),**scores)
                pairs,_,_,_=temporal_summary(e.frame.loc[e.va],va_labels)
                real=pairs[(pairs.cluster>=0)&(pairs.previous_cluster>=0)]
                row['persistence_all_including_noise']=float(1-pairs.changed.mean())
                row['persistence_assigned_pairs_only']=float(1-real.changed.mean()) if len(real) else np.nan
                row['consecutive_assigned_pairs']=len(real)
                row['noise_pair_fraction']=float(1-len(real)/len(pairs))
                self.rows.append(row)
                for k in range(2,11):
                    base=e.labels[f'KMeans_k{k}'][e.va]
                    score=validity(e.Xva[assigned],base[assigned],sample,k) if assigned.any() else {}
                    matched.append(dict(density_model=key,baseline=f'KMeans_k{k}',same_assigned_rows=int(assigned.sum()),
                        same_silhouette_rows=len(sample),ari_assigned=float(adjusted_rand_score(va_labels[assigned],base[assigned])) if assigned.any() else np.nan,
                        **score))
                assigned_table=e.frame.loc[e.va,['turbine','split','window_start']].copy()
                assigned_table['cluster']=va_labels;assigned_table['strength']=strength;assigned_table['noise']=~assigned
                assigned_table.to_csv(self.out/f'hdbscan_assignments_c{c}_s{s}.csv.gz',index=False,compression='gzip')
                with (self.out/'models'/f'{key}.pkl').open('wb') as stream:pickle.dump(model,stream)
                e.table('hdbscan_comparison.csv',self.rows);e.table('hdbscan_matched_coverage_kmeans.csv',matched)
                print(key,'clusters',selection_clusters,'coverage',assigned.mean(),'silhouette',scores['silhouette'],flush=True)
            except Exception as error:
                e.log_failure('HDBSCAN',{'min_cluster_size':c,'min_samples':s},error)
                timing.append(dict(model=key,status='FAILED',error=str(error),runtime_seconds=time.perf_counter()-started))
        json_out(self.out/'hdbscan_failed_configurations.json',timing)
        stability=[]
        for a,b in itertools.combinations(self.labels,2):
            x,y=self.labels[a][e.va],self.labels[b][e.va];common=(x>=0)&(y>=0)
            stability.append(dict(model_a=a,model_b=b,ari_all_including_noise=adjusted_rand_score(x,y),
                common_assigned_n=int(common.sum()),ari_common_assigned=adjusted_rand_score(x[common],y[common]) if common.any() else np.nan,
                noise_indicator_agreement=float(np.mean((x==-1)==(y==-1)))))
        e.table('hdbscan_configuration_stability.csv',stability)
        self.profiles()
        self.density_figures()

    def restore_density_cache(self):
        path=self.out/'hdbscan_cache_manifest.json'
        if not path.is_file():return False
        cache=json.loads(path.read_text(encoding='utf-8'))
        if cache['input_sha256']!=digest(self.e.input_dir/'windows_2022.csv.gz') or cache['features']!=self.e.features:return False
        if cache['library_version']!=__import__('importlib').metadata.version('hdbscan'):return False
        if any(digest(item['path'])!=item['sha256'] for item in cache['files']):return False
        rows=pd.read_csv(self.out/'hdbscan_comparison.csv')
        if set(zip(rows.min_cluster_size,rows.min_samples))!={(c,s) for c in [100,300,800] for s in [10,30]}:return False
        for row in rows.itertuples():
            with (self.out/'models'/f'{row.model}.pkl').open('rb') as stream:model=pickle.load(stream)
            assert model.min_cluster_size==row.min_cluster_size and model.min_samples==row.min_samples
            assert model.metric=='euclidean' and model.algorithm=='prims_kdtree' and model.prediction_data
            assert model.cluster_selection_method=='eom' and not model.allow_single_cluster
            assignments=pd.read_csv(self.out/f'hdbscan_assignments_c{row.min_cluster_size}_s{row.min_samples}.csv.gz')
            assert len(assignments)==int(self.e.va.sum())
            assert np.array_equal(assignments.turbine,self.e.frame.loc[self.e.va,'turbine'])
            assert pd.to_datetime(assignments.window_start,utc=True).equals(self.e.frame.loc[self.e.va,'window_start'].reset_index(drop=True))
            labels=np.full(len(self.e.X),-1,dtype=int);labels[self.e.tr]=model.labels_;labels[self.e.va]=assignments.cluster
            self.models[row.model]=model;self.labels[row.model]=labels
        self.rows=rows.to_dict('records')
        json_out(self.out/'hdbscan_cache_reuse.json',dict(validated=True,configurations_reused=6,
            input_sha256=cache['input_sha256'],feature_order_verified=True,model_and_assignment_checksums_verified=True,
            source_execution=cache['source_execution']))
        print('Verified and reused six executed HDBSCAN fits/predictions; regenerating physical/quality evidence.',flush=True)
        return True

    def ward_profiles(self):
        """Retain the exact 4000-row sensitivity population for physical interpretation."""
        e=self.e;sample=pd.read_csv(self.out/'ward_sample.csv');data=e.frame.loc[sample.window_row].copy()
        assert np.array_equal(data.turbine,sample.turbine)
        X=data[e.features].to_numpy();subset=np.sort(np.random.default_rng(SEED+1).choice(len(X),3000,replace=False))
        from sklearn.cluster import AgglomerativeClustering
        stored=pd.read_csv(self.out/'ward_sensitivity.csv')
        profiles=[];quality=[];distribution=[];label_tables=[]
        physical=[f'{name}__mean' for name in UNITS]
        flags=[c for c in data if c.endswith('__fraction') or c.startswith('coordinate_imputed__')]
        for k in range(2,11):
            ward=AgglomerativeClustering(n_clusters=k,linkage='ward',metric='euclidean').fit(X)
            km=KMeans(n_clusters=k,random_state=SEED,**KM).fit(X)
            for name,model in [('Ward',ward),('KMeans_same_sample',km)]:
                labels=model.labels_
                score=validity(X,labels,subset,k);before=stored[(stored.method==name)&(stored.k==k)].iloc[0]
                assert np.allclose([score[x] for x in ['silhouette','dbi','chi']],[before[x] for x in ['silhouette','dbi','chi']],rtol=1e-10,atol=1e-10)
                part=data.copy();part['cluster']=labels
                lab=part[['turbine','window_start','cluster']].copy();lab.insert(0,'method',name);lab.insert(1,'k',k);label_tables.append(lab)
                for q,label in [(.25,'q25'),(.5,'median'),(.75,'q75')]:
                    table=part.groupby('cluster')[physical].quantile(q).reset_index();table.insert(0,'method',name);table.insert(1,'k',k);table['quantile']=label;profiles.append(table)
                table=part.groupby('cluster')[flags].mean().reset_index();table.insert(0,'method',name);table.insert(1,'k',k);quality.append(table)
                table=part.groupby(['turbine','cluster']).size().rename('count').reset_index();table.insert(0,'method',name);table.insert(1,'k',k);distribution.append(table)
        for name,parts in [('physical_profiles',profiles),('quality_burden',quality),('distribution_by_turbine',distribution),('sample_assignments',label_tables)]:
            e.table(f'ward_{name}.csv',pd.concat(parts,ignore_index=True))
        json_out(self.out/'ward_profile_verification.json',dict(sample_n=len(X),k_range=list(range(2,11)),
            existing_metrics_reproduced=True,tolerance={'rtol':1e-10,'atol':1e-10},
            reason='Partition reconstruction only to add missing physical/quality profiles; existing sweep decisions unchanged.',
            validation_prediction_available=False))

    def density_figures(self):
        frame=pd.read_csv(self.out/'hdbscan_comparison.csv')
        fig,axes=plt.subplots(1,2,figsize=(13,4.8))
        labels=frame.model.str.replace('HDBSCAN_','',regex=False).str.replace('_','/',regex=False)
        bars=axes[0].bar(labels,frame.assigned_coverage_selection)
        axes[0].bar_label(bars,labels=[f'{100*x:.2f}%' for x in frame.assigned_coverage_selection],fontsize=8,padding=3)
        axes[0].set(ylabel='Assigned selection coverage',ylim=(0,1),title='HDBSCAN coverage; complement is noise')
        axes[0].tick_params(axis='x',rotation=35)
        axes[1].scatter(frame.assigned_coverage_selection,frame.silhouette)
        grouped=frame.groupby(['assigned_coverage_selection','silhouette']).model.apply(list)
        for (coverage,score),names in grouped.items():
            text='\n'.join(name.replace('HDBSCAN_','').replace('_','/') for name in names)
            right=coverage>frame.assigned_coverage_selection.median()
            axes[1].annotate(text,(coverage,score),xytext=(-5 if right else 5,7),textcoords='offset points',ha='right' if right else 'left',fontsize=8)
        axes[1].margins(x=.15,y=.25)
        axes[1].set(xlabel='Assigned coverage',ylabel='Silhouette on assigned rows only',title='Conditional scores; compare coverage explicitly')
        fig.tight_layout();fig.savefig(self.out/'figures/hdbscan_coverage_scores.png',dpi=300);plt.close(fig)

    def profiles(self):
        e=self.e;profiles=[];quality=[];sizes=[];distributions=[];monthly=[];transitions=[];persistence=[];association=[]
        physical=[f'{name}__mean' for name in UNITS]
        flags=[c for c in e.frame if c.endswith('__fraction') or c.startswith('coordinate_imputed__')]
        for key,labels in self.labels.items():
            data=e.frame.copy();data['cluster']=labels;data['noise']=labels==-1
            for split in ['train','validation']:
                part=data[data.split.eq(split)]
                for q,label in [(.25,'q25'),(.5,'median'),(.75,'q75')]:
                    table=part.groupby('cluster')[physical].quantile(q).reset_index()
                    table.insert(0,'model',key);table.insert(1,'split',split);table['quantile']=label;profiles.append(table)
                table=part.groupby('cluster')[flags].mean().reset_index();table.insert(0,'model',key);table.insert(1,'split',split);quality.append(table)
                table=part.groupby('cluster').size().rename('count').reset_index();table['fraction']=table['count']/len(part)
                table.insert(0,'model',key);table.insert(1,'split',split);sizes.append(table)
                table=part.groupby(['turbine','cluster']).size().rename('count').reset_index();table.insert(0,'model',key);table.insert(1,'split',split);distributions.append(table)
                table=part.assign(month=part.window_start.dt.strftime('%Y-%m')).groupby(['month','cluster']).size().rename('count').reset_index();table.insert(0,'model',key);monthly.append(table)
            _,t,_,tm=temporal_summary(e.frame,labels)
            t.insert(0,'model',key);tm.insert(0,'model',key);persistence.append(t);transitions.append(tm)
            for family in ['physical_','missing_','filled_','empirical_iqr_','empirical_z_','coordinate_imputed__']:
                cols=[c for c in flags if c.startswith(family)]
                if cols:
                    burden=e.frame.loc[e.va,cols].gt(0).any(axis=1)
                    association.append(dict(model=key,family=family,cluster_burden_nmi_including_noise=normalized_mutual_info_score(burden,labels[e.va])))
        for name,parts in [('physical_profiles',profiles),('quality_burden',quality),('cluster_sizes',sizes),('distribution_by_turbine',distributions),('distribution_by_month',monthly),('temporal_persistence',persistence),('temporal_transitions',transitions)]:
            e.table(f'hdbscan_{name}.csv',pd.concat(parts,ignore_index=True) if parts else pd.DataFrame())
        e.table('hdbscan_quality_association.csv',association)

    def redundancy_sensitivity(self):
        e=self.e
        weights=np.array([1/np.sqrt(2) if c.startswith(('coordinate__rotor__','coordinate__generator__')) else 1.0 for c in e.features])
        json_out(self.out/'redundancy_sensitivity_protocol.json',dict(label='separate distance-weighting sensitivity; main preprocessing untouched',
            rationale='Paired rotor/generator contributions overlap; reduce combined weight of the two 5-coordinate families.',
            features=e.features,weights=weights.tolist(),k_range=list(range(2,11)),seed=SEED,kmeans=KM))
        rows=[]
        for k in range(2,11):
            started=time.perf_counter();model=KMeans(n_clusters=k,random_state=SEED,**KM).fit(e.Xtr*weights)
            labels=model.predict(e.Xva*weights)
            weighted=validity(e.Xva*weights,labels,e.sil_sample,k)
            original=validity(e.Xva,labels,e.sil_sample,k)
            rows.append(dict(k=k,ari_with_main=adjusted_rand_score(e.labels[f'KMeans_k{k}'][e.va],labels),
                silhouette_weighted_geometry=weighted['silhouette'],silhouette_original_geometry=original['silhouette'],
                dbi_original_geometry=original['dbi'],chi_original_geometry=original['chi'],
                runtime_seconds=time.perf_counter()-started))
        e.table('redundancy_sensitivity.csv',rows)

    def restore_centroid_cache(self):
        e=self.e;path=self.out/'centroid_cache_manifest.json'
        if not path.is_file():return False
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['input_sha256']!=digest(e.input_dir/'windows_2022.csv.gz'):return False
        if manifest['features']!=e.features or manifest['kmeans']!=KM or manifest['fcm']!=FCM or manifest['seeds']!=SEEDS:return False
        for record in manifest['files']:
            if digest(record['path'])!=record['sha256']:return False
        e.rows=pd.read_csv(self.out/'model_comparison.csv').to_dict('records')
        assert len(e.rows)==18
        e.kmeans_rows=pd.read_csv(self.out/'kmeans_sweep.csv').to_dict('records')
        e.fcm_rows=pd.read_csv(self.out/'fcm_comparison.csv').to_dict('records')
        e.shortlist=list(range(2,11))
        e.stability_rows=pd.read_csv(self.out/'initialization_stability.csv').to_dict('records')
        from kelmarsh_stage3 import membership
        for row in e.rows:
            key=row['model']
            with (self.out/'models'/f'{key}.pkl').open('rb') as stream:model=pickle.load(stream)
            e.models[key]=model
            e.labels[key]=model.predict(e.X) if row['method']=='KMeans' else membership(e.X,model['centers']).argmax(axis=1)
        e.failures=json.loads((self.out/'experiment_failures.json').read_text(encoding='utf-8'))
        e.reused_centroid_cache=True
        json_out(self.out/'centroid_cache_reuse.json',dict(validated=True,input_sha256=manifest['input_sha256'],
            source_execution=manifest['source_execution'],actual_fits_reused={'KMeans_main_including_seed_stability':45,'FCM':45,'Ward':9,'KMeans_Ward_shared_sample':9},seed_stability_runs_also_reused=True,
            rationale='Retain previously executed valid experiments; extended notebook runs in a fresh kernel and regenerates labels from saved models.'))
        print('Verified cache: all K-Means/FCM/Ward k=2–10 experiments already executed; restoring trained models and measured results.')
        return True

    def append_report(self):
        e=self.e
        def md(frame):return frame.to_markdown(index=False,floatfmt='.5f')
        diagnosis=json.loads((self.out/'feature_space_diagnosis.json').read_text(encoding='utf-8'))
        density=pd.read_csv(self.out/'hdbscan_comparison.csv')
        sensitivity=pd.read_csv(self.out/'redundancy_sensitivity.csv')
        matched=pd.read_csv(self.out/'hdbscan_matched_coverage_kmeans.csv')
        matched_two=matched[matched.baseline.eq('KMeans_k2')][['density_model','same_assigned_rows','same_silhouette_rows','ari_assigned','silhouette','dbi','chi']]
        phys=pd.read_csv(self.out/'hdbscan_physical_profiles.csv')
        reference=phys[(phys.model=='HDBSCAN_c300_s10')&(phys.split=='validation')&(phys['quantile']=='median')]
        report=f"""

## G. Steering tambahan: diagnosis fitur dan structural challenger

Diagnosis dilakukan pada matriks frozen 60 koordinat sebelum pemodelan pada notebook lengkap. Korelasi mean rotor–generator = {diagnosis['rotor_generator_correlations']['mean']:.8f}; koefisien variasi jarak pasangan training = {diagnosis['distance_cv']:.5f}. Median jarak pasangan biasa = {diagnosis['ordinary_pair_median']:.5f}, dibanding pasangan yang melibatkan 1% norma terbesar = {diagnosis['extreme_pair_median']:.5f}. Sebanyak 1% observasi dengan norma terbesar menyumbang {100*diagnosis['top_one_percent_fraction_total_squared_norm']:.3f}% jumlah kuadrat norma. Temuan ini memberi konteks redundansi dan ekor ekstrem, bukan bukti kausal kegagalan algoritma. Nilai variance, korelasi lengkap, dan kontribusi kelompok tersedia pada CSV diagnosis. Preprocessing tidak diubah.

HDBSCAN 0.8.44 fit pada seluruh Januari–Juni; Jul–Des di-assign menggunakan approximate_predict dengan prediction_data=True. Assignment mendekati penempatan dalam tree training yang tetap dan tidak menemukan cluster baru pada periode seleksi. Lihat [API resmi](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html). Grid dibatasi sebelum membaca hasil: min_cluster_size=100/300/800 × min_samples=10/30, metric Euclidean, EOM, alpha=1, epsilon=0, allow_single_cluster=False, prims_kdtree dengan cache per min_samples. HDBSCAN tidak mempunyai target k; jumlah kelompok di bawah merupakan hasil densitas.

{md(density[['model','train_clusters','selection_clusters','assigned_coverage_selection','selection_noise_fraction','silhouette','dbi','chi','silhouette_n','runtime_seconds']])}

Silhouette/DBI/CHI HDBSCAN hanya mencakup non-noise, sehingga tidak setara dengan skor K-Means/FCM full coverage. hdbscan_matched_coverage_kmeans.csv membandingkan K-Means k=2–10 pada observasi assigned dan subset Silhouette yang persis sama untuk setiap konfigurasi HDBSCAN. Noise diberi cluster=-1 untuk tabel profil/kualitas/distribusi; tidak dianggap satu operating regime fisik. Persistence termasuk noise dapat tampak tinggi ketika label -1 dominan; assigned-pair persistence dan coverage pasangan dilaporkan terpisah. HDBSCAN deterministik untuk input tetap; stabilitas yang relevan di sini ialah antar konfigurasi, bukan seed arbitrer.

Berikut pembanding K-Means k=2 pada assigned population identik untuk setiap konfigurasi density; tabel sebelumnya adalah metrik HDBSCAN pada populasi tersebut. Semua k pembanding ada pada CSV, tanpa memilih skor dari coverage yang berbeda.

{md(matched_two)}

Profil fisik referensi HDBSCAN min_cluster_size=300/min_samples=10 di bawah mencakup median seleksi dan baris noise=-1. Seluruh konfigurasi, kuartil, quality burden, NMI, distribusi dan transisi diekspor. Kelompok dengan RPM mendekati nol/pitch besar/temperatur rendah tidak otomatis diberi label shutdown atau fault.

{md(reference[['cluster','wind__mean','power__mean','rotor__mean','generator__mean','pitch__mean','ambient__mean','bearing__mean','oil__mean','frequency__mean']])}

Ward memiliki profil dan quality audit pada sampel training yang persis sama: ward_physical_profiles.csv, ward_quality_burden.csv, ward_distribution_by_turbine.csv dan ward_sample_assignments.csv. Skor partisi saat rekonstruksi profil diverifikasi cocok dengan sweep tersimpan (rtol=atol=1e-10); tidak ada prediction validation Ward yang diklaim.

Sensitivity redundansi tetap memakai 60 kolom: masing-masing koordinat rotor/generator dibobot 1/√2, keluarga lain bobot satu, lalu K-Means k=2–10 fit training dan predict seleksi. Ini eksperimen geometri terpisah, bukan penggantian preprocessing utama; skor dalam geometri asli dilaporkan agar pengaruh label dapat dibandingkan.

{md(sensitivity)}

ARI sensitivitas bobot pada k=2 = {sensitivity.loc[sensitivity.k.eq(2),'ari_with_main'].iloc[0]:.5f}. Selisih ini menunjukkan ketergantungan pada definisi jarak, bukan membuktikan preprocessing alternatif lebih benar. Cluster quality tidak memiliki ground truth pada eksperimen ini.

GMM tidak dijalankan: diagnosis korelasi dan ekor ekstrem belum menunjukkan keuntungan empiris model komponen Gaussian elliptical yang dapat diidentifikasi; korelasi saja bukan justifikasi menambah kandidat. Tidak dibuat klaim bahwa GMM pasti lebih buruk. Klaim dasar paper-algoritma mengikuti mapping steering pengguna (Time2Feat #10, Indoor Building #12, VGO #68, Offshore Wind #33, hierarchical #5); full-text/records 11 paper asli masih unavailable sehingga mapping tersebut belum diaudit secara independen. Ward linkage adalah keputusan eksperimen, FCM biasa bukan optimized FCM paper.

RQ1 dan RQ2 belum terjawab penuh. Ablation paling informatif berikutnya ialah controlled StandardScaler versus RobustScaler dengan window/split/algoritma tetap, serta window 1 jam versus 6 jam dengan cakupan waktu yang dipadankan. Ini usulan lanjutan, bukan hasil eksperimen sekarang. RQ3/RQ4 dibahas dari eksperimen algoritma aktual dan profil fisik.

Audit kesetaraan preprocessing tersedia di PREPROCESSING_EQUIVALENCE_AUDIT.md. Notebook asli, seluruh parameter, artefak, dan helper lokal yang telah dieksekusi dipertahankan sesuai context update. Tidak dilakukan rerun preprocessing. Full-array equivalence terhadap run asli belum dapat dibuktikan karena ekspor run asli tidak tersedia.
"""
        for path in self.out.iterdir():
            if path.is_file():report=report.replace(path.name,f'[{path.name}](<{path.as_posix()}>)')
        destination=e.root/'STAGE3_PAPER_RESULTS.md'
        text=destination.read_text(encoding='utf-8');text=text.split('\n## G. Steering tambahan:')[0]
        destination.write_text(text+report,encoding='utf-8')
        density_status=dict(tested_configurations=len(self.rows),failures=json.loads((self.out/'hdbscan_failed_configurations.json').read_text(encoding='utf-8')))
        status=json.loads((self.out/'execution_status.json').read_text(encoding='utf-8'));status['hdbscan']=density_status
        status['feature_diagnosis_completed']=True;status['redundancy_sensitivity_completed']=True
        json_out(self.out/'execution_status.json',status)
