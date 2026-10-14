"""Final artifact integrity, numerical reproducibility, and presentation verification."""
from pathlib import Path
import sys
import json
import pickle
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.stage3_deps'))
import numpy as np
import pandas as pd
import nbformat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from PIL import Image
from kelmarsh_stage3 import Experiment, digest, json_out, KM, SEED

started=time.perf_counter()
experiment=Experiment(ROOT)
experiment.load()
out=experiment.out
checks={}
frozen=json.loads((out/'preprocessing_frozen_manifest.json').read_text(encoding='utf-8'))
assert all(digest(item['path'])==item['sha256'] for item in frozen['files'])
checks['preprocessing_sources_and_core_artifacts_unchanged_during_steering']=True
for name in ['Kelmarsh_2022_Preprocessing_Local_Reconstructed','Kelmarsh_Clustering_Stage3_Local']:
    nb=nbformat.read(ROOT/(name+'.ipynb'),as_version=4)
    nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    status=json.loads((ROOT/(name+'.execution.json')).read_text(encoding='utf-8'))
    assert status['status']=='COMPLETED'
    checks[name]={'all_code_cells_executed':len(code),'no_error_outputs':True,
                  'fresh_kernel_runtime_seconds':status['runtime_seconds']}
original=ROOT.parent/'Kelmarsh_2022_Preprocessing_Literature_Benchmark.ipynb'
change=json.loads((experiment.input_dir/'local_preprocessing_changes.json').read_text(encoding='utf-8'))
assert digest(original)==change['original_sha256']
original_nb=nbformat.read(original,as_version=4)
local_nb=nbformat.read(ROOT/'Kelmarsh_2022_Preprocessing_Local_Reconstructed.ipynb',as_version=4)
assert all(original_nb.cells[i].source==local_nb.cells[i+1].source for i in change['unchanged_numerical_code_cells'])
checks['original_notebook_unchanged']=True
checks['original_numerical_cells_preserved']=len(change['unchanged_numerical_code_cells'])
assignment=pd.read_csv(out/'final_cluster_assignments.csv.gz')
assert len(assignment)==len(experiment.frame)
assert not assignment[['turbine','window_start']].duplicated().any()
config=json.loads((out/'final_model_configuration.json').read_text(encoding='utf-8'))
with (out/'models/final_model.pkl').open('rb') as stream:model=pickle.load(stream)
predicted=model.predict(experiment.X)
assert np.array_equal(predicted,assignment.final_estimator_cluster.to_numpy())
mapping={int(k):v for k,v in config['raw_to_canonical_labels'].items()}
canonical=np.array([mapping[int(x)] for x in predicted])
assert np.array_equal(canonical,assignment.cluster.to_numpy())
rerun=KMeans(n_clusters=config['k'],random_state=SEED,**KM).fit(experiment.X)
assert np.array_equal(rerun.labels_,predicted)
centroid_delta=float(np.max(np.abs(rerun.cluster_centers_-model.cluster_centers_)))
assert np.allclose(rerun.cluster_centers_,model.cluster_centers_,rtol=1e-12,atol=1e-12)
checks['fixed_seed_full_year_assignment_labels_identical']=True
checks['fixed_seed_centroid_max_abs_difference']=centroid_delta
checks['fixed_seed_centroid_comparison_tolerance']={'rtol':1e-12,'atol':1e-12}
checks['fixed_seed_inertia_abs_difference']=float(abs(rerun.inertia_-model.inertia_))
checks['saved_model_assignments_match_all_windows']=True
initial=pd.read_csv(out/'fcm_initializations.csv')
assert initial.converged.eq(initial.final_membership_delta.lt(1e-5)).all()
failed=initial[~initial.converged]
checks['fcm_nonconverged_initializations']=len(failed)
comparison=pd.read_csv(out/'model_comparison.csv')
assert len(comparison)==18
for method in ['KMeans','FCM']:
    assert sorted(comparison.loc[comparison.method.eq(method),'k'].tolist())==list(range(2,11))
assert len(initial)==45 and initial.groupby('k').size().eq(5).all()
assert comparison.evaluation_n.eq(26360).all() and comparison.silhouette_n.eq(3000).all()
ward=pd.read_csv(out/'ward_sensitivity.csv')
assert len(ward)==18
assert sorted(ward.loc[ward.method.eq('Ward'),'k'].tolist())==list(range(2,11))
assert len(pd.read_csv(out/'initialization_stability.csv'))==180
checks['matching_method_k_ranges']={'KMeans':list(range(2,11)),'FCM':list(range(2,11)),'Ward':list(range(2,11))}
checks['candidate_rows']=len(comparison)
checks['same_selection_population_and_silhouette_sample']=True
assert pd.read_csv(out/'cluster_sizes.csv')['count'].sum()==52241
for q in ['median','q25','q75']:
    profile=pd.read_csv(out/f'cluster_profiles_{q}.csv')
    assert profile.shape==(2,51) and np.isfinite(profile.to_numpy()).all()
quality=pd.read_csv(out/'quality_burden_by_cluster.csv')
assert np.isfinite(quality.to_numpy()).all()
# Regenerate only figure formatting and report links from frozen numerical artifacts.
experiment.rows=comparison.to_dict('records')
experiment.kmeans_rows=pd.read_csv(out/'kmeans_sweep.csv').to_dict('records')
experiment.shortlist=json.loads((out/'fcm_candidate_selection.json').read_text(encoding='utf-8'))['candidate_k']
experiment.final_config=config
experiment.final_model=config['selected_candidate']
experiment.final_labels=canonical
experiment.failures=json.loads((out/'experiment_failures.json').read_text(encoding='utf-8'))
experiment.figures()
experiment.paper_report()
from stage3_structural import StructuralStudy
structural=StructuralStudy(experiment)
density=pd.read_csv(out/'hdbscan_comparison.csv')
assert len(density)==6
assert set(zip(density.min_cluster_size,density.min_samples))=={(c,s) for c in [100,300,800] for s in [10,30]}
assert np.allclose(density.assigned_coverage_selection+density.selection_noise_fraction,1)
matched=pd.read_csv(out/'hdbscan_matched_coverage_kmeans.csv')
assert len(matched)==54
for row in density.itertuples():
    labels=pd.read_csv(out/f'hdbscan_assignments_c{row.min_cluster_size}_s{row.min_samples}.csv.gz')
    assert len(labels)==26360 and labels.noise.eq(labels.cluster.eq(-1)).all()
    assert labels.cluster.ge(0).sum()==row.evaluation_n
    assert matched[matched.density_model.eq(row.model)].same_assigned_rows.eq(row.evaluation_n).all()
    with (out/'models'/f'{row.model}.pkl').open('rb') as stream: density_model=pickle.load(stream)
    import hdbscan
    sample=np.sort(np.random.default_rng(SEED).choice(len(experiment.Xva),256,replace=False))
    predicted_density,strength=hdbscan.approximate_predict(density_model,experiment.Xva[sample])
    assert np.array_equal(predicted_density,labels.cluster.to_numpy()[sample])
    assert np.allclose(strength,labels.strength.to_numpy()[sample],rtol=1e-12,atol=1e-12)
    structural.models[row.model]=density_model
    complete_labels=np.full(len(experiment.X),-1,dtype=int)
    complete_labels[experiment.tr]=density_model.labels_
    complete_labels[experiment.va]=labels.cluster.to_numpy()
    structural.labels[row.model]=complete_labels
assert len(pd.read_csv(out/'hdbscan_configuration_stability.csv'))==15
assert len(pd.read_csv(out/'redundancy_sensitivity.csv'))==9
diagnosis=json.loads((out/'feature_space_diagnosis.json').read_text(encoding='utf-8'))
assert diagnosis['main_input_sha256']==digest(experiment.input_dir/'windows_2022.csv.gz')
assert not diagnosis['preprocessing_altered']
checks['hdbscan_six_configurations_and_54_matched_coverage_comparisons']=True
checks['saved_hdbscan_models_reproduce_256_fixed_selection_labels_and_strengths_each']=True
checks['feature_diagnosis_and_separate_redundancy_sensitivity']=True
checks['original_full_array_equivalence_proven']=False
structural.rows=density.to_dict('records')
structural.profiles()
structural.density_figures()
structural.append_report()
figures=[]
for path in (out/'figures').glob('*.png'):
    with Image.open(path) as image:
        assert min(image.size)>900
        assert min(image.info.get('dpi',(0,0)))>299
        figures.append({'path':str(path),'pixels':list(image.size),'dpi':list(image.info['dpi'])})
checks['high_resolution_figures']=figures
checks['runtime_seconds']=time.perf_counter()-started
json_out(out/'final_verification.json',checks)
# Preserve measured notebook runtime instead of the report-only regeneration time.
execution=json.loads((ROOT/'Kelmarsh_Clustering_Stage3_Local.execution.json').read_text(encoding='utf-8'))
status=json.loads((out/'execution_status.json').read_text(encoding='utf-8'))
status.update(runtime_seconds=execution['runtime_seconds'],final_verification_passed=True,
              reporting_regeneration='categorical cluster ticks, terminology and evidence links; numerical algorithms unchanged')
json_out(out/'execution_status.json',status)
candidate_status=json.loads((out/'candidate_execution_status.json').read_text(encoding='utf-8'))
candidate_status['status']='COMPLETED_REVIEW_AND_REFIT'
json_out(out/'candidate_execution_status.json',candidate_status)
json_out(out/'output_manifest.json',[dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))
    for path in sorted(out.rglob('*')) if path.is_file() and path.name!='output_manifest.json'])
print(json.dumps({k:v for k,v in checks.items() if k!='high_resolution_figures'},indent=2))
