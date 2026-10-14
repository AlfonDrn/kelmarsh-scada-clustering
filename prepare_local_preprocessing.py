"""Create a local copy, retaining all numerical benchmark cells unchanged."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.stage3_deps'))
import nbformat

original = ROOT.parent / 'Kelmarsh_2022_Preprocessing_Literature_Benchmark.ipynb'
notebook = nbformat.read(original, as_version=4)
source = copy.deepcopy(notebook)
changes = []


def replace(index, text, reason):
    before = notebook.cells[index].source
    notebook.cells[index].source = text
    changes.append({'cell_index': index, 'reason': reason,
                    'original_sha256': hashlib.sha256(before.encode()).hexdigest(),
                    'local_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'original_source': before, 'local_source': text})


bootstrap = """from pathlib import Path
import sys, os
ROOT = Path.cwd()
if not (ROOT/'kelmarsh_2022.py').is_file():
    raise RuntimeError('Open/run from the project directory containing kelmarsh_2022.py')
sys.path.insert(0, str(ROOT/'.stage3_deps'))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
"""
first = source.cells[1].source.replace(
    "papers = json.loads((OUT/'papers.json').read_text(encoding='utf-8'))",
    "papers = None  # Original literature JSON unavailable; no invented evidence.")
first = first.replace('capped_interpolate, self_check', 'capped_interpolate, self_check, source_path')
replace(1, bootstrap + first, 'Local dependency path, reconstructed helper, explicit missing literature JSON')
metadata = """import csv, re
from kelmarsh_2022 import read_header as audit_header
# Only header-supported facts are reconstructed. Asset geometry and external mapping remain unavailable.
selected_units = pd.DataFrame([
    {'feature': short, '2022_header': header,
     'header_unit': header.rsplit('(',1)[-1].rstrip(')'),
     'mapping_unit': None, 'unit_source': 'actual 2022 CSV header'}
    for short, header in SIGNALS.items()])
display(selected_units)
selected_units.to_csv(OUT/'selected_units.csv', index=False)
display(Markdown('External turbine metadata and signal mapping are unavailable. '
                 'Units use actual CSV headers; no coordinates, hub heights, or mapping conflict are invented.'))
"""
replace(3, metadata, 'Missing external metadata: retain actual header units; mark unsupported facts unavailable')
for index, old, new in [
    (7, "ROOT/'kelmarsh_data/scada/2022'/turbine/'scada.csv'", "source_path(ROOT,turbine,'scada')"),
    (17, "ROOT/'kelmarsh_data/scada/2022'/turbine/'events.csv'", "source_path(ROOT,turbine,'events')")]:
    replace(index, source.cells[index].source.replace(old, new), 'Adapt raw file paths only')
replace(25, "display(Markdown('Original papers.json unavailable: literature comparison cannot be reconstructed from titles. Numerical benchmarks remain unchanged.'))",
        'Do not fabricate missing structured full-text audit')
replace(67, "display(Markdown('Literature coverage audit is pending original papers.json/full texts. No PRISMA or replication claim is made.'))",
        'Missing literature resource; preserve gap explicitly')
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        ast.parse('\n'.join(l for l in cell.source.splitlines() if not l.lstrip().startswith('%')))
notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
    '# Local reconstruction authorized by the user\n\nOriginal notebook remains unchanged. '
    'The raw loader and whole-gap helper were reconstructed from its declared contract. '
    'All numerical benchmark, selection, and export cells retain their original code. '
    'External metadata, auxiliary site CSVs, and structured literature audit remain unavailable. '
    'This is a documented reconstruction, not an exact replication of the unavailable helper.'))
notebook.cells.append(nbformat.v4.new_code_cell("""assert len(df)==314018 and len(regular)==315360
assert fill_allowed==['ambient'] and selected_scale=='standard'
assert selected_hours==1 and selected_completeness==1.0
assert int(final_table.filter(regex='^filled_').sum().sum())==19
assert len(final_windows)==52241
assert len([c for c in final_windows if c.startswith('coordinate__')])==60
assert np.isfinite(final_windows.filter(regex='^coordinate__').to_numpy()).all()
print('Frozen preprocessing consistency checks passed: 52241 windows, 60 coordinates, 19 ambient fills.')
"""))
nbformat.validate(notebook)
destination = ROOT / 'Kelmarsh_2022_Preprocessing_Local_Reconstructed.ipynb'
nbformat.write(notebook, destination)
out = ROOT/'artifacts/kelmarsh_2022'
out.mkdir(parents=True, exist_ok=True)
unchanged = [i for i,c in enumerate(source.cells) if c.cell_type=='code' and i not in [x['cell_index'] for x in changes]]
for i in unchanged:
    assert source.cells[i].source == notebook.cells[i+1].source
(out/'local_preprocessing_changes.json').write_text(json.dumps({
    'original_path': str(original), 'original_sha256': hashlib.file_digest(original.open('rb'),'sha256').hexdigest(),
    'copy_path': str(destination), 'user_authorized_reconstruction': True,
    'unchanged_numerical_code_cells': unchanged, 'changes': changes,
    'missing_resources': ['original kelmarsh_2022.py','papers.json','turbines.csv','signal_mapping.csv','auxiliary site CSVs'],
    'reconstructed_helper': str(ROOT/'kelmarsh_2022.py')}, indent=2), encoding='utf-8')
print(destination)
