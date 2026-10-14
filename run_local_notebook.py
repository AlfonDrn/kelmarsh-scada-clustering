"""Execute an actual Jupyter kernel, save success or failure with cell output."""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.stage3_deps'))
os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT/'.stage3_deps'), str(ROOT)])
os.environ['JUPYTER_PATH'] = str(ROOT/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT/'.jupyter/runtime')
os.environ['IPYTHONDIR'] = str(ROOT/'.jupyter/ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT/'.runtime/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'.runtime/numba')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import nbformat
from nbclient import NotebookClient

def main():
    path = ROOT / sys.argv[1]
    kernel = ROOT/'.jupyter/kernels/kelmarsh_local'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Kelmarsh Local Python', 'language': 'python'}), encoding='utf-8')
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    starts = {}
    timing = []
    def on_start(cell, cell_index, **kwargs):
        if cell.cell_type=='code':
            starts[cell_index] = time.perf_counter()
            print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:90]}', flush=True)
    def on_done(cell, cell_index, **kwargs):
        if cell.cell_type=='code':
            elapsed = time.perf_counter()-starts[cell_index]
            timing.append({'cell':cell_index,'seconds':elapsed})
            print(f'Completed cell {cell_index}: {elapsed:.2f}s', flush=True)
            nbformat.write(notebook, path)
    started = time.perf_counter()
    status = 'FAILED'
    client = NotebookClient(notebook, timeout=3600, kernel_name='kelmarsh_local',
        resources={'metadata':{'path':str(ROOT)}}, on_cell_start=on_start,
        on_cell_executed=on_done, record_timing=True)
    try:
        client.execute()
        status = 'COMPLETED'
    finally:
        nbformat.write(notebook, path)
        (path.with_suffix('.execution.json')).write_text(json.dumps({
            'status':status,'runtime_seconds':time.perf_counter()-started,
            'cells':timing,'executable':sys.executable},indent=2),encoding='utf-8')
    print(f'{status}: {path}', flush=True)

if __name__=='__main__':
    main()
