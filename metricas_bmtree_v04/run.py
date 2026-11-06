#!/usr/bin/env python3
"""Run both RAM indexes on one synthetic 2D workload and print the eight metrics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from adaptive import Node, Region, train_forest, adapt, write_tree

HERE = Path(__file__).resolve().parent

def make_workload(n, seed):
    rng = np.random.default_rng(seed)
    domain = 1 << 20
    seen = set()
    old = []
    while len(old) < n:
        p = tuple(int(v) for v in rng.integers(0, domain, 2))
        if p not in seen:
            seen.add(p)
            old.append(p)
    old = np.array(old, dtype=np.int64)
    new = old.copy()
    ids = np.flatnonzero((old[:, 0] < domain//2) & (old[:, 1] < domain//2))
    # All replacements differ from all original points and from one another.
    for i in ids:
        while True:
            p = (int(rng.integers(0, domain//16)), int(rng.integers(0, domain//2)))
            if p not in seen:
                seen.add(p)
                new[i] = p
                break
    center = rng.integers(0, domain, (300, 2))
    width = rng.integers(domain//100, domain//6, (300, 2))
    queries = np.column_stack([np.maximum(0, center-width//2), np.minimum(domain-1, center+width//2)])
    permutation = rng.permutation(len(queries))
    train_ids, test_ids = permutation[:240], permutation[240:]
    return old, new, ids, queries[train_ids], queries[test_ids], train_ids, test_ids

ROWS = [
    ('latencia_media_consulta_us', 'Latencia media de consulta', 'us/consulta', 3),
    ('qps', 'Throughput de consultas (QPS)', 'consultas/s', 2),
    ('actualizacion_amortizada_us', 'Tiempo medio amortizado de actualizacion', 'us/operacion', 3),
    ('adaptacion_ms', 'Tiempo de adaptacion', 'ms', 3),
    ('degradacion_pct', 'Degradacion de latencia', '%', 3),
    ('recall_pct', 'Recall (micro)', '%', 3),
    ('construccion_total_ms', 'Tiempo total de construccion', 'ms', 3),
    ('memoria_indice_mib', 'Memoria del indice residente', 'MiB', 3),
]

def final_metrics(raw, initial=0.0, controller=0.0, bmtree=False):
    adaptation = raw['adaptation_runtime_seconds'] + controller
    # In FlexFlood maintenance is already inside the timed update loop.
    update = raw['update_runtime_seconds'] + (adaptation if bmtree else 0)
    u = raw['update_operations']
    return {
        'latencia_media_consulta_us': raw['query_after_us'],
        'qps': raw['qps'],
        'actualizacion_amortizada_us': update*1e6/u if u else None,
        'adaptacion_ms': adaptation*1000,
        'degradacion_pct': raw['degradation_pct'],
        'recall_pct': raw['recall']*100,
        'construccion_total_ms': (initial + raw['build_runtime_seconds'])*1000,
        'memoria_indice_mib': raw['index_bytes_after']/(1024**2),
    }

def summary(results):
    lines = ['','METRICAS FINALES', 'Consultas: datos actualizados y mantenimiento completado.', '']
    lines.append(f'{"Metrica":45} {"BMTree*":>16} {"FlexFlood":>16}  Unidad')
    lines.append('-'*100)
    for key, title, unit, places in ROWS:
        vals = []
        for name in ('BMTree', 'FlexFlood'):
            value = results[name]['metrics'][key]
            vals.append('N/A' if value is None else f'{value:,.{places}f}')
        lines.append(f'{title:45} {vals[0]:>16} {vals[1]:>16}  {unit}')
    lines += ['', '* BMTree experimental: reimplementacion independiente con GAS y backup maximo; ver FIDELIDAD.md.',
              'Exactitud: ambos indices coinciden con fuerza bruta en todas las consultas de prueba.',
              'Memoria: estimacion estructural residente, incluye almacenamiento propio de registros.',
              'Adaptacion: suma de componentes en BMTree; mantenimiento durante updates en FlexFlood.',
              'Demo sintetica; no son resultados OSM ni una reproduccion identica de los papers.']
    for name in ('BMTree', 'FlexFlood'):
        raw=results[name]['raw']
        lines.append(f"{name}: TP={raw['true_positives']}, esperados={raw['expected_results']}, "
                     f"FN={raw['false_negatives']}, FP={raw['false_positives']}, "
                     f"consultas verificadas={raw['validated_queries']}, vacias={raw['empty_queries']}.")
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='comparacion_metricas_01')
    parser.add_argument('--points', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample', type=int, default=2000)
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--rollouts', type=int, default=32)
    parser.add_argument('--shift-threshold', type=float, default=.1)
    parser.add_argument('--retrain-budget', type=float, default=.5)
    parser.add_argument('--detect-depth', type=int, default=5)
    parser.add_argument('--block-size', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--flex-learning-seconds', type=float, default=2.0)
    parser.add_argument('--flex-samples', type=int, default=20)
    a = parser.parse_args()
    if a.points < 1000 or min(a.sample, a.levels, a.rollouts, a.repeats) < 1 or a.flex_samples < 4 or a.flex_learning_seconds <= 0:
        parser.error('points>=1000, flex-samples>=4 y presupuestos positivos requeridos')
    if not 0<=a.shift_threshold<=1 or not 0<a.retrain_budget<=1 or min(a.detect_depth,a.block_size)<1 or a.levels>40:
        parser.error('threshold en [0,1], budget en (0,1], depth/block positivos, levels<=40')
    try:
        import sklearn
    except ImportError:
        parser.error('Instala dependencias: python -m pip install -r metricas_bmtree_v04/requirements.txt')
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Elige otro --out para conservar tus resultados anteriores')
    out.mkdir(parents=True, exist_ok=True)
    # Lock because original FlexFlood trainer uses paths relative to its headers.
    import fcntl
    lock = open(HERE/'flex/train/run.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Ya hay una ejecucion usando este paquete; espera a que termine')
    print('1/5 Preparando datos y consultas comunes...', flush=True)
    old, new, ids, train, test, train_ids, test_ids = make_workload(a.points, a.seed)
    assert not (set(map(tuple, train)) & set(map(tuple, test)))
    rng = np.random.default_rng(a.seed)
    sample_ids = rng.choice(len(old), min(a.sample, len(old)), replace=False)
    bits = (20, 20)
    root = Region((), (0, 0), ((1<<20)-1, (1<<20)-1), (0, 0), -1)
    print('2/5 Aprendiendo BMTree y preparando su adaptacion...', flush=True)
    begin = time.perf_counter()
    tree, initial = train_forest(Node(0), bits, [root], old[sample_ids], train, rng,
                                levels=a.levels, rollouts=a.rollouts, block_size=a.block_size)
    initial_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    updated, selected, adaptation = adapt(tree, bits, old[sample_ids], new[sample_ids], train, train,
                                          seed=a.seed, levels=a.levels, rollouts=a.rollouts, block_size=a.block_size,
                                          threshold=a.shift_threshold,budget=a.retrain_budget,max_depth=a.detect_depth)
    controller_seconds = time.perf_counter()-begin
    write_tree(out/'old_tree.txt', tree, bits)
    write_tree(out/'new_tree.txt', updated, bits)
    arrays = {'points': old, 'insertions': new[ids], 'deletions': ids,
              'train_queries': train, 'test_queries': test,
              'regions': np.array([lo+hi for r in selected for lo,hi in r.domains()], dtype=np.int64).reshape(-1,4)}
    for name, array in arrays.items():
        np.savetxt(out/(name+'.tsv'), array, fmt='%d')
    print('3/5 Compilando los dos indices en C++17...', flush=True)
    for backend in ('bmtree', 'flexflood'):
        subprocess.run(['g++', '-O3', '-DNDEBUG', '-std=c++17', str(HERE/(backend+'_metrics.cpp')),
                        '-o', str(out/backend)], check=True)
    results = {}
    raw_results = {}
    env = os.environ.copy()
    env['METRICAS_SEED'] = str(a.seed)
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    for step, backend, name in ((4, 'bmtree', 'BMTree'), (5, 'flexflood', 'FlexFlood')):
        print(f'{step}/5 Midiendo {name} y verificando resultados...', flush=True)
        cmd = [str(out/backend), str(out), str(a.repeats)]
        if backend == 'flexflood':
            cmd += [str(a.flex_learning_seconds), str(a.flex_samples), str(a.seed)]
        run = subprocess.run(cmd, env=env, text=True, capture_output=True)
        (out/(backend+'_stderr.txt')).write_text(run.stderr)
        (out/(backend+'_raw.json')).write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f'{name} fallo (codigo {run.returncode}). Revisa {out}/{backend}_stderr.txt y el JSON. No se publica una comparacion con resultados incorrectos.\n{run.stderr}')
        raw = json.loads(run.stdout)
        if not raw['exact_results_verified']:
            raise RuntimeError(f'{name}: resultados incorrectos')
        raw_results[name] = raw
        results[name] = {'metrics': final_metrics(raw, initial_seconds if backend=='bmtree' else 0,
                           controller_seconds if backend=='bmtree' else 0, backend=='bmtree'), 'raw': raw}
    assert raw_results['BMTree']['checksum_after'] == raw_results['FlexFlood']['checksum_after']
    report = {'config':vars(a), 'dataset':'synthetic unique 2D points; localized distribution shift',
              'dimensions':2, 'points_before':len(old), 'points_after':len(new),
              'training_queries':len(train), 'test_queries':len(test), 'repeats':a.repeats,
              'training_query_indices':train_ids.tolist(), 'test_query_indices':test_ids.tolist(),
              'initial_bmtree_training_seconds':initial_seconds, 'initial_bmtree_training':initial,
              'bmtree_controller_seconds':controller_seconds, 'bmtree_adaptation':adaptation,
              'python':platform.python_version(), 'numpy':np.__version__, 'sklearn':sklearn.__version__,
              'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
              'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.tsv')},
              'results':results,
              'memory_definition':'structural resident owned bytes incl payload; excludes allocator metadata, RSS, temporary training memory',
              'implementation':'independent BMTree v0.4 (GAS + max backup, Sections V-VI); instrumented FlexFlood reference with trainer invocation fixes',
              'validation':'micro recall; all held-out rectangles; exact multiset equality before and after updates'}
    (out/'resultados.json').write_text(json.dumps(report, indent=2))
    table = summary(results)
    (out/'metricas.txt').write_text(table+'\n')
    with open(out/'metricas.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metrica', 'unidad', 'BMTree', 'FlexFlood'])
        for key, title, unit, places in ROWS:
            writer.writerow([title, unit, results['BMTree']['metrics'][key], results['FlexFlood']['metrics'][key]])
    status='SI' if adaptation['retraining_performed'] else 'NO'
    table += f"\nBMTree reentrenamiento ejecutado: {status}; regiones={len(selected)}; motivo={adaptation['reason']}."
    (out/'metricas.txt').write_text(table+'\n')
    print(table)
    print(f'\nGuardado en: {out}/metricas.txt, metricas.csv y resultados.json')

if __name__ == '__main__':
    main()
