#!/usr/bin/env python3
"""Reproducible offline batch experiment. No PostgreSQL or GPU required."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from adaptive import Node, Region, read_tree, write_tree, adapt, train_forest

HERE=Path(__file__).resolve().parent

def integers(path,cols,bits):
    with open(path) as f: x=json.load(f) if str(path).endswith('.json') else np.loadtxt(f,ndmin=2)
    a=np.asarray(x)
    if a.ndim!=2 or a.shape[1]!=cols or not np.issubdtype(a.dtype,np.number):
        raise ValueError(f'{path}: expected N x {cols} numeric array')
    if not np.isfinite(a).all() or np.any(a!=np.floor(a)): raise ValueError('Use fixed-domain integer coordinates; do not truncate OSM silently')
    a=a.astype(np.int64)
    upper=np.array([(1<<b)-1 for b in bits]*(cols//2))
    if np.any(a<0) or np.any(a>upper): raise ValueError('Coordinate outside tree domain')
    if cols==4 and np.any(a[:,:2]>a[:,2:]): raise ValueError('Expected xmin ymin xmax ymax, inclusive')
    return a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true')
    p.add_argument('--tree');p.add_argument('--data');p.add_argument('--new-data');p.add_argument('--queries');p.add_argument('--new-queries')
    p.add_argument('--out',default='adaptive_run')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sample',type=int,default=2000)
    p.add_argument('--rollouts',type=int,default=32)
    p.add_argument('--levels',type=int,default=3)
    p.add_argument('--threshold',type=float,default=.1)
    p.add_argument('--budget',type=float,default=.5)
    p.add_argument('--block-size',type=int,default=50)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--learn-initial',action='store_true',help='Train a fresh initial tree on training queries only; prevents inherited training leakage')
    a=p.parse_args()
    if min(a.sample,a.rollouts,a.levels,a.block_size,a.repeats)<1: p.error('positive sample/rollouts/levels/block-size/repeats required')
    if not 0<a.budget<=1 or not 0<=a.threshold<=1:p.error('budget in (0,1], threshold in [0,1]')
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory to preserve earlier measurements')
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)
    if a.demo:
        bits=(10,10)
        tree=Node(0,Node(1,Node(0,Node(),Node()),Node(0,Node(),Node())),Node(1,Node(0,Node(),Node()),Node(0,Node(),Node())))
        old=rng.integers(0,1024,(10000,2)); new=old.copy()
        ids=np.flatnonzero((old[:,0]<512)&(old[:,1]<512))
        new[ids,0]=rng.integers(0,64,len(ids));new[ids,1]=rng.integers(0,512,len(ids))
        centers=rng.integers(0,1024,(300,2)); widths=rng.integers(5,160,(300,2))
        qo=np.concatenate([np.maximum(0,centers-widths//2),np.minimum(1023,centers+widths//2)],axis=1)
        qn=qo.copy(); qn[:150]=np.column_stack([rng.integers(0,48,150),rng.integers(0,300,150),rng.integers(48,64,150),rng.integers(300,512,150)])
    else:
        if not all((a.tree,a.data,a.new_data,a.queries)):p.error('Need --tree --data --new-data --queries (or --demo)')
        tree,bits=read_tree(a.tree)
        old=integers(a.data,2,bits);new=integers(a.new_data,2,bits)
        qo=integers(a.queries,4,bits);qn=integers(a.new_queries,4,bits) if a.new_queries else qo.copy()
        if old.shape!=new.shape: p.error('This controlled-cardinality runner requires equal snapshot shapes')
        if qo.shape!=qn.shape:p.error('Old/new query windows must have the same number of queries for disjoint positional split')
    if len(qo)<10 or not len(old):p.error('Need >=10 queries and nonempty data')
    # Partition indices once: no held-out workload enters detection or training.
    order=rng.permutation(len(qo));cut=int(.8*len(order))
    train_ids=order[:cut]; test_ids=order[cut:];test=qo[test_ids]
    # Catch exact duplicate test rectangles also appearing in either training window.
    heldout={tuple(q) for q in test}
    train_old=np.array([q for q in qo[train_ids] if tuple(q) not in heldout],dtype=np.int64).reshape(-1,4)
    train_new=np.array([q for q in qn[train_ids] if tuple(q) not in heldout],dtype=np.int64).reshape(-1,4)
    if not len(train_old) or not len(train_new):p.error('Too few distinct queries for independent training/test')
    sample_ids=rng.choice(len(old),min(a.sample,len(old)),replace=False)
    initial_report={'mode':'imported tree; previous training/test separation not verified'}
    if a.learn_initial:
        initial_begin=time.perf_counter()
        root=Region((),(0,0),tuple((1<<b)-1 for b in bits),(0,0),-1)
        tree,initial_report=train_forest(Node(0),bits,[root],old[sample_ids],train_old,rng,
            levels=a.levels,rollouts=a.rollouts,block_size=a.block_size)
        initial_report['seconds']=time.perf_counter()-initial_begin
        initial_report['mode']='fresh initial training, disjoint test queries'
    elif a.demo:
        initial_report={'mode':'fixed handcrafted demo tree; no training queries used'}
    begin=time.perf_counter()
    updated,selected,report=adapt(tree,bits,old[sample_ids],new[sample_ids],train_old,train_new,
        seed=a.seed,threshold=a.threshold,budget=a.budget,block_size=a.block_size,levels=a.levels,rollouts=a.rollouts)
    report['adaptation_python_wall_seconds']=time.perf_counter()-begin
    write_tree(out/'old_tree.txt',tree,bits);write_tree(out/'new_tree.txt',updated,bits)
    changed_ids=np.flatnonzero(np.any(old!=new,axis=1))
    np.savetxt(out/'points.tsv',old,fmt='%d')
    np.savetxt(out/'deletions.tsv',changed_ids,fmt='%d')
    np.savetxt(out/'insertions.tsv',new[changed_ids],fmt='%d')
    np.savetxt(out/'test_queries.tsv',test,fmt='%d')
    np.savetxt(out/'regions.tsv',np.array([r.lo+r.hi for r in selected],dtype=np.int64).reshape(-1,4),fmt='%d')
    (out/'split.json').write_text(json.dumps({'train_indices':train_ids.tolist(),'test_indices':test_ids.tolist(),'test_source':'old workload, fixed across all stages','training_duplicates_removed':True},indent=2))
    binary=out/'benchmark'
    command=['g++','-O3','-DNDEBUG','-std=c++17',str(HERE/'benchmark.cpp'),'-o',str(binary)]
    subprocess.run(command,check=True)
    result=subprocess.run([str(binary),str(out),str(a.repeats)],check=True,capture_output=True,text=True)
    metrics=json.loads(result.stdout)
    # Full controller overhead + detection/retraining + in-index rekeying; no
    # double-counting detection/training. File export/compilation are setup.
    metrics['adaptation_seconds']=report['adaptation_python_wall_seconds']+metrics['reindex_seconds']
    u=metrics['update_operations']
    metrics['update_with_adaptation_us']=(metrics['raw_update_seconds']+metrics['adaptation_seconds'])*1e6/u if u else None
    metrics['update_without_adaptation_us']=metrics['raw_update_seconds']*1e6/u if u else None
    final={'implementation':'independent BMTree partial-retraining reconstruction v0.1',
        'not_an_official_reproduction':True,'dataset':'synthetic demo' if a.demo else str(Path(a.data).resolve()),
        'config':vars(a),'python':platform.python_version(),'numpy':np.__version__,
        'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        'points_before':len(old),'points_after':len(new),'test_queries':len(test),
        'training_queries_old':len(train_old),'training_queries_new':len(train_new),
        'initial_training':initial_report,'adaptation':report,'metrics':metrics}
    (out/'results.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final,indent=2))
    print(f'\nSaved: {out / "results.json"}')
if __name__=='__main__':main()
