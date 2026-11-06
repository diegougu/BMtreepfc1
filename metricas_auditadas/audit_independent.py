"""Additional audit, independent Python brute-force oracle for native answers."""
import itertools
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from adaptive import Node, Region, regions, encode, write_tree
HERE=Path(__file__).resolve().parent

def random_tree(rng,used=(0,0),depth=0):
    valid=[i for i in range(2) if used[i]<3]
    if not valid or depth>=5 or rng.random()<.2:return Node()
    dim=int(rng.choice(valid));u=list(used);u[dim]+=1
    if rng.random()<.35:return Node(dim,child=random_tree(rng,tuple(u),depth+1))
    return Node(dim,random_tree(rng,tuple(u),depth+1),random_tree(rng,tuple(u),depth+1))

def main():
    points=np.array(list(itertools.product(range(8),repeat=2)))
    intervals=list(itertools.combinations_with_replacement(range(8),2))
    boxes=np.array([[x,y,X,Y] for x,X in intervals for y,Y in intervals])
    rs=list(regions(Node(0,Node(),Node()),(3,3),2))[1:]
    centers=np.array(list(itertools.product(np.arange(0,8,.5),repeat=2)))
    assert np.all(sum(r.mask(centers).astype(int) for r in rs)==1)
    native_queries=0
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);exe=tmp/'probe'
        subprocess.run(['g++','-O2','-std=c++17',str(HERE/'audit_probe.cpp'),'-o',str(exe)],check=True)
        faults=subprocess.check_output([str(exe)],text=True).strip()
        for seed in range(20):
            rng=np.random.default_rng(seed)
            tree=Node(0,random_tree(rng,(1,0)),random_tree(rng,(1,0)))
            next_tree=Node(0,random_tree(rng,(1,0)),tree.right)
            for t in (tree,next_tree):
                keys=encode(t,points,(3,3)).reshape(8,8)
                assert len(set(keys.flat))==64
                assert (np.diff(keys,axis=0)>0).all() and (np.diff(keys,axis=1)>0).all()
            np.testing.assert_array_equal(encode(tree,points[points[:,0]>=4],(3,3)),encode(next_tree,points[points[:,0]>=4],(3,3)))
            order=rng.permutation(64);old=points[order[:48]];added=points[order[48:]]
            final=np.concatenate([old[16:],added]);oracle={tuple(map(int,p)) for p in final}
            write_tree(tmp/'old_tree.txt',tree,(3,3));write_tree(tmp/'new_tree.txt',next_tree,(3,3))
            for name,arr in [('points',old),('insertions',added),('deletions',np.arange(16)),('test_queries',boxes),('train_queries',boxes[:4]),('regions',np.array([[0,0,3,7]]))]:np.savetxt(tmp/(name+'.tsv'),arr,fmt='%d')
            answers=subprocess.check_output([str(exe),str(tmp)],text=True).splitlines()
            assert len(answers)==len(boxes)
            for q,line in zip(boxes,answers):
                words=list(map(int,line.split()));count=words[0];coords=words[1:]
                assert len(coords)==2*count
                actual=list(zip(coords[::2],coords[1::2]))
                expected=sorted((x,y) for x,y in oracle if q[0]<=x<=q[2] and q[1]<=y<=q[3])
                assert actual==expected,(seed,q,actual,expected)
            native_queries+=len(boxes)
    print(json.dumps({'random_tree_pairs':20,'native_queries_after_updates_and_rekey':native_queries,
        'native_queries_before_updates':native_queries,'half_integer_centers':len(centers),
        'fault_injection':faults,'passed':True},indent=2))
if __name__=='__main__':main()
