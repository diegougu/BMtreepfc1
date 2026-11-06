"""Compare original Node/ZNode methods without importing training dependencies.
Usage: python audit_original_encoder.py /path/to/original/bmtree/tree.py
"""
import ast,itertools,sys
from pathlib import Path
import numpy as np
from adaptive import Node,encode
from audit_independent import random_tree
source=Path(sys.argv[1]).read_text()
names={'compute_child_range','z_order_heuristic','generate_value_from_binary','Node','ZNode'}
body=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
ns={};exec(compile(ast.Module(body=body,type_ignores=[]),sys.argv[1],'exec'),ns)
originalNode,originalZ=ns['Node'],ns['ZNode']
def convert(n,used=(0,0)):
    if n is None:return None
    o=originalNode(0,-1,sum(used),[[0]*3,[1]*3,[0]*3,[1]*3],list(used),[0,63],originalZ([3,3]))
    if n.dim<0:return o
    o.chosen_bit=n.dim;u=list(used);u[n.dim]+=1
    o.left_children=convert(n.left,tuple(u));o.right_children=convert(n.right,tuple(u));o.child=convert(n.child,tuple(u))
    return o
points=np.array(list(itertools.product(range(8),repeat=2)))
for seed in range(100):
    rng=np.random.default_rng(seed)
    t=Node(0,random_tree(rng,(1,0)),random_tree(rng,(1,0)))
    o=convert(t)
    expected=[o.get_value([[int(c) for c in f'{int(x):03b}'] for x in p],0)[0] for p in points]
    np.testing.assert_array_equal(encode(t,points,(3,3)),expected)
print('PASS: 6400 keys across 100 trees agree with original Node/ZNode methods; training not compared.')
