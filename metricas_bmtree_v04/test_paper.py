import itertools
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import adaptive as a

HERE=Path(__file__).resolve().parent

class PaperTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array(list(itertools.product(range(8),repeat=2)))
        self.root=a.Region((),(0,0),(7,7),(0,0),-1)
        self.queries=np.array([[0,0,2,7],[1,2,6,4],[4,0,7,2],[0,3,7,6]])

    def test_maximum_backup_and_printed_uct(self):
        p=a.Policy(a.Node(),(),{})
        for x in [-3,-1,-2]: p.observe(x)
        self.assertEqual(p.value,-1)
        self.assertEqual(p.visits,3)
        self.assertAlmostEqual(a.uct(p,9),-1/3+math.sqrt(math.log(9)/3))

    def test_gas_matches_independent_one_bit_scan(self):
        tree=a.Node(0,a.Node(),a.Node())
        frontier=((('left',),(1,0)),(('right',),(1,0)))
        def cost(t): return a.scan_ranges(t,self.points,self.queries,(3,3),2).sum()
        actions=a.greedy_action(tree,frontier,(3,3),cost)
        current=tree
        for (path,used),action in zip(frontier,actions):
            scores=[]
            for dim in range(2):
                for split in (1,0):
                    node=a.Node(dim,a.Node(),a.Node()) if split else a.Node(dim,child=a.Node())
                    scores.append(cost(a.replace(current,path,node)))
            current,_=a.fill_one(current,path,used,action,(3,3))
            self.assertEqual(cost(current),min(scores))

    def test_shared_nodes_domain_and_semicenters(self):
        tree=a.Node(0,child=a.Node(0,a.Node(),a.Node()))
        rs=list(a.regions(tree,(3,3),4))
        left=next(r for r in rs if r.path==('child','left'))
        right=next(r for r in rs if r.path==('child','right'))
        self.assertEqual(left.ratio((3,3)),.5)
        expected=(self.points[:,0]//2)%2==0
        np.testing.assert_array_equal(left.mask(self.points),expected)
        centers=self.points.astype(float)+.5
        self.assertTrue(np.all(left.mask(centers)^right.mask(centers)))
        self.assertEqual(len(left.domains()),2)
        self.assertEqual(len(a.cells(tree,rs[0],(3,3))),2)

    def test_scan_range_against_explicit_block_search(self):
        tree=a.Node(1,a.Node(0),a.Node(1))
        keys=sorted(map(int,a.encode(tree,self.points,(3,3))))
        expected=[]
        for q in self.queries:
            lo,hi=a.encode(tree,np.array([q[:2],q[2:]]),(3,3))
            starts=keys[::3]
            def block(v): return max(0,max((i for i,k in enumerate(starts) if k<=v),default=0))
            expected.append(block(hi)-block(lo))
        np.testing.assert_array_equal(a.scan_ranges(tree,self.points,self.queries,(3,3),3),expected)

    def test_full_learning_deterministic_exact_and_deeper_than_three(self):
        outputs=[]
        for _ in range(2):
            t,report=a.train_forest(a.Node(),(3,3),[self.root],self.points,self.queries,
                np.random.default_rng(44),levels=5,rollouts=20,block_size=2)
            outputs.append(t)
            self.assertEqual(report['deepest_level'],5)
            self.assertGreater(report['gas_calls'],0)
            keys=a.encode(t,self.points,(3,3)).reshape(8,8)
            self.assertEqual(len(set(keys.flat)),64)
            self.assertTrue((np.diff(keys,axis=0)>0).all())
            self.assertTrue((np.diff(keys,axis=1)>0).all())
        self.assertEqual(*outputs)

    def test_no_shift_does_not_train(self):
        t=a.Node(0,a.Node(1,a.Node(),a.Node()),a.Node(1,a.Node(),a.Node()))
        new,regions,report=a.adapt(t,(3,3),self.points,self.points,self.queries,self.queries)
        self.assertEqual(t,new)
        self.assertFalse(regions)
        self.assertFalse(report['rounds'])

    def test_detector_prunes_below_unshifted_ancestor(self):
        tree=a.Node(0,a.Node(1,a.Node(0),a.Node(0)),a.Node(1,a.Node(0),a.Node(0)))
        # Same quadrant counts; changed distribution one level deeper.
        old=np.array([[0,0],[3,3],[0,4],[3,7],[4,0],[7,3],[4,4],[7,7]])
        new=np.array([[0,0],[1,3],[0,4],[1,7],[4,0],[7,3],[4,4],[7,7]])
        self.assertFalse(a.detect(tree,(3,3),old,new,self.queries,self.queries,alpha=1,threshold=.01))

    def test_algorithm_two_joint_initial_forest(self):
        tree=a.Node(0,a.Node(1,a.Node(),a.Node()),a.Node(1,a.Node(),a.Node()))
        rs=[r for r in a.regions(tree,(3,3),4) if len(r.path)==2]
        candidates=[(2,-1,r,.5) for r in rs]
        calls=[]
        def train(t,b,selected,*args,**kwargs):
            calls.append(tuple(r.path for r in selected))
            return t,{'sr_before':10,'sr_after':8}
        with patch.object(a,'detect',return_value=candidates),patch.object(a,'train_forest',side_effect=train):
            _,selected,_=a.adapt(tree,(3,3),self.points,self.points,self.queries,self.queries,budget=.8)
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0]),3)
        self.assertLess(sum(r.ratio((3,3)) for r in selected),.8)

    def test_real_distribution_shift_triggers_partial_training(self):
        rng=np.random.default_rng(42)
        def make(depth):
            return a.Node(depth%2,make(depth+1),make(depth+1)) if depth<5 else a.Node()
        tree=make(0)
        old=rng.integers(0,32,(1000,2));new=old.copy()
        new[:,0]//=4;new[:,1]//=2
        q=np.array([[x,y,min(x+6,31),min(y+6,31)] for x in range(0,32,3) for y in range(0,32,3)])
        updated,rs,report=a.adapt(tree,(5,5),old,new,q,q,alpha=1,threshold=.01,
            budget=.8,levels=3,rollouts=16,block_size=10)
        self.assertTrue(report['retraining_performed'])
        self.assertGreater(report['rounds'][0]['gas_calls'],0)
        self.assertLess(report['space_ratio'],.8)
        points=np.array(list(itertools.product(range(32),repeat=2)))
        inside=np.zeros(len(points),dtype=bool)
        for r in rs: inside |= r.mask(points)
        np.testing.assert_array_equal(a.encode(tree,points[~inside],(5,5)),a.encode(updated,points[~inside],(5,5)))

    def test_shared_partial_retraining_native_and_untouched_keys(self):
        tree=a.Node(0,child=a.Node(0,a.Node(1),a.Node(1)))
        selected=[r for r in a.regions(tree,(3,3),4) if r.path==('child','left')]
        intervals=list(itertools.combinations_with_replacement(range(8),2))
        boxes=np.array([[x,y,X,Y] for x,X in intervals for y,Y in intervals])
        next_tree,report=a.train_forest(tree,(3,3),selected,self.points,boxes[::25],
            np.random.default_rng(9),levels=3,rollouts=16,block_size=2)
        outside=~selected[0].mask(self.points)
        np.testing.assert_array_equal(a.encode(tree,self.points[outside],(3,3)),a.encode(next_tree,self.points[outside],(3,3)))
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            a.write_tree(tmp/'old_tree.txt',tree,(3,3));a.write_tree(tmp/'new_tree.txt',next_tree,(3,3))
            for name,arr in [('points',self.points),('insertions',self.points[:8]),('deletions',np.arange(8)),('test_queries',boxes),('train_queries',self.queries),('regions',np.array([lo+hi for r in selected for lo,hi in r.domains()]))]:
                np.savetxt(tmp/(name+'.tsv'),arr,fmt='%d')
            subprocess.run(['g++','-O2','-std=c++17',str(HERE/'audit_probe.cpp'),'-o',str(tmp/'probe')],check=True)
            result=subprocess.check_output([str(tmp/'probe'),str(tmp)],text=True).splitlines()
            self.assertEqual(len(result),1296)
            for q,line in zip(boxes,result):
                values=list(map(int,line.split()))
                actual=list(zip(values[1::2],values[2::2]))
                expected=[tuple(p) for p in self.points if q[0]<=p[0]<=q[2] and q[1]<=p[1]<=q[3]]
                self.assertEqual(actual,expected)

if __name__=='__main__': unittest.main()
