import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from adaptive import Node,Region,encode,read_tree,write_tree,regions,js,detect,select,adapt,train_forest

HERE=Path(__file__).resolve().parent

class MappingTests(unittest.TestCase):
    def setUp(self):
        self.bits=(3,3)
        self.points=np.array(list(itertools.product(range(8),repeat=2)))
        self.tree=Node(0,Node(1,Node(0),Node(1)),Node(0,child=Node(1)))
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tree.txt';write_tree(path,self.tree,self.bits)
            tree,bits=read_tree(path)
            self.assertEqual(tree,self.tree);self.assertEqual(bits,self.bits)
    def test_bijection_and_coordinate_monotonicity(self):
        keys=encode(self.tree,self.points,self.bits).reshape(8,8)
        self.assertEqual(len(set(keys.flat)),64)
        self.assertTrue(np.all(np.diff(keys,axis=0)>0))
        self.assertTrue(np.all(np.diff(keys,axis=1)>0))
    def test_all_small_boxes_have_no_false_negatives(self):
        keys=encode(self.tree,self.points,self.bits)
        for x,X in itertools.combinations_with_replacement(range(8),2):
            for y,Y in itertools.combinations_with_replacement(range(8),2):
                bounds=encode(self.tree,[[x,y],[X,Y]],self.bits)
                actual=np.all((self.points>=[x,y])&(self.points<=[X,Y]),axis=1)
                candidate=(keys>=bounds[0])&(keys<=bounds[1])
                self.assertTrue(np.all(candidate[actual]))
    def test_matches_original_repository_split_and_shared_nodes(self):
        sys.path.insert(0,str(HERE.parent))
        try:
            from bmtree.bmtree_env import BMTEnv
            from int2binary import Int2BinaryTransformer
        except ImportError as e:
            self.skipTest(f'Optional original repo dependencies unavailable: {e}')
        for split in (0,1):
            env=BMTEnv(self.points.tolist(),None,list(self.bits),None,Int2BinaryTransformer([7,7]),0,4)
            env.tree.take_action(0,split)
            env.tree.take_action(1,split)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'tree.txt';env.tree.save(path,4)
                loaded,bits=read_tree(path)
                np.testing.assert_array_equal(encode(loaded,self.points,bits),[env.tree.output(p.tolist()) for p in self.points])
    def test_js_empty_and_symmetry(self):
        self.assertEqual(js([0,0],[0,0]),0)
        self.assertAlmostEqual(js([1,0],[0,1]),1)
        self.assertAlmostEqual(js([1,0],[0,0]),1)
        self.assertAlmostEqual(js([2,3],[6,1]),js([6,1],[2,3]))
    def test_no_shift_does_not_retrain(self):
        q=np.array([[0,0,3,3],[4,4,7,7]])
        t,chosen,report=adapt(self.tree,self.bits,self.points,self.points,q,q)
        self.assertEqual(t,self.tree);self.assertEqual(chosen,[])
    def test_region_budget_and_nonoverlap(self):
        rs=list(regions(self.tree,self.bits,4))
        cs=[(len(r.path),-1,r,1) for r in rs]
        chosen=select(cs,self.bits,.5)
        self.assertLess(sum(r.ratio(self.bits) for r in chosen),.5)
        for a,b in itertools.combinations(chosen,2):self.assertFalse(np.any(a.mask(self.points)&b.mask(self.points)))
    def test_partial_training_preserves_other_keys(self):
        r=next(r for r in regions(self.tree,self.bits,3) if r.path==('left',))
        q=np.array([[0,0,2,7],[0,0,0,7],[1,0,1,7]])
        t,report=train_forest(self.tree,self.bits,[r],self.points,q,np.random.default_rng(5),levels=3,rollouts=16,block_size=2)
        outside=~r.mask(self.points)
        np.testing.assert_array_equal(encode(t,self.points[outside],self.bits),encode(self.tree,self.points[outside],self.bits))
        self.assertLessEqual(report['sr_after'],report['sr_before'])
        self.assertEqual(len(set(encode(t,self.points,self.bits))),len(self.points))
    def test_deterministic_search(self):
        r=Region((),(0,0),(7,7),(0,0),-1);q=np.array([[0,0,2,7],[0,0,0,7]])
        a,ra=train_forest(self.tree,self.bits,[r],self.points,q,np.random.default_rng(3),rollouts=8,block_size=2)
        b,rb=train_forest(self.tree,self.bits,[r],self.points,q,np.random.default_rng(3),rollouts=8,block_size=2)
        self.assertEqual(a,b);self.assertEqual(ra,rb)
    def test_cpp_duplicate_points_updates_and_boundary_queries(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            old=np.vstack([self.points,[[0,0],[0,0],[7,7]]])
            new=old.copy();new[[0,1,64]]=[[7,7],[0,0],[3,4]]
            q=np.array([[0,0,7,7],[0,0,0,0],[7,7,7,7],[3,4,3,4],[0,0,0,7],[7,0,7,7],[1,1,1,1],[2,2,5,5],[0,0,7,0],[0,7,7,7]])
            treefile=p/'tree.txt';write_tree(treefile,self.tree,self.bits)
            for name,x in [('old',old),('new',new),('queries',q)]:np.savetxt(p/(name+'.tsv'),x,fmt='%d')
            cmd=[sys.executable,str(HERE/'run.py'),'--tree',str(treefile),'--data',str(p/'old.tsv'),'--new-data',str(p/'new.tsv'),'--queries',str(p/'queries.tsv'),'--out',str(p/'run'),'--sample','67','--rollouts','8','--threshold','0','--repeats','2']
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            report=json.loads((p/'run/results.json').read_text())
            self.assertTrue(report['metrics']['exact_results_verified'])
            self.assertEqual(report['metrics']['update_operations'],6)

if __name__=='__main__':unittest.main(verbosity=2)
