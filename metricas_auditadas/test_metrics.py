"""Integration checks: identical boundary queries, empty results and updates."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from adaptive import Node, write_tree

HERE=Path(__file__).resolve().parent

class CommonBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fcntl
        cls.lock=open(HERE/'flex/train/run.lock','a')
        fcntl.flock(cls.lock,fcntl.LOCK_EX)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        for name in ('bmtree','flexflood'):
            subprocess.run(['g++','-O2','-std=c++17',str(HERE/(name+'_metrics.cpp')),'-o',str(cls.root/name)],check=True)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        cls.lock.close()
    def prepare(self, directory):
        directory.mkdir()
        old=np.array([(i,j) for i in range(32) for j in range(32)],dtype=np.int64)
        added=np.array([(100+i,100+i) for i in range(100)],dtype=np.int64)
        queries=np.array([[0,0,31,31],[0,0,0,0],[31,31,31,31],[5,5,5,5],
                          [100,100,199,199],[100,100,100,100],[200,200,201,201],
                          [0,0,255,255],[5,0,5,31]],dtype=np.int64)
        for name,values in [('points',old),('insertions',added),('deletions',np.arange(100)),
                            ('train_queries',queries[[0,4,7]]),('test_queries',queries),('regions',np.empty((0,4)))]:
            np.savetxt(directory/(name+'.tsv'),values,fmt='%d')
        write_tree(directory/'old_tree.txt',Node(0),(20,20))
        write_tree(directory/'new_tree.txt',Node(0),(20,20))
    def test_boundary_empty_queries_and_updates_both_backends(self):
        work=self.root/'boundary';self.prepare(work)
        results=[]
        for name in ('bmtree','flexflood'):
            cmd=[str(self.root/name),str(work),'2']
            if name=='flexflood':cmd+=['0.1','4','42']
            result=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'METRICAS_SEED':'42'})
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            m=json.loads(result.stdout);results.append(m)
            self.assertTrue(m['exact_results_verified'])
            self.assertEqual(m['recall'],1)
            self.assertEqual(m['false_positives'],0)
            self.assertEqual(m['update_operations'],200)
            self.assertGreater(m['index_bytes_after'],0)
            self.assertAlmostEqual(m['qps']*m['query_after_us'],1e6,places=6)
        self.assertEqual(results[0]['checksum_after'],results[1]['checksum_after'])
    def test_duplicate_initial_points_are_rejected(self):
        work=self.root/'duplicates';self.prepare(work)
        with open(work/'points.tsv','a') as f:f.write('0 0\n')
        for name in ('bmtree','flexflood'):
            cmd=[str(self.root/name),str(work),'1']
            if name=='flexflood':cmd+=['0.1','4','42']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('unicos',result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
