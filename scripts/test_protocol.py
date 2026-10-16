"""Small correctness checks independent of model output quality."""
import unittest
import numpy as np
import torch
import cv2
from PIL import Image
from evaluate import edge_metrics, kid
from torchmetrics.image.kid import poly_mmd
from run_experiments import configs

class ProtocolTests(unittest.TestCase):
 def test_matrix(self):
  matrix=configs(); self.assertEqual(len(matrix),18); self.assertEqual(len({c['name'] for c in matrix}),18)
  self.assertEqual(sum(200 for c in matrix),3600)
 def test_identical_edges(self):
  image=np.zeros((512,512,3),dtype=np.uint8); image[100:400,100:400]=255
  gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  reference=Image.fromarray(cv2.Canny(cv2.GaussianBlur(gray,(3,3),1),100,200,apertureSize=3,L2gradient=True))
  result=edge_metrics(Image.fromarray(image),reference,reference)
  self.assertEqual(result['edge_f1'],1); self.assertEqual(result['retained_edge_recall'],1); self.assertEqual(result['edge_distance'],0)
 def test_empty_prediction(self):
  ref=np.zeros((512,512),dtype=np.uint8); ref[100,:]=255
  r=edge_metrics(Image.new('RGB',(512,512)),Image.fromarray(ref),Image.fromarray(ref))
  self.assertEqual(r['edge_f1'],0); self.assertIsNone(r['edge_distance'])
 def test_kid_formula(self):
  rng=torch.Generator().manual_seed(91); x=torch.randn(10,8,generator=rng,dtype=torch.float64); y=torch.randn(10,8,generator=rng,dtype=torch.float64)
  kxx=(x@x.T/8+1)**3; kyy=(y@y.T/8+1)**3; kxy=(x@y.T/8+1)**3
  expected=(kxx.sum()-kxx.trace())/(10*9)+(kyy.sum()-kyy.trace())/(10*9)-2*kxy.mean()
  self.assertAlmostEqual(float(poly_mmd(x,y)),float(expected),places=10)
 def test_kid_reproducible(self):
  g=torch.Generator().manual_seed(3); x=torch.randn(100,8,generator=g); y=torch.randn(100,8,generator=g)
  self.assertEqual(kid(x,y),kid(x,y))

if __name__=='__main__': unittest.main()
