import copy,unittest
from runtime_guard import assert_profile

class ResumeGuardTests(unittest.TestCase):
 def setUp(self):self.profile=dict(gpus=['RTX 5060 Ti, 580.178.04, 16311'],cpu='Ryzen 7 7700',packages={'torch':'2.7.0+cu128'})
 def test_same_profile(self):assert_profile(self.profile,copy.deepcopy(self.profile))
 def test_changed_gpu_rejected(self):
  changed=copy.deepcopy(self.profile);changed['gpus']=['RTX 3090, 580.178.04, 24576']
  with self.assertRaisesRegex(RuntimeError,'Do not mix timing'):assert_profile(self.profile,changed)
 def test_changed_dependency_rejected(self):
  changed=copy.deepcopy(self.profile);changed['packages']['torch']='2.8.0+cu128'
  with self.assertRaisesRegex(RuntimeError,'packages'):assert_profile(self.profile,changed)
if __name__=='__main__':unittest.main()
