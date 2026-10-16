"""Integration: changed hardware must fail before any model is loaded."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_experiments as runner

class RunnerGuardIntegration(unittest.TestCase):
 def test_changed_hardware_stops_before_loading(self):
  original=runner.ROOT
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in ['data/manifest.json','configs/model_revisions.json','results/environment.json','results/hardware_profile.json']:
    target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((original/name).read_bytes())
   changed=json.loads((root/'results/hardware_profile.json').read_text());changed['gpus']=['different GPU']
   with patch.object(runner,'ROOT',root),patch.object(runner,'capture_profile',return_value=changed),patch.object(runner,'load_pipeline') as load,patch.object(sys,'argv',['run_experiments.py','--only','lcm_4step']):
    with self.assertRaisesRegex(RuntimeError,'Do not mix timing'):runner.main()
    load.assert_not_called()
   self.assertEqual((root/'results/environment.json').read_bytes(),(original/'results/environment.json').read_bytes())

if __name__=='__main__':unittest.main()
