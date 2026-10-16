"""Check cached-feature KID aggregation against the public TorchMetrics class."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCH_HOME']=str(ROOT/'models/torch')
import torch
from torchmetrics.image.kid import KernelInceptionDistance

def main():
 torch.set_num_threads(4)
 real=torch.load(ROOT/'results/features/real_1024.pt',weights_only=True)['inception'].double()
 fake=torch.load(ROOT/'results/features/lcm_4step.pt',weights_only=True)['inception'][::2].double()
 assert real.shape==fake.shape==(100,2048) and torch.isfinite(real).all() and torch.isfinite(fake).all()
 assert not torch.equal(real,fake)
 metric=KernelInceptionDistance(subsets=100,subset_size=50)
 metric.real_features=[real];metric.fake_features=[fake]
 torch.manual_seed(2026)
 mean,std=metric.compute()
 expected=json.loads((ROOT/'results/metrics/lcm_4step.json').read_text())['summary']['kid_by_seed'][0]
 assert abs(float(mean)-expected['mean'])<1e-12 and abs(float(std)-expected['subset_std'])<1e-12
 result=dict(status='PASS',scope='KID cached-feature aggregation versus official TorchMetrics class; first seed of LCM 4-step',mean=float(mean),std=float(std),real_feature_mean_norm=float(real.norm(dim=1).mean()),fake_feature_mean_norm=float(fake.norm(dim=1).mean()),max_difference=max(abs(float(mean)-expected['mean']),abs(float(std)-expected['subset_std'])))
 (ROOT/'results/metric_consistency.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
