"""Offline evaluation with cached features; generation is never included in this stage."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt
from transformers import CLIPModel, CLIPProcessor
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torchmetrics.image.kid import poly_mmd
from run_experiments import ROOT, configs, sha, atomic_json
os.environ.setdefault('TORCH_HOME',str(ROOT/'models/torch'))

torch.set_num_threads(8); cv2.setNumThreads(1)

def edge_metrics(image,reference,condition,tolerance=2):
 gray=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
 pred=cv2.Canny(cv2.GaussianBlur(gray,(3,3),1),100,200,apertureSize=3,L2gradient=True)>0
 ref=np.asarray(reference)>0; inp=np.asarray(condition)>0
 if not pred.any() or not ref.any():
  return dict(edge_f1=0.,retained_edge_recall=0.,edge_distance=None)
 d_ref=distance_transform_edt(~ref); d_pred=distance_transform_edt(~pred)
 precision=float((d_ref[pred]<=tolerance).mean()); recall=float((d_pred[ref]<=tolerance).mean())
 return dict(edge_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,retained_edge_recall=float((d_pred[inp]<=tolerance).mean()) if inp.any() else None,edge_distance=float((d_ref[pred].mean()+d_pred[ref].mean())/(2*np.hypot(*pred.shape))))

def bootstrap(values):
 values=np.asarray(values,dtype=float); rng=np.random.default_rng(2026)
 sims=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
 return [float(x) for x in np.quantile(sims,[.025,.975])]

def kid(real,fake):
 # One generated image per independent original per seed. Report seeds separately.
 gen=torch.Generator().manual_seed(2026); values=[]
 for _ in range(100):
  ri=torch.randperm(len(real),generator=gen)[:50]; fi=torch.randperm(len(fake),generator=gen)[:50]
  values.append(float(poly_mmd(real[ri].double(),fake[fi].double())))
 return dict(mean=float(np.mean(values)),subset_std=float(np.std(values)),subsets=100,subset_size=50)

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--only',default=''); args=parser.parse_args()
 rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text())['rows'] if r['split']=='test']; mapping={r['image_id']:r for r in rows}
 cache=ROOT/'results/features'; cache.mkdir(parents=True,exist_ok=True)
 clip=CLIPModel.from_pretrained(str(ROOT/'models/clip'),local_files_only=True).eval().to('cuda')
 processor=CLIPProcessor.from_pretrained(str(ROOT/'models/clip'),local_files_only=True)
 inception=FeatureExtractorInceptionV3('inception-v3-compat',['2048']).eval().to('cuda')
 feature_protocol=dict(clip_revision=json.loads((ROOT/'configs/model_revisions.json').read_text())['clip'][1],inception_sha256=sha(ROOT/'models/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'),protocol='uint8-inception2048-clip-b32-v1')
 def extract(paths,prompts,cache_path):
  fingerprint=hashlib.sha256(json.dumps(dict(files=[(str(p.relative_to(ROOT)),sha(p)) for p in paths],prompts=prompts,protocol=feature_protocol),sort_keys=True).encode()).hexdigest()
  if cache_path.exists():
   saved=torch.load(cache_path,map_location='cpu',weights_only=True)
   if saved['fingerprint']==fingerprint: return saved
  features=[]; scores=[]
  for offset in range(0,len(paths),8):
   images=[Image.open(p).convert('RGB') for p in paths[offset:offset+8]]
   pixels=torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2).to('cuda')
   with torch.inference_mode():
    features.append(inception(pixels)[0].cpu())
    if prompts:
     inputs=processor(images=images,text=prompts[offset:offset+8],return_tensors='pt',padding=True,truncation=True,max_length=77).to('cuda')
     result=clip(**inputs)
     scores.extend((100*(result.image_embeds*result.text_embeds).sum(-1).clamp(min=0)).cpu().tolist())
   if offset%40==0: print('FEATURES',cache_path.name,offset,'/',len(paths),flush=True)
  saved=dict(fingerprint=fingerprint,inception=torch.cat(features),clip=scores)
  torch.save(saved,cache_path); return saved
 real={}
 for res in [512,1024]: real[res]=extract([ROOT/r[f'reference_{res}'] for r in rows],None,cache/f'real_{res}.pt')['inception']
 for c in configs():
  if args.only and c['name'] not in args.only.split(','): continue
  directory=ROOT/'results/runs'/c['name']; receipts=[]
  for row in rows:
   for si in range(2):
    p=directory/f"{row['image_id']:012d}_s{si}.json"
    if not p.exists(): raise RuntimeError(f'Incomplete experiment: missing {p}')
    receipt=json.loads(p.read_text()); assert receipt['sha256']==sha(ROOT/receipt['path']); receipts.append(receipt)
  features=extract([ROOT/r['path'] for r in receipts],[r['prompt'] for r in receipts],cache/f"{c['name']}.pt")
  individual=[]
  for i,receipt in enumerate(receipts):
   row=mapping[receipt['image_id']]
   value=dict(image_id=row['image_id'],seed_index=receipt['seed_index'],clip_score=features['clip'][i])
   if c['resolution']==512:
    condition=c['condition'] if c['condition']!='none' else 'clean'
    image=Image.open(ROOT/receipt['path']); name=f"{row['image_id']:012d}.png"
    value.update(edge_metrics(image,Image.open(ROOT/'data/edges/clean'/name),Image.open(ROOT/'data/edges'/condition/name)))
   individual.append(value)
  lat=np.array([r['latency_s'] for r in receipts]); peak=np.array([r['peak_allocated_mib'] for r in receipts])
  kids=[kid(real[c['resolution']],features['inception'][si::2]) for si in range(2)]
  summary=dict(config=c,n_originals=100,n_generated=200,kid_mean=float(np.mean([k['mean'] for k in kids])),kid_by_seed=kids,latency_mean_s=float(lat.mean()),latency_median_s=float(np.median(lat)),latency_p95_s=float(np.quantile(lat,.95)),throughput_images_per_s=float(len(lat)/lat.sum()),peak_allocated_mib_max=float(peak.max()),peak_allocated_mib_mean=float(peak.mean()),metric_protocol='v1: CLIP ViT-B/32 100*max(0,cos); KID Inception2048 polynomial degree3 gamma1/2048 coef1; 100 subsets of 50 per seed; KID raw units; timing excludes I/O/loading/metrics; 3 warmups per config')
  for key in ['clip_score','edge_f1','retained_edge_recall','edge_distance']:
   if key not in individual[0]: continue
   vals=[]
   for i in range(0,len(individual),2):
    pair=[v[key] for v in individual[i:i+2] if v[key] is not None]
    if pair: vals.append(float(np.mean(pair)))
   summary[key+'_mean']=float(np.mean(vals)); summary[key+'_ci95']=bootstrap(vals); summary[key+'_n_originals']=len(vals)
  atomic_json(ROOT/'results/metrics'/f"{c['name']}.json",dict(summary=summary,per_image=individual))
  print('EVALUATED',c['name'],json.dumps(summary),flush=True)

if __name__=='__main__': main()
