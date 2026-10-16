"""Read-only integrity acceptance, requiring all 18 configurations and five core metrics."""
import json, math, random
from PIL import Image,ImageOps
import numpy as np
import cv2
from run_experiments import ROOT, configs, sha, atomic_json

def main():
 assert json.loads((ROOT/'results/model_verification.json').read_text())['status']=='PASS'
 for path,expected in json.loads((ROOT/'data/files_sha256.json').read_text()).items():
  assert sha(ROOT/path)==expected, path
 manifest=json.loads((ROOT/'data/manifest.json').read_text()); rows=manifest['rows']; test=[r for r in rows if r['split']=='test']; dev=[r for r in rows if r['split']=='dev']
 assert len(test)==100 and len(dev)==20 and len({r['image_id'] for r in rows})==120
 annotations=json.loads((ROOT/'data/captions_val2017.json').read_text())
 candidates=sorted(annotations['images'],key=lambda x:x['id']); random.Random(2026).shuffle(candidates)
 assert [r['image_id'] for r in rows]==[r['id'] for r in candidates[:120]]
 captions={}
 for a in sorted(annotations['annotations'],key=lambda x:x['id']):captions.setdefault(a['image_id'],a)
 for r in rows:
  assert captions[r['image_id']]['id']==r['caption_id'] and captions[r['image_id']]['caption']==r['caption']
  for key in ['original','reference_512','reference_1024']:
   assert sha(ROOT/r[key])==r[key+'_sha256']
  original=Image.open(ROOT/r['original']).convert('RGB')
  for res in [512,1024]:
   assert np.array_equal(np.array(ImageOps.fit(original,(res,res),method=Image.Resampling.LANCZOS)),np.array(Image.open(ROOT/r[f'reference_{res}'])))
  edges={}
  for condition in ['clean','erase25','erase50','sparse','dense']:
   p=ROOT/'data/edges'/condition/f"{r['image_id']:012d}.png"
   with Image.open(p) as im:
    assert im.size==(512,512); edges[condition]=np.array(im)>0
  assert np.all(~edges['erase50']|edges['erase25']) and np.all(~edges['erase25']|edges['clean'])
  gray=cv2.GaussianBlur(cv2.cvtColor(np.array(Image.open(ROOT/r['reference_512'])),cv2.COLOR_RGB2GRAY),(3,3),1)
  for condition,(lo,hi) in {'clean':(100,200),'sparse':(150,250),'dense':(50,100)}.items():
   assert np.array_equal(edges[condition],cv2.Canny(gray,lo,hi,apertureSize=3,L2gradient=True)>0)
  for pct in [25,50]:
   actual=1-float(edges[f'erase{pct}'].sum())/max(1,edges['clean'].sum())
   assert abs(actual-r['erased_edge_fraction'][str(pct)])<1e-12
 checked=0
 for c in configs():
  out=ROOT/'results/runs'/c['name']; receipts=list(out.glob('*.json')); assert len(receipts)==200,(c['name'],len(receipts))
  assert len(list(out.glob('*.png')))==200
  for r in test:
   for si in range(2):
    p=out/f"{r['image_id']:012d}_s{si}.json"; v=json.loads(p.read_text())
    assert v['config']==c and v['prompt']==r['caption'] and v['seed']==r['seeds'][si]
    image=ROOT/v['path']; assert sha(image)==v['sha256']
    with Image.open(image) as im: assert im.size==(c['resolution'],c['resolution'])
    assert math.isfinite(v['latency_s']) and v['latency_s']>0 and v['peak_allocated_mib']>0
    checked+=1
  met=json.loads((ROOT/'results/metrics'/f"{c['name']}.json").read_text()); s=met['summary']
  assert len(met['per_image'])==200 and s['n_originals']==100
  for key in ['kid_mean','clip_score_mean','latency_mean_s','throughput_images_per_s','peak_allocated_mib_max']: assert math.isfinite(s[key])
  assert abs(s['latency_mean_s']*s['throughput_images_per_s']-1)<1e-10
 result=dict(status='PASS',images_verified=checked,configurations=18,unique_originals=120,manifest_sha256=sha(ROOT/'data/manifest.json'),scope='File integrity, coverage, finite metrics and protocol identities; not a full regeneration replication')
 atomic_json(ROOT/'results/verification.json',result); print(json.dumps(result))

if __name__=='__main__': main()
