"""Freeze a NEW reproducible COCO val2017 subset; never claim original subset recovery."""
from pathlib import Path
import json, hashlib, random, zipfile
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image, ImageOps
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 archive=DATA/'annotations_trainval2017.zip'
 assert hashlib.md5(archive.read_bytes()).hexdigest()=='f4bbac642086de4f52a3fdda2de5fa2c', 'COCO annotation archive checksum mismatch'
 with zipfile.ZipFile(archive) as z:
  raw=z.read('annotations/captions_val2017.json')
 (DATA/'captions_val2017.json').write_bytes(raw)
 ann=json.loads(raw)
 captions={}
 for a in sorted(ann['annotations'],key=lambda x:x['id']): captions.setdefault(a['image_id'],[]).append(a)
 images=sorted(ann['images'],key=lambda x:x['id'])
 random.Random(2026).shuffle(images)
 chosen=images[:120]
 for d in ['originals','reference_512','reference_1024','edges/clean','edges/sparse','edges/dense','edges/erase25','edges/erase50']:
  (DATA/d).mkdir(parents=True,exist_ok=True)
 def fetch(im):
  path=DATA/'originals'/im['file_name']
  if not path.exists():
   url='http://images.cocodataset.org/val2017/'+im['file_name']
   r=requests.get(url,timeout=90); r.raise_for_status()
   Image.open(__import__('io').BytesIO(r.content)).verify()
   path.write_bytes(r.content)
  return path
 with ThreadPoolExecutor(max_workers=8) as pool: paths=list(pool.map(fetch,chosen))
 rows=[]
 for i,(im,path) in enumerate(zip(chosen,paths)):
  image=Image.open(path).convert('RGB'); ident=im['id']; name=f'{ident:012d}.png'
  for res in [512,1024]:
   ImageOps.fit(image,(res,res),method=Image.Resampling.LANCZOS).save(DATA/f'reference_{res}'/name)
  gray=cv2.cvtColor(np.array(Image.open(DATA/'reference_512'/name)),cv2.COLOR_RGB2GRAY)
  gray=cv2.GaussianBlur(gray,(3,3),1)
  edges={key:cv2.Canny(gray,lo,hi,apertureSize=3,L2gradient=True) for key,(lo,hi) in {'clean':(100,200),'sparse':(150,250),'dense':(50,100)}.items()}
  # Nested rectangles, same deterministic center/aspect per image, calibrated by edge count.
  rng=np.random.default_rng(2026+ident); cx,cy=rng.uniform(.3,.7,2)*512; aspect=float(rng.uniform(.6,1.6))
  yy,xx=np.mgrid[:512,:512]; dist=np.maximum(abs(xx-cx)/np.sqrt(aspect),abs(yy-cy)*np.sqrt(aspect))
  nonzero=dist[edges['clean']>0]; erase={}
  for percent in [25,50]:
   radius=float(np.quantile(nonzero,percent/100)) if len(nonzero) else 0
   mask=dist<=radius
   out=edges['clean'].copy(); out[mask]=0; edges[f'erase{percent}']=out
   erase[str(percent)]=float(((edges['clean']>0)&mask).sum()/max(1,(edges['clean']>0).sum()))
  for key,edge in edges.items(): Image.fromarray(edge).save(DATA/'edges'/key/name)
  cap=captions[ident][0]
  rows.append(dict(image_id=ident,split='dev' if i<20 else 'test',caption=cap['caption'],caption_id=cap['id'],original=str(path.relative_to(ROOT)),original_sha256=sha(path),reference_512=f'data/reference_512/{name}',reference_1024=f'data/reference_1024/{name}',reference_512_sha256=sha(DATA/'reference_512'/name),reference_1024_sha256=sha(DATA/'reference_1024'/name),seeds=[2026000+ident*2,2026001+ident*2],erased_edge_fraction=erase))
 manifest=dict(subset_id='coco-val2017-new-seed2026-v1',original_report_subset_recovered=False,sampling='Python random.Random(2026).shuffle(sorted(images, key=id)); first 20 dev, next 100 test; lowest annotation id caption; no output-dependent exclusions',annotations_md5=hashlib.md5(archive.read_bytes()).hexdigest(),annotations_sha256=sha(DATA/'captions_val2017.json'),preprocessing='PIL ImageOps.fit center square, Lanczos; same spatial crop at 512 and 1024; captions not manually screened',rows=rows)
 target=DATA/'manifest.json'
 if target.exists(): assert json.loads(target.read_text())==manifest, 'Frozen manifest would change'
 target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print('DATA READY:',len(rows),'images; manifest SHA256',sha(target),flush=True)

if __name__=='__main__': main()
