"""Resumable inference. Each saved image has a configuration and timing receipt."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, gc, hashlib, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetPipeline,
 StableDiffusionAdapterPipeline, ControlNetModel, T2IAdapter, UNet2DConditionModel,
 LCMScheduler, EulerDiscreteScheduler, DDIMScheduler)
from safetensors.torch import load_file
from runtime_guard import capture_profile, assert_profile

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)); tmp.replace(p)

def configs():
 out=[]
 for method in ['lcm','lightning']:
  for steps in [4,2,8]:
   out.append(dict(name=f'{method}_{steps}step',method=method,steps=steps,resolution=1024,guidance=0.0,condition='none'))
 out.append(dict(name='sdxl_30step',method='sdxl',steps=30,resolution=1024,guidance=5.0,condition='none'))
 for method in ['controlnet','adapter']:
  for condition in ['clean','erase25','erase50','sparse','dense']:
   out.append(dict(name=f'{method}_{condition}',method=method,steps=30,resolution=512,guidance=7.5,condition=condition,strength=1.5))
 out.append(dict(name='sd15_30step',method='sd15',steps=30,resolution=512,guidance=7.5,condition='none'))
 return out

def load_pipeline(c):
 method=c['method']; kw=dict(torch_dtype=torch.float16,variant='fp16',local_files_only=True)
 if method in ['lcm','lightning','sdxl']:
  if method=='lightning':
   unet=UNet2DConditionModel.from_config(str(MODEL/'sdxl'/'unet'))
   unet.load_state_dict(load_file(str(MODEL/'lightning'/f"sdxl_lightning_{c['steps']}step_unet.safetensors")))
   kw['unet']=unet.to(dtype=torch.float16)
  pipe=StableDiffusionXLPipeline.from_pretrained(str(MODEL/'sdxl'),add_watermarker=False,**kw)
  if method=='lcm':
   pipe.load_lora_weights(str(MODEL/'lcm'),weight_name='pytorch_lora_weights.safetensors')
   pipe.fuse_lora(); pipe.unload_lora_weights()
   pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
  else: pipe.scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config,**({'timestep_spacing':'trailing'} if method=='lightning' else {}))
 else:
  kw.update(safety_checker=None,feature_extractor=None,requires_safety_checker=False)
  if method=='controlnet':
   control=ControlNetModel.from_pretrained(str(MODEL/'controlnet'),**{k:v for k,v in kw.items() if k in ['torch_dtype','variant','local_files_only']})
   pipe=StableDiffusionControlNetPipeline.from_pretrained(str(MODEL/'sd15'),controlnet=control,**kw)
  elif method=='adapter':
   adapter=T2IAdapter.from_pretrained(str(MODEL/'adapter'),torch_dtype=torch.float16,local_files_only=True)
   assert adapter.config.in_channels==1
   pipe=StableDiffusionAdapterPipeline.from_pretrained(str(MODEL/'sd15'),adapter=adapter,**kw)
  else: pipe=StableDiffusionPipeline.from_pretrained(str(MODEL/'sd15'),**kw)
  pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
 pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
 return pipe

def inputs(c,row,seed):
 # CPU-created unit Gaussian latent shared across methods within each resolution family.
 gen=torch.Generator(device='cpu').manual_seed(seed)
 latent=torch.randn((1,4,c['resolution']//8,c['resolution']//8),generator=gen,dtype=torch.float32).to('cuda',dtype=torch.float16)
 kwargs=dict(prompt=row['caption'],height=c['resolution'],width=c['resolution'],num_inference_steps=c['steps'],guidance_scale=c['guidance'],negative_prompt='',latents=latent,generator=gen)
 if c['condition']!='none':
  edge=Image.open(ROOT/'data/edges'/c['condition']/f"{row['image_id']:012d}.png")
  kwargs['image']=edge.convert('RGB' if c['method']=='controlnet' else 'L')
  kwargs['controlnet_conditioning_scale' if c['method']=='controlnet' else 'adapter_conditioning_scale']=c['strength']
 return kwargs

def generate(pipe,c,row,seed):
 kwargs=inputs(c,row,seed) # Latent preparation and input disk I/O excluded.
 torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter()
 with torch.inference_mode(): image=pipe(**kwargs).images[0]
 torch.cuda.synchronize(); elapsed=time.perf_counter()-start
 return image,dict(latency_s=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--only',default=''); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--limit',type=int)
 args=parser.parse_args()
 manifest=json.loads((ROOT/'data/manifest.json').read_text()); dev=[r for r in manifest['rows'] if r['split']=='dev']; test=[r for r in manifest['rows'] if r['split']=='test']
 rows=dev[:2] if args.smoke else test
 if args.limit: rows=rows[:args.limit]
 base=ROOT/'results'/('smoke' if args.smoke else 'runs')
 atomic_json(ROOT/'configs/experiment_matrix.json',configs())
 environment_path=ROOT/'results/environment.json'
 profile_path=ROOT/'results/hardware_profile.json'
 if not environment_path.exists():
  if any((ROOT/'results/runs').glob('*/*.json')):
   raise RuntimeError('Saved outputs exist but original environment metadata is missing. Restore it before resuming.')
  atomic_json(environment_path,dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),platform=platform.platform(),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),source_sha256=sha(Path(__file__)),manifest_sha256=sha(ROOT/'data/manifest.json')))
 current_profile=None
 previous_key=None; pipe=None
 for c in configs():
  if args.only and c['name'] not in args.only.split(','): continue
  fingerprint=hashlib.sha256(json.dumps(dict(config=c,manifest=sha(ROOT/'data/manifest.json'),models=json.loads((ROOT/'configs/model_revisions.json').read_text()),protocol='v1'),sort_keys=True).encode()).hexdigest()
  out=base/c['name']; out.mkdir(parents=True,exist_ok=True)
  pending=[]
  for row in rows:
   for si,seed in enumerate(row['seeds'][:1] if args.smoke else row['seeds']):
    path=out/f"{row['image_id']:012d}_s{si}.png"; receipt=path.with_suffix('.json')
    if receipt.exists():
     saved=json.loads(receipt.read_text())
     if saved['config_hash']==fingerprint and path.exists() and sha(path)==saved['sha256']: continue
     raise RuntimeError(f'Invalid saved output, investigate before resuming: {path}')
    pending.append((row,si,seed,path,receipt))
  if not pending: print('SKIP COMPLETE',c['name'],flush=True); continue
  if current_profile is None: current_profile=capture_profile()
  if profile_path.exists():
   assert_profile(json.loads(profile_path.read_text()),current_profile)
  else:
   if any((ROOT/'results/runs').glob('*/*.json')):
    raise RuntimeError('Saved outputs exist but frozen hardware profile is missing. Restore it before resuming.')
   atomic_json(profile_path,current_profile)
  key=(c['method'], c['steps'] if c['method']=='lightning' else 0)
  if key!=previous_key:
   if pipe is not None: del pipe
   gc.collect(); torch.cuda.empty_cache(); pipe=load_pipeline(c); previous_key=key
  print('START',c['name'],'remaining',len(pending),flush=True)
  atomic_json(out/'pipeline_metadata.meta',dict(scheduler_class=type(pipe.scheduler).__name__,scheduler_config=dict(pipe.scheduler.config),source_sha256=sha(Path(__file__)),manifest_sha256=sha(ROOT/'data/manifest.json'),config_hash=fingerprint))
  for w in range(3): generate(pipe,c,dev[w],dev[w]['seeds'][0])
  for idx,(row,si,seed,path,receipt) in enumerate(pending):
   image,timing=generate(pipe,c,row,seed)
   arr=np.asarray(image)
   if arr.std()<1: raise RuntimeError(f'Near-constant output: {path}')
   tmp=path.with_suffix('.tmp.png'); image.save(tmp); tmp.replace(path)
   atomic_json(receipt,dict(config=c,config_hash=fingerprint,image_id=row['image_id'],seed_index=si,seed=seed,prompt=row['caption'],path=str(path.relative_to(ROOT)),sha256=sha(path),**timing))
   if idx%10==0 or idx==len(pending)-1: print(c['name'],idx+1,'/',len(pending),f"{timing['latency_s']:.3f}s",f"{timing['peak_allocated_mib']:.0f}MiB",flush=True)
  print('COMPLETE',c['name'],flush=True)

if __name__=='__main__': main()
