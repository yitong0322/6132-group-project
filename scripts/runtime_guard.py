"""Detect major hardware/software changes before appending inference timings."""
import os, platform, subprocess
from importlib.metadata import version
from pathlib import Path

PACKAGES=['torch','torchvision','diffusers','transformers','peft','accelerate','numpy','pillow','opencv-python','torchmetrics','torch-fidelity','scipy']

def capture_profile():
 gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 cpu='unknown'
 if Path('/proc/cpuinfo').exists():
  for line in Path('/proc/cpuinfo').read_text().splitlines():
   if line.startswith('model name'):
    cpu=line.split(':',1)[1].strip();break
 return dict(gpus=[line.strip() for line in gpu_lines],cpu=cpu,python_major_minor='.'.join(platform.python_version_tuple()[:2]),os=platform.system(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),packages={p:version(p) for p in PACKAGES})

def assert_profile(saved,current):
 differences=[key for key in saved if saved[key]!=current.get(key)]
 if differences:
  raise RuntimeError('Hardware/software differs from the frozen run ('+', '.join(differences)+'). Do not mix timing measurements. Use a separate project/results directory for the new environment; keep this completed run unchanged.')
