"""Verify local weight bytes against official pinned HF LFS SHA256 metadata."""
import json, hashlib
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[1]

def main():
 config=json.loads((ROOT/'configs/model_revisions.json').read_text()); results=[]
 for key,(repo,revision,_) in config.items():
  info=HfApi().model_info(repo,revision=revision,files_metadata=True)
  for entry in info.siblings:
   p=ROOT/'models'/key/entry.rfilename
   if not p.is_file() or not entry.lfs: continue
   hasher=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''): hasher.update(chunk)
   digest=hasher.hexdigest(); expected=entry.lfs.sha256
   assert digest==expected, str(p)
   results.append(dict(path=str(p.relative_to(ROOT)),repo=repo,revision=revision,sha256=digest,bytes=p.stat().st_size,official_sha256=expected))
  print('VERIFIED',key,flush=True)
 (ROOT/'results/model_verification.json').write_text(json.dumps(dict(status='PASS',weights=results),indent=2))

if __name__=='__main__': main()
