"""Package a verified completed experiment without environments or model weights."""
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 assert json.loads((ROOT/'results/verification.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'results/status.json').read_text())['complete']
 exclude={'.venv','models','__pycache__','__MACOSX','.git'}
 files=[]
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or any(part in exclude for part in rel.parts):continue
  if p.name in ['report.zip','annotations_trainval2017.zip','.DS_Store','handoff_checksums.json','package_receipt.json','active_process.json']:continue
  files.append(p)
 checksums={str(p.relative_to(ROOT)):digest(p) for p in files}
 checkfile=ROOT/'handoff_checksums.json'; checkfile.write_text(json.dumps(checksums,ensure_ascii=False,indent=2)); files.append(checkfile)
 target=ROOT.parent/f"6132_handoff_{datetime.now():%Y%m%d}.zip"
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
  for p in files:z.write(p,arcname='6132/'+str(p.relative_to(ROOT)))
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  for name,expected in checksums.items():assert hashlib.sha256(z.read('6132/'+name)).hexdigest()==expected,name
 result=dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size,files=len(files),zip_crc_and_member_sha256='PASS',excluded=['models/','.venv/','duplicate original zip','COCO annotation archive zip'])
 target.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+target.name+'\n')
 (ROOT/'results/package_receipt.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))

if __name__=='__main__':main()
