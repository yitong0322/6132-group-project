"""Sequential GPU workflow with stage receipts; failed stages stop visibly."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
stages=[
 ('tests',['scripts/test_protocol.py']),
 ('smoke_lightning',['scripts/run_experiments.py','--smoke','--only','lightning_4step']),
 ('generation',['scripts/run_experiments.py']),
 ('evaluation',['scripts/evaluate.py']),
 ('verification',['scripts/verify_project.py']),
 ('report',['scripts/build_report.py']),
 ('package',['scripts/package_project.py']),
]

def main():
 receipts=[]
 for name,args in stages:
  start=time.time(); print('STAGE_START',name,flush=True)
  log=ROOT/'logs'/f'{name}.log'
  with log.open('a') as f:
   f.write(f'\nRUN START {time.ctime()}\n'); f.flush()
   process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   (ROOT/'results/active_process.json').write_text(json.dumps(dict(stage=name,pid=process.pid,start=start,log=str(log.relative_to(ROOT)))))
   code=process.wait()
  receipts.append(dict(stage=name,start=start,end=time.time(),exit_code=code,log=str(log.relative_to(ROOT))))
  (ROOT/'results/stage_receipts.json').write_text(json.dumps(receipts,indent=2))
  print('STAGE_END',name,'exit',code,'elapsed',time.time()-start,flush=True)
  if code:sys.exit(code)
 (ROOT/'results/active_process.json').write_text(json.dumps(dict(stage='complete',finished=time.time())))

if __name__=='__main__':main()
