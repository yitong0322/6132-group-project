"""Lightweight status display, no torch imports or GPU work."""
from pathlib import Path
import json,os,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 counts={p.name:len(list(p.glob('*.png'))) for p in sorted((ROOT/'results/runs').glob('*')) if p.is_dir()}
 state=json.loads((ROOT/'results/active_process.json').read_text()) if (ROOT/'results/active_process.json').exists() else {}
 alive=False
 if state.get('pid'):
  try:os.kill(state['pid'],0);alive=True
  except ProcessLookupError:pass
 status=dict(stage=state.get('stage'),process_alive=alive,generated=sum(counts.values()),expected=3600,counts=counts,metrics=len(list((ROOT/'results/metrics').glob('*.json'))))
 print(json.dumps({k:v for k,v in status.items() if k!='counts'}|dict(completed_configurations=sum(v==200 for v in counts.values()),unfinished={k:v for k,v in counts.items() if v<200}),ensure_ascii=False))
 text=['# 实验实时进度','',f'更新时间：{time.ctime()}',f"当前阶段：{status['stage']}；子进程仍在运行：{alive}",f"正式生成：{status['generated']} / 3600；指标完成：{status['metrics']} / 18",'', '| 配置 | 已生成 |','|---|---:|']+[f'| {k} | {v}/200 |' for k,v in counts.items()]
 text+=['','这是进度快照，不是完成证明。最终结果见EXPERIMENT_REPORT.md及results/verification.json。','']
 (ROOT/'LIVE_STATUS.md').write_text('\n'.join(text))
if __name__=='__main__':main()
