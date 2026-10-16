"""Render only measured metrics; absent results remain pending."""
import json
import textwrap
from datetime import datetime,timezone
from run_experiments import ROOT, configs, atomic_json, sha

def main():
 manifest=json.loads((ROOT/'data/manifest.json').read_text())
 environment=json.loads((ROOT/'results/environment.json').read_text())
 profile=json.loads((ROOT/'results/hardware_profile.json').read_text()) if (ROOT/'results/hardware_profile.json').exists() else {}
 stages=json.loads((ROOT/'results/stage_receipts.json').read_text()) if (ROOT/'results/stage_receipts.json').exists() else []
 generation=next((s for s in stages if s['stage']=='generation'),None)
 lines=['# 本机实验结果与交接状态','', '## Material Passport','',
 '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
 f'- Origin Date: {datetime.now(timezone.utc).isoformat()}',
 '- Verification Status: UNVERIFIED（未做全量重新生成复现；本机实际输出的文件完整性验收见 results/verification.json）',
 '- Version Label: local-coco-new-v1','',
 '## 数据与实验范围','',
 f"子集：`{manifest['subset_id']}`；20张开发、100张测试；每张测试图两个种子。原始报告子集未找回，本次重新抽取，全部方法共享本次固定ID和captions。",
 f"清单SHA256：`{sha(ROOT/'data/manifest.json')}`。完整ID、文本、种子与原图路径见 `data/manifest.json`。",
 f"设备：{environment['gpu']}；CPU：{profile.get('cpu','见环境记录')}；PyTorch {environment['torch']}，CUDA runtime {environment['cuda']}。完整版本锁见requirements-lock.txt。",
 (f"生成阶段记录耗时：{(generation['end']-generation['start'])/3600:.2f}小时，含模型加载、预热和图像保存；与表中排除这些开销的单次Latency口径不同。" if generation else '生成阶段仍在进行；最终耗时将在阶段结束后填入。'),
 '', '详细协议、指标公式、运行/续跑和交接方式见 [README.md](README.md)。以下数字只来自本次结果文件，不使用历史报告数字填空。','',
 '## 已测试指标与进度','',
 '| 配置 | 正式图像/200 | 已计算指标 |','|---|---:|---|']
 summaries=[]; total=0
 for c in configs():
  count=len(list((ROOT/'results/runs'/c['name']).glob('*.json'))); total+=count
  p=ROOT/'results/metrics'/f"{c['name']}.json"
  if p.exists():
   summaries.append(json.loads(p.read_text())['summary']); state='KID、CLIPScore、Latency、Throughput、Peak VRAM'
   if c['resolution']==512: state+='、Edge F1、保留边缘召回率、边缘距离'
  else: state='待评测；有逐图收据时仅说明生成耗时/显存已记录'
  lines.append(f"| {c['name']} | {count} | {state} |")
 complete=total==3600 and len(summaries)==18
 if complete:
  from PIL import Image,ImageDraw
  test=[r for r in manifest['rows'] if r['split']=='test'][:3]
  for group,names in [('sdxl',['lcm_2step','lcm_4step','lcm_8step','lightning_2step','lightning_4step','lightning_8step','sdxl_30step']),('sd15',['reference','edge','controlnet_clean','adapter_clean','sd15_30step'])]:
   canvas=Image.new('RGB',(len(names)*240,3*300),'white'); draw=ImageDraw.Draw(canvas)
   for ri,row in enumerate(test):
    ident=row['image_id']; y=ri*300
    draw.text((5,y+4),f"COCO {ident}: {row['caption']}",fill='black')
    for ci,name in enumerate(names):
     if name=='reference': path=ROOT/row['reference_512']
     elif name=='edge': path=ROOT/'data/edges/clean'/f'{ident:012d}.png'
     else:path=ROOT/'results/runs'/name/f'{ident:012d}_s0.png'
     draw.text((ci*240+5,y+24),name,fill='black')
     canvas.paste(Image.open(path).convert('RGB').resize((232,232)),(ci*240+4,y+48))
   canvas.save(ROOT/'results'/f'gallery_{group}.jpg',quality=90)
 lines+=['',f"正式输出：**{total}/3600**；完成指标汇总：**{len(summaries)}/18**；状态：**{'已完成生成与自动评测' if complete else '进行中/尚未完整'}**。",'',
 '## 自动指标结果','',
 'KID为原始单位，无偏估计可能为负，保留负值而不截断；先分别计算两个生成种子，再取均值；CLIPScore按图像平均。峰值显存为200次调用的最大allocated值。吞吐量为排除外部I/O的batch=1串行值。', '',
 '| 配置 | KID ↓ | CLIPScore ↑ | 平均延迟 s ↓ | 吞吐 张/s ↑ | 峰值 MiB ↓ |',
 '|---|---:|---:|---:|---:|---:|']
 for s in summaries:
  lines.append(f"| {s['config']['name']} | {s['kid_mean']:.6f} | {s['clip_score_mean']:.3f} | {s['latency_mean_s']:.4f} | {s['throughput_images_per_s']:.4f} | {s['peak_allocated_mib_max']:.1f} |")
 lines+=['','## 旧方法结构指标','', '| 配置 | Edge F1 ↑ | 保留边缘召回率 ↑ | 边缘距离 ↓ |','|---|---:|---:|---:|']
 for s in summaries:
  if 'edge_f1_mean' in s: lines.append(f"| {s['config']['name']} | {s['edge_f1_mean']:.4f} | {s['retained_edge_recall_mean']:.4f} | {s['edge_distance_mean']:.6f} |")
 lines+=['','## 解释与未执行项目','',
 '- 1024²的SDXL纯文本组与512²的SD1.5边缘控制组分别分析，不能用跨组排名声称某加速算法因果优于另一种控制架构。',
 '- KID每种子100张，100次50张子集的标准差是子集波动，不是置信区间；小样本结论需谨慎。完整逐图分数、CLIP/边缘bootstrap区间、各种子KID见results/metrics。',
 '- 当前不包含真实人工评分；没有伪造或根据模型自动推断人工评分。',
 '- 历史报告的强度扫描、尺度消融、控制时段消融没有在本次重跑，不能标记为本机已复现。',
 '- 原图中心裁剪可能裁掉caption对象；本次没有逐图人工语义筛查。',
 '- 模型版本固定；CUDA跨硬件/版本不保证逐像素一致。',
 '- 各检查点预训练数据不同，本次不能保证COCO验证图像从未出现在模型预训练数据中。',
 '', '## 交给下一位同学','',
 '1. 先读README.md和本报告，检查results/verification.json的状态。',
 '2. 使用data/manifest.json，不重新随机划分；原图、参考图、边缘条件均已包含。',
 '3. 建立新环境并安装requirements-lock.txt；运行scripts/download_models.py下载固定revision权重，再运行scripts/verify_models.py核验。',
 '4. run_experiments.py自动跳过配置与哈希均有效的已完成项；evaluate.py复用特征缓存；最后build_report.py重新汇总。',
 '5. 续跑保护会检查硬件和依赖版本；换机器或环境测量必须使用独立实验目录，避免混入本次效率统计。新增实验请保留旧结果。人工评价和旧消融属于后续任务。','',
 '## 验证记录','',
 '- results/verification.json：3600张输出、18组指标、数据清单与处理结果、配置和SHA256完整性验收通过；没有全量重新生成第二遍。',
 '- results/model_verification.json：固定revision模型权重的官方LFS SHA256核验通过。',
 '- 协议测试5项、运行环境保护测试3项、续跑保护集成测试1项通过；完整配置跳过测试通过。',
 '- results/metric_consistency.json：缓存特征上的LCM 4步种子0 KID与TorchMetrics实现一致，最大绝对差约4.34e-19；此项是公式与聚合一致性核对。',
 '- results/handoff_code_verification.json：交接代码与实际执行源码的四个核心函数AST一致；新增部分是续跑环境保护。','']
 if complete:
  lines+=['## 固定样本预览','', '按冻结测试序列前3张、种子索引0展示，没有按结果筛选。SDXL组只接收文本；SD1.5控制方法额外接收边缘。预览不代替总体指标或人工评价。','', '![SDXL组](results/gallery_sdxl.jpg)','', '![SD1.5 clean组](results/gallery_sd15.jpg)','']
 (ROOT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines))
 atomic_json(ROOT/'results/status.json',dict(generated=total,expected=3600,evaluated=len(summaries),expected_configurations=18,complete=complete))
 print('REPORT',total,'images',len(summaries),'metric groups')

if __name__=='__main__': main()
