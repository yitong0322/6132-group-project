# 图像生成实验与同学交接

本项目在本机 RTX 5060 Ti 16GB 上比较 LCM-LoRA（SDXL）与 SDXL-Lightning，并在同一新抽取 COCO 子集上重跑 ControlNet Canny、T2I-Adapter Canny，补齐共同指标。保留原生 SDXL 与 SD 1.5 基线。仅推理与评测，不训练或蒸馏模型。

**运行进度快照见 `LIVE_STATUS.md`；完成情况以 `results/metrics/*.json`、`results/verification.json` 和 `EXPERIMENT_REPORT.md` 为准。没有相应结果文件的指标不得称为已测试。** `report./` 与 `report.zip` 是收到的历史报告，不属于本次测量。不要将历史报告的 RTX3090 数据与本机结果混在一起。

## 数据子集

- 数据来源：COCO 2017 validation（val2017），官方图像及 `captions_val2017.json`。
- 子集 ID：`coco-val2017-new-seed2026-v1`。
- **这不是已经找回的历史子集。** 历史报告没有附120张图的完整ID、划分、caption或噪声清单，本机检索未找到，因此重新抽样；所有本次方法共享新清单。
- 抽样：将5000张验证图按图像ID升序排列，用 Python `random.Random(2026).shuffle` 打乱，前20张开发、随后100张测试。每图选 annotation ID 最小的原始 caption，无按模型输出挑选或排除。
- `data/manifest.json` 保存全部120个ID、split、caption及caption ID、原图路径和SHA256、处理图SHA256、两个固定种子。原图在 `data/originals/`，512与1024参考图分别在 `data/reference_512/`、`data/reference_1024/`。
- 可直接阅读的完整120行清单见 [docs/DATASET.md](docs/DATASET.md)。
- 中心裁剪使用 PIL `ImageOps.fit`、Lanczos；两种分辨率对应同一方形视野。未人工核查caption中的对象是否被裁掉，这是本次协议限制，不继承历史报告“人工审核4张替换”的声称。
- 测试集每图两个种子，共200张输出/配置；20张开发图不计入正式评测，前三张用于预热，前两张用于试跑。本次不重做开发集强度搜索。
- 完整标注归档MD5：`f4bbac642086de4f52a3fdda2de5fa2c`，已在准备脚本中校验。下载源 `http://images.cocodataset.org/`；HTTPS域名证书不匹配，故用官方HTTP地址，不关闭TLS验证。单张原图的SHA256是本次下载指纹，并非独立官方哈希认证。

## 实验矩阵

| 方法 | 基础模型 | 分辨率 | 步数/条件 | 正式输出 |
|---|---|---:|---|---:|
| LCM-LoRA | SDXL Base 1.0 | 1024² | 2/4/8步，LCMScheduler，guidance=0，融合LoRA | 600 |
| SDXL-Lightning | SDXL Base 1.0 | 1024² | 2/4/8步各自完整U-Net，Euler trailing，guidance=0 | 600 |
| 原生SDXL基线 | SDXL Base 1.0 | 1024² | Euler默认时间间隔，30步，CFG=5 | 200 |
| ControlNet Canny | SD 1.5 | 512² | DDIM 30步，CFG=7.5，强度1.5，五种条件 | 1000 |
| T2I-Adapter Canny | SD 1.5 | 512² | DDIM 30步，CFG=7.5，强度1.5，五种条件 | 1000 |
| 原生SD 1.5基线 | SD 1.5 | 512² | DDIM 30步，CFG=7.5 | 200 |
| 合计 | | | 18个配置 | 3600 |

统一FP16、batch size=1、空负提示、PyTorch默认SDPA、无编译、无CPU offload、一次仅一个生成系统驻留GPU。同一分辨率组使用同一CPU生成的FP32单位高斯初始噪声，转换FP16后传入各pipeline；不同采样器的噪声缩放及后续随机过程遵循各自实现。不同分辨率不声称拥有逐元素相同的噪声。

五种边缘条件：clean Canny100/200；sparse150/250；dense50/100；erase25/erase50为从clean删除约25%/50%边缘像素的嵌套矩形。灰度、3×3 Gaussian sigma=1、Canny aperture=3、L2gradient=True。矩形中心和长宽比由图像ID确定并保持嵌套，实际删除比例记录在manifest。ControlNet输入RGB复制，Adapter输入单通道。强度1.5是沿用历史报告设置，未在新开发集重新优化。

本次范围为上述主实验及补指标；不包含原报告的开发强度扫描、Adapter尺度消融、ControlNet时段消融与人工评分。它们不能因本次主实验完成而标成已复现。

## 指标与口径

| 指标 | 计算方式 | 使用范围 |
|---|---|---|
| KID ↓ | torch-fidelity Inception-v3 2048维特征，三次多项式无偏MMD²，gamma=1/2048、coef=1；每个生成种子分别100张真实/100张生成，100次抽取50张子集，报告两个种子KID均值及各自subset标准差 | 全部方法 |
| CLIPScore ↑ | OpenAI CLIP ViT-B/32，`100*max(0, cosine(image,text))`；每图先平均两个种子，再按100张原图bootstrap5000次得到95%区间 | 全部方法 |
| Latency ↓ | 每配置先预热3次，再记录每次正式生成端到端耗时；CUDA计时前后同步；包含文本编码、控制预处理、去噪、VAE、CPU图像返回，不含模型加载、磁盘I/O、初始latent准备、评测 | 全部方法 |
| Throughput ↑ | batch=1串行推理吞吐量=`200/sum(200次生成耗时)`，不含磁盘写入等外部开销；不是批量或最大吞吐量 | 全部方法 |
| Peak VRAM ↓ | 每次调用重置CUDA峰值统计，包含常驻模型；主表报告200次 `max_memory_allocated` 的最大值，均值也保存；另保留reserved，不是nvidia-smi总占用 | 全部方法 |
| Edge F1 ↑ | 生成图按clean参数提边缘，与clean参考比较，距离容差2像素；精确度/召回率调和平均 | 旧方法与SD1.5基线 |
| 保留边缘召回率 ↑ | 输入保留边缘在生成边缘2像素以内的比例 | 旧方法；SD1.5基线按clean参考 |
| 归一化双向边缘距离 ↓ | 两方向最近边缘距离均值再除图像对角线 | 旧方法与SD1.5基线 |

KID报告**原始数值，不乘100或1000**。无偏估计可能为负，不截断。subset标准差不是置信区间；100张测试原图的KID较不稳定，不应当作大样本排行榜。两种子不作为200张独立原图进行CLIP/边缘统计。真实参考按各组分辨率处理后，由Inception内部统一缩放；未把原始照片与生成图配对像素相似度冒充KID。

跨组存在基础模型、分辨率、边缘输入等差异。效率—质量结论主要在SDXL组内比较；边缘控制结论在SD1.5组内比较。所有测量来自本机，模型间的训练差异不能解释为孤立架构因果效应。

## 运行与续跑

建议 Linux、Python3.10、支持CUDA12.8的NVIDIA驱动。交接时不要复制 `.venv`：本机为复用现有CUDA PyTorch，以 `--system-site-packages` 创建了环境；接收者应按下列命令新建独立环境。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/verify_models.py
# 交接包内已包含冻结数据与清单，不必重新抽样。
.venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/verify_project.py
```

单项运行：`--only lcm_4step`；多个配置用逗号连接。`run_experiments.py --smoke` 只生成开发集试跑图，写到 `results/smoke/`，不混入正式结果。`--limit N`仅用于调试，不能称为完整实验。恢复运行会核对配置指纹及输出SHA256，跳过有效已完成项；如不匹配会报错，不覆盖旧结果。已有的 environment.json 不会被覆盖。存在待生成项时还会核对 hardware_profile.json 中的GPU、驱动、CPU和关键依赖版本，阻止不同环境的耗时混入本次实验；换机器或环境重新测量请使用独立实验目录，保留相同数据清单。

本次实际执行的生成源码保存在 `results/source_generation_executed.py`，其SHA256与原环境记录一致。交接版在生成结束后增加上述续跑保护，四个核心配置/加载/输入/生成函数的AST一致性检查通过，见 `results/handoff_code_verification.json`。

`scripts/prepare_data.py` 需要原始标注zip，只用于从头重建数据；现成交接包无需此步骤。必须保留 `data/manifest.json` 作为同学继续实验的固定划分。

## 文件与交接

- `scripts/`：下载、数据准备、生成、评测、汇总与验收脚本。
- `configs/`：固定模型revision和实验矩阵。
- `data/`：冻结清单、caption标注、120张原图、两种分辨率参考图、五种边缘条件。
- `results/runs/`：正式PNG及逐图JSON收据，包括配置、prompt、种子、SHA256、耗时、显存。
- `results/metrics/`：已完成指标及逐图分数；`results/features/`缓存评测特征。
- `results/environment.json`：本机环境；`logs/`：实际运行日志，含失败记录。
- `EXPERIMENT_REPORT.md`：由实际结果生成的进度、指标结果、完成状态。
- `report./`：原始历史交付，保留用于对照，里面的数字不代表本次结果。
- `report./report/human_evaluation/`也是历史材料，对应旧实验的匿名图片；不能将它的评分当作本次新子集的人工评价。本次尚未组织真实人工评分。

交接包排除虚拟环境、下载缓存和模型权重，保留固定revision下载脚本。这样接收者可以直接分析现有结果，下载模型后继续生成。需要离线交接时可另行复制整个 `models/` 目录。

## 官方来源

- https://cocodataset.org/#download
- https://huggingface.co/latent-consistency/lcm-lora-sdxl
- https://huggingface.co/ByteDance/SDXL-Lightning
- https://huggingface.co/lllyasviel/control_v11p_sd15_canny
- https://huggingface.co/TencentARC/t2iadapter_canny_sd15v2
- https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html
- https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html

## GitHub交接包

完整图像与评测缓存通过仓库Release分片交付，下载与合并见 [docs/GITHUB_HANDOFF.md](docs/GITHUB_HANDOFF.md)。Git本身保留代码、清单、原图、逐图收据和指标。
