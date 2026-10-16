# 实验完成与接续记录

本次约定矩阵已全部完成，无待生成或待评测项。18组配置，每组100张测试原图×2种子，共3600张正式输出；生成阶段约2.96小时，评测约122秒。

- 数据：COCO val2017，新冻结120张（20 dev、100 test），完整清单见 data/manifest.json 和 docs/DATASET.md。历史完整划分未找回，不能称为历史子集精确恢复。
- 新方法：LCM-LoRA SDXL与SDXL-Lightning，各2/4/8步；原生SDXL 30步基线。
- 旧方法：ControlNet Canny、T2I-Adapter Canny，各clean、erase25、erase50、sparse、dense；SD1.5 30步基线。
- 全部18组已测KID、CLIPScore、Latency、Throughput、Peak VRAM；旧方法与SD1.5基线另测Edge F1、保留边缘召回率、归一化双向边缘距离。
- 正式结果：EXPERIMENT_REPORT.md、results/metrics/；逐图结果和运行收据：results/runs/。
- 验收：results/verification.json 为PASS；模型权重哈希核验、KID实现一致性核对和续跑保护测试均通过。
- 原始执行源码：results/source_generation_executed.py。交接脚本增加环境保护，不覆盖原环境记录；核心函数AST一致性核验见results/handoff_code_verification.json。
- 不同机器或依赖环境应使用独立实验目录重新测量，不能把新耗时混入旧结果；沿用冻结数据清单。
- 历史强度扫描、尺度/时段消融与真实人工评分未在本批运行，属于可选后续工作。

交接zip位于项目目录同级。zip不含models与.venv，包含数据、代码、结果、日志和固定版本下载说明。最终压缩包大小、SHA256及成员CRC/SHA256验收结果见本机results/package_receipt.json与zip同级.sha256文件；receipt本身不装入zip以避免自引用。完整操作见README.md。
