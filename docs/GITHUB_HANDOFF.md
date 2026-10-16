# GitHub交接

这是一个项目，共4个method：LCM-LoRA（SDXL）、SDXL-Lightning、ControlNet Canny、T2I-Adapter Canny；另外包含原生SDXL和SD1.5两个基线。不同步数和边缘条件合计18组配置。实验结果见EXPERIMENT_REPORT.md。

Git保存代码、依赖锁、文档、原图、冻结清单、逐图JSON收据、指标和历史报告。生成PNG、处理后的参考图/边缘与特征缓存在Release完整交接包中；模型权重和虚拟环境不包含，按README下载和安装。

完整交接包分为两个二进制分片，下载同一Release中的全部分片及SHA256文件后合并：

```bash
cat 6132_handoff_20261016.zip.part01 6132_handoff_20261016.zip.part02 > 6132_handoff_20261016.zip
sha256sum -c 6132_handoff_20261016.zip.sha256
unzip 6132_handoff_20261016.zip
```

解压包保存本次完成时的完整实验快照。Git仓库另包含GitHub交接说明及忽略规则。仅克隆Git不能直接核验所有3600张PNG；请先下载完整包。新机器重新测量使用独立目录，避免混合效率结果。
