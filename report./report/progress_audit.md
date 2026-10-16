# 实验进度与复核记录（2026-09-19）

结论：2,840张计划图像已在RTX 3090完成，未发现需要重跑正式实验的证据；目前只剩三位评价者的真实人工评分。原实验从环境搭建到完成交付约4小时46分，早于15小时节点。本次复核另外记录，不计入正式图像数。

| 阶段 | 已完成 / 计划 |
| --- | ---: |
| E1 开发强度扫描 | 160 / 160 |
| E2 五条件正式比较 | 2,000 / 2,000 |
| E3 无控制SD1.5 | 200 / 200 |
| E4 Adapter尺度消融 | 240 / 240 |
| E5 ControlNet控制时段 | 240 / 240 |

## 本次核验

- 原验收重新通过：全部图片、模型及输入文件校验，数据划分、嵌套擦除掩码、输入通道、配对噪声、U-Net和控制分支调用次数一致。
- 从原始标注重建种子2026候选顺序和caption选择；120张裁剪、三个Canny阈值条件和240份初始噪声精确一致。保留生成前逐图语义审查和4项排除理由；未按模型输出重新筛样本。
- 四套模型共29个文件与各固定revision的Hugging Face官方LFS SHA256/Git对象校验值匹配。
- 120张原图逐字节匹配服务器公共COCO val2017归档；图像归档与标注归档MD5分别为442b8da7639aecaf257c1dceb8ba8c80、f4bbac642086de4f52a3fdda2de5fa2c，与[公开实现记录的校验值](https://github.com/mlverse/torchvision/blob/main/R/dataset-coco.R)一致。在线逐图复核确认1张后遇到超时，未声称120张均完成在线重下载核验。网络证据及限制原样保存。
- 从全部2,840张生成图重算3,640行边缘指标，误差小于1e-12。指标行多于图像数是因为200张无控制基线各复用于五种条件，未重复计入生成数量。
- 在当前RTX 3090重新生成固定测试图313130、seed index 0的7种配置：SD1.5、完整ControlNet、完整Adapter、最高/最低尺度消融、前/后15步控制。全部与保存结果逐像素一致，对应7个CLIPScore误差小于1e-5。该抽查不代表对全部输出重新生成。
- 原有17项统计表/PNG图重新生成后SHA256一致，PDF仍为7页；三份40行匿名评分模板保持空白，80张A/B图全部匹配冻结映射，包内不展示模型名和原始照片。
- 7项单元与回归测试通过，pip依赖一致性检查通过。保留原锁定环境；requests对已有chardet版本有兼容提示，但离线推理/评测、实际复核与pip check均通过，本次未升级实验依赖。

## 修复与溯源

1. 修复完成后查询状态仍累加机时、丢失交付字段的问题；完成后不再误触发15小时提醒。
2. 修复下载器对已有同大小文件直接跳过SHA256验证的问题，并为小文件下载补充落盘前哈希检查；损坏文件不会被当作有效缓存。
3. README改为使用完整requirements-lock.txt安装，补充独立核验入口。新增audit_experiment.py、verify_saved_report.py及针对性回归测试。

推理代码、选定强度、冻结数据、正式生成图、记录及统计结论未改变。修复前源码、完成时间、旧交付清单和旧源码快照保存在results/audit/pre_audit_revision.zip。当前代码将重新快照并随完整交付包提供。

主要证据：results/audit/verification.json、official_models.json、official_sources.json、coco_archive_checksums.json、report_reproduction.json、human_packet.json；详细日志见logs/current_audit_retry.log和logs/current_report_reproduction.log。

## 下一步

将deliverables/human_evaluation_anonymous.zip分别交给三位真实评价者，独立填写各自40行模板。合并为results/human_ratings.csv后运行`.venv/bin/python -m edge_control human-summary`，再依据真实评分补充报告。未生成或推断任何人工偏好分数。
