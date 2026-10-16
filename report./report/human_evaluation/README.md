# 人工评价说明

打开 index.html，三位评价者分别独立填写 rater1_template.csv、rater2_template.csv、rater3_template.csv 中对应的40行。三个维度分别填 A、B 或 tie；不查看其他评价者评分。合并三份CSV（仅保留一个表头）后保存为本项目 results/human_ratings.csv；ratings_template.csv 是含全部120行的替代模板，无需重复填写。随后运行 `python -m edge_control human-summary`。盲法映射存放在数据清单中，不向评价者提供。
