# 04｜存量节点业务利润推荐 v4.1

本目录是当前 v4.1 可交付版本，包含模型、推理代码、运行说明和结果快照；不包含内网导出的原始 Excel/CSV，也不包含数据抽取与清洗脚本。

## 1. 模型目标

针对已有静态属性的存量节点，对所有满足准入条件且利润训练样本数大于 3 的业务进行成本、收入和利润评估，输出每个节点的候选业务排序。

业务准入规则：

- 汇聚节点：只允许推荐汇聚业务；
- 专线节点：允许推荐汇聚业务或专线业务。

## 2. 模型结构

模型由两条路线组成：

1. 金额路线：相似节点 XGBoost 成本模型与收入模型分别预测金额，使用“预测收入－预测成本”得到预测利润；再叠加少量稳定性排序信号。
2. 去偏利润路线：相似节点直接利润 XGBoost 与无业务先验的 XGBoost Pairwise Ranker 融合，修正热门业务集中问题。

两条路线分别输出 Top3，再按名次积分合并为每个节点最多 5 个业务候选。

详细结构与验证指标见 [MODEL_CARD.md](MODEL_CARD.md)。

## 3. 目录说明

```text
04_profit_recommendation_v4/
├─ score_recent7d_businesses_v4.py       # v4.1 主运行入口
├─ score_recent7d_all_businesses_v3.py   # 金额及相似节点评分公共逻辑
├─ predict_similar_node_v3.py            # 相似节点模型加载与特征计算
├─ run_v4.ps1                            # Windows 一键运行脚本
├─ requirements.txt
├─ MODEL_CARD.md
├─ outputs/<run_id>/
│  ├─ matched_similar_node_v3_model_artifact/  # 成本、收入、利润基模型
│  ├─ debiased_pairwise_ranker_v4_artifact/    # Pairwise 排序模型
│  └─ debiased_pairwise_ranker_v4_artifact/calibration.json
│                                             # 排序分数校准参数
└─ results/                              # 本次结果快照
```

## 4. 环境安装

建议使用 Python 3.11 或 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 5. 准备运行输入

本交付不包含内网原始数据。运行前请将最新节点—业务日明细放到：

```text
outputs/01a0225a-43b2-7991-a2c3-00e9e4064c5d/aggregation_dedicated_daily_outcomes_last30d.csv
```

脚本只从该文件识别最近 7 个实际有数据的日期和活跃节点；模型训练参考与静态属性已经封装在模型目录中。

## 6. 运行

```powershell
.\run_v4.ps1
```

或直接运行：

```powershell
python score_recent7d_businesses_v4.py
```

首次运行需要对候选节点—业务组合评分，耗时通常在十几分钟量级。中间分业务评分保存在 `.codex_work/v4_1_asymmetric_delivery_score_chunks/`，中断后可以复用。

## 7. 运行输出

运行结果写入 `outputs/01a0225a-43b2-7991-a2c3-00e9e4064c5d/`：

- `v4_1_recent7d_feasible_amount_profit_top3.csv`：金额路线 Top3；
- `v4_1_recent7d_debiased_profit_top3.csv`：去偏利润路线 Top3；
- `v4_1_recent7d_combined_top5.csv`：两条路线合并结果；
- `v4_1_recent7d_recommendation_summary.json`：覆盖率、集中度和验证指标摘要。

`results/` 保存的是 2026-08-21 至 2026-08-27 这一批数据对应的交付结果，包括 Top1 审核表、高确定性切换候选表和业务互换建议。

## 8. 使用边界

- 去偏利润路线输出的是排序修正分，不应直接当作精确金额；
- 高确定性候选仍需检查容量、合同计费和部署可行性；
- 业务 ID `10000292` 在原始数据中缺少名称，涉及该业务的建议上线前必须先补齐业务主数据；
- 不要把内网原始明细、下载 Excel 或带鉴权信息的文件提交到 Git。
