# 01 反事实成本与收入预测

第一版正式模型：对已经在其他节点出现、但目标节点未运行过的业务，预测该节点的 7 天成本与收入。

## 方法

- 节点内部 90/10 业务遮挡验证。
- 节点静态特征、启动前带宽特征、业务身份和启动时间。
- Leave-One-Out 节点先验和业务先验，避免训练标签泄漏。
- 节点—业务矩阵分解。
- Raw、Log1p、Tweedie 等多种 XGBoost 回归模型。
- 293,969 条部分窗口记录构造的历史日均增强特征。
- 非负且权重和为 1 的 WAPE 约束融合。

## 最终结果

| 目标 | MSE | MAE | R² | WAPE |
|---|---:|---:|---:|---:|
| 成本 | 117,087.18 | 86.92 | 0.6399 | 56.88% |
| 收入 | 203,341.71 | 106.97 | 0.5321 | 61.12% |

`counterfactual_validation_error_details.csv`包含 867 条验证记录的真实金额、预测金额、金额差和百分比误差。

## 运行顺序

```bash
python run_counterfactual_matrix_completion.py
python optimize_counterfactual_blends.py
python run_counterfactual_history_augmentation.py
python export_counterfactual_validation_errors.py
```

前三步会重新训练模型，并覆盖当前模型产物和结果。

