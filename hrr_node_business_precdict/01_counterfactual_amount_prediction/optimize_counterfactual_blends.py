#!/usr/bin/env python3
"""复用已训练基模型，训练受约束融合层：WAPE优先且权重非负、和为1。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from xgboost import XGBRegressor

from run_counterfactual_matrix_completion import (
    ARTIFACT_DIR, HERE, METRICS_OUT, PREDICTIONS_OUT, TARGETS,
    MFModel, attach_history, load_frame, masked_split, metrics,
)
from run_cost_regression_optimization import prepare_categories, prepare_numeric


OUT = HERE / "counterfactual_constrained_blend_metrics.json"


def load_mf(short: str, name: str) -> MFModel:
    data = np.load(ARTIFACT_DIR / f"{short}_{name}.npz", allow_pickle=True)
    nodes = [str(value) for value in data["node_values"].tolist()]
    businesses = [str(value) for value in data["business_values"].tolist()]
    return MFModel(
        float(data["mu"]), data["node_bias"], data["business_bias"],
        data["node_factors"], data["business_factors"],
        {value: idx for idx, value in enumerate(nodes)},
        {value: idx for idx, value in enumerate(businesses)}, int(data["best_epoch"]),
    )


def convex_l1(y: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    """线性规划精确最小化验证集绝对误差；WAPE分母固定，因此也最小化WAPE。"""
    n, k = prediction_matrix.shape
    objective = np.r_[np.zeros(k), np.ones(n)]
    upper_a = np.vstack([
        np.c_[prediction_matrix, -np.eye(n)],
        np.c_[-prediction_matrix, -np.eye(n)],
    ])
    upper_b = np.r_[y, -y]
    equality_a = np.zeros((1, k + n))
    equality_a[0, :k] = 1.0
    result = linprog(objective, A_ub=upper_a, b_ub=upper_b,
                     A_eq=equality_a, b_eq=np.array([1.0]),
                     bounds=[(0.0, 1.0)] * k + [(0.0, None)] * n, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:k]


def convex_mse(y: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    k = prediction_matrix.shape[1]
    scale = max(float(np.mean(y ** 2)), 1.0)
    result = minimize(
        lambda w: float(np.mean((y - prediction_matrix @ w) ** 2) / scale),
        np.full(k, 1.0 / k), method="SLSQP", bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.clip(result.x, 0.0, 1.0) / np.clip(result.x.sum(), 1e-12, None)


def candidate_predictions(train: pd.DataFrame, valid: pd.DataFrame, target: str,
                          short: str, artifact: dict) -> dict[str, np.ndarray]:
    train, valid = attach_history(train, valid, target, short)
    predictions = {
        "business_prior": valid[f"{short}_business_prior"].to_numpy(),
        "node_history_prior": valid[f"{short}_node_prior"].to_numpy(),
    }
    prepare_categories([train, valid], artifact["categorical_features"])
    base_numeric = [name for name in artifact["numeric_features"] if not name.endswith("__missing")]
    prepare_numeric([train, valid], base_numeric, train)
    for name in ["xgb_raw_depth4", "xgb_raw_depth8", "xgb_log1p", "xgb_tweedie"]:
        model = XGBRegressor()
        model.load_model(ARTIFACT_DIR / f"{short}_{name}.json")
        pred = model.predict(valid[artifact["xgb_features"]])
        if name == "xgb_log1p":
            pred = np.expm1(pred)
        predictions[name] = np.clip(pred, 0.0, None)
    mf_name = artifact["best_mf"]
    predictions[mf_name] = load_mf(short, mf_name).predict(valid)
    return predictions


def main() -> None:
    frame = load_frame()
    train, valid, split = masked_split(frame)
    artifacts = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    previous = json.loads(METRICS_OUT.read_text(encoding="utf-8"))
    output = valid[["node_id", "business", "business_online_day", *TARGETS.values()]].copy()
    results = {"split": split, "targets": {}}
    for short, target in TARGETS.items():
        predictions = candidate_predictions(train, valid, target, short, artifacts[short])
        names = list(predictions)
        matrix = np.column_stack([predictions[name] for name in names])
        y = valid[target].to_numpy(float)
        l1_weights = convex_l1(y, matrix)
        mse_weights = convex_mse(y, matrix)
        variants = {name: metrics(y, pred) for name, pred in predictions.items()}
        variants["wape_constrained_blend"] = metrics(y, matrix @ l1_weights)
        variants["mse_constrained_blend"] = metrics(y, matrix @ mse_weights)
        old_metrics = previous["targets"][short]["selected_validation_metrics"]
        # 部署选择：优先WAPE；若多个方案WAPE相近（绝对差<=1个百分点），选MSE更低者。
        best_wape = min(row["wape"] for row in variants.values())
        eligible = [name for name, row in variants.items() if row["wape"] <= best_wape + .01]
        selected = min(eligible, key=lambda name: variants[name]["mse"])
        if selected == "wape_constrained_blend":
            selected_pred, selected_weights = matrix @ l1_weights, dict(zip(names, l1_weights.tolist()))
        elif selected == "mse_constrained_blend":
            selected_pred, selected_weights = matrix @ mse_weights, dict(zip(names, mse_weights.tolist()))
        else:
            selected_pred, selected_weights = predictions[selected], {selected: 1.0}
        output[f"predicted_{short}_7d"] = np.clip(selected_pred, 0.0, None)
        output[f"{short}_absolute_error"] = np.abs(y - selected_pred)
        results["targets"][short] = {
            "previous_unconstrained_mse_selection": old_metrics,
            "variants": variants,
            "wape_blend_weights": {name: float(w) for name, w in zip(names, l1_weights) if w > 1e-7},
            "mse_blend_weights": {name: float(w) for name, w in zip(names, mse_weights) if w > 1e-7},
            "recommended_model": selected,
            "recommended_weights": {name: float(w) for name, w in selected_weights.items() if w > 1e-7},
            "recommended_metrics": variants[selected],
        }
    output = output.rename(columns={"cum_cost_7d": "actual_cost_7d", "cum_revenue_7d": "actual_revenue_7d"})
    output["actual_profit_7d"] = output.actual_revenue_7d - output.actual_cost_7d
    output["predicted_profit_7d"] = output.predicted_revenue_7d - output.predicted_cost_7d
    output.to_csv(PREDICTIONS_OUT, index=False)
    results["profit_metrics"] = metrics(output.actual_profit_7d, output.predicted_profit_7d, clip=False)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    report = HERE / "反事实金额矩阵补全优化报告.md"
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n## 第二轮：WAPE约束优化\n\n")
        handle.write("融合权重限制为非负且总和等于1，避免只追求大额MSE导致整体金额膨胀。\n\n")
        for short, label in [("cost", "成本"), ("revenue", "收入")]:
            row = results["targets"][short]
            m = row["recommended_metrics"]
            handle.write(f"- {label}推荐：`{row['recommended_model']}`；MSE {m['mse']:.2f}，MAE {m['mae']:.2f}，R² {m['r2']:.4f}，WAPE {m['wape']:.2%}，金额≥10 WAPE {m['amount_ge_10_wape']:.2%}。\n")
    print(json.dumps({short: {"model": row["recommended_model"], **row["recommended_metrics"]}
                      for short, row in results["targets"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
