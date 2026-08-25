#!/usr/bin/env python3
"""90/10节点内业务遮挡：矩阵补全 + 节点侧信息的成本/收入反事实金额实验。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from run_cost_regression_optimization import (
    CAT_FEATURES, NODES, OUTCOMES, PRE_RUN_BW_FEATURES, STATIC_NUM_FEATURES,
    best_convex_blend, model_config, prepare_categories, prepare_numeric,
)


HERE = Path(__file__).resolve().parent
METRICS_OUT = HERE / "counterfactual_matrix_completion_metrics.json"
PREDICTIONS_OUT = HERE / "counterfactual_masked_validation_predictions.csv"
REPORT_OUT = HERE / "反事实金额矩阵补全优化报告.md"
ARTIFACT_DIR = HERE / "counterfactual_matrix_completion_artifact"
RANDOM_STATE = 42
TARGETS = {"cost": "cum_cost_7d", "revenue": "cum_revenue_7d"}


def metrics(y: np.ndarray, pred: np.ndarray, clip: bool = True) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if clip:
        pred = np.clip(pred, 0.0, None)
    mse = mean_squared_error(y, pred)
    cutoff = float(np.quantile(y, .99))
    high = y >= cutoff
    nontrivial = y >= 10.0
    return {
        "mse": float(mse), "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
        "wape": float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-12)),
        "smape": float(np.mean(2 * np.abs(y - pred) / np.maximum(np.abs(y) + np.abs(pred), 1e-6))),
        "amount_ge_10_count": int(nontrivial.sum()),
        "amount_ge_10_wape": float(np.abs(y[nontrivial] - pred[nontrivial]).sum() / max(np.abs(y[nontrivial]).sum(), 1e-12)),
        "top_1pct_cutoff": cutoff,
        "top_1pct_rmse": float(math.sqrt(mean_squared_error(y[high], pred[high]))),
    }


def load_frame() -> pd.DataFrame:
    outcome_cols = ["node_id", "business", "business_online_day", "outcome_days", *TARGETS.values()]
    outcomes = pd.read_csv(OUTCOMES, usecols=outcome_cols, dtype={"node_id": str, "business": str})
    frame = outcomes[(outcomes.outcome_days >= 7) & outcomes[list(TARGETS.values())].notna().all(axis=1)].copy()
    header = pd.read_csv(NODES, nrows=0).columns
    wanted = ["node_id"] + [c for c in CAT_FEATURES[1:] + STATIC_NUM_FEATURES + PRE_RUN_BW_FEATURES if c in header]
    nodes = pd.read_csv(NODES, usecols=list(dict.fromkeys(wanted)), dtype={"node_id": str}, low_memory=False)
    frame = frame.merge(nodes.drop_duplicates("node_id"), on="node_id", how="inner", validate="many_to_one")
    date = pd.to_datetime(frame.business_online_day, errors="coerce")
    frame["planned_start_month"] = date.dt.month.astype(float)
    frame["planned_start_day_index"] = (date - pd.Timestamp("2026-01-01")).dt.days.astype(float)
    return frame.reset_index(drop=True)


def masked_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """精确隐藏10%记录，同时保证验证节点和业务在训练中仍有观测。"""
    rng = np.random.default_rng(RANDOM_STATE)
    node_count = frame.node_id.value_counts().to_dict()
    business_count = frame.business.value_counts().to_dict()
    remaining_node, remaining_business = dict(node_count), dict(business_count)
    target_size = int(round(len(frame) * .10))
    candidates = rng.permutation(len(frame))
    held: list[int] = []
    for idx in candidates:
        row = frame.iloc[int(idx)]
        node, business = row.node_id, row.business
        if remaining_node[node] <= 1 or remaining_business[business] <= 1:
            continue
        held.append(int(idx))
        remaining_node[node] -= 1
        remaining_business[business] -= 1
        if len(held) == target_size:
            break
    if len(held) != target_size:
        raise RuntimeError(f"只能安全遮挡{len(held)}条，目标为{target_size}条")
    valid_mask = np.zeros(len(frame), dtype=bool)
    valid_mask[held] = True
    train, valid = frame[~valid_mask].copy(), frame[valid_mask].copy()
    diagnostics = {
        "all_rows": len(frame), "train_rows": len(train), "validation_rows": len(valid),
        "train_ratio": len(train) / len(frame), "validation_ratio": len(valid) / len(frame),
        "all_nodes": frame.node_id.nunique(), "validation_nodes": valid.node_id.nunique(),
        "train_businesses": train.business.nunique(), "validation_businesses": valid.business.nunique(),
        "validation_nodes_missing_from_train": int((~valid.node_id.isin(train.node_id)).sum()),
        "validation_businesses_missing_from_train": int((~valid.business.isin(train.business)).sum()),
    }
    return train, valid, diagnostics


def grouped_stats(frame: pd.DataFrame, group: str, target: str, prefix: str) -> pd.DataFrame:
    global_mean = float(frame[target].mean())
    grouped = frame.groupby(group, observed=True)[target]
    stats = grouped.agg(["count", "sum", "mean", lambda x: x.quantile(.9)])
    stats.columns = ["count", "sum", "mean", "p90"]
    alpha = 5.0 if group == "node_id" else 10.0
    stats["prior"] = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
    return stats.rename(columns={c: f"{prefix}_{c}" for c in stats.columns})


def attach_history(train: pd.DataFrame, valid: pd.DataFrame, target: str, short: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练行用leave-one-out；验证行只看90%训练数据。"""
    tr, va = train.copy(), valid.copy()
    global_mean = float(tr[target].mean())
    for group, prefix, alpha in [("node_id", f"{short}_node", 5.0), ("business", f"{short}_business", 10.0)]:
        sums = tr.groupby(group, observed=True)[target].transform("sum")
        counts = tr.groupby(group, observed=True)[target].transform("count")
        tr[f"{prefix}_prior"] = (sums - tr[target] + alpha * global_mean) / (counts - 1 + alpha)
        tr[f"{prefix}_known_count_log1p"] = np.log1p(np.maximum(counts - 1, 0))
        stats = grouped_stats(tr, group, target, prefix)
        key = va[group]
        va[f"{prefix}_prior"] = key.map(stats[f"{prefix}_prior"]).fillna(global_mean)
        va[f"{prefix}_known_count_log1p"] = np.log1p(key.map(stats[f"{prefix}_count"]).fillna(0.0))
    return tr, va


@dataclass
class MFModel:
    mu: float
    node_bias: np.ndarray
    business_bias: np.ndarray
    node_factors: np.ndarray
    business_factors: np.ndarray
    node_index: dict[str, int]
    business_index: dict[str, int]
    best_epoch: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        u = frame.node_id.astype(str).map(self.node_index).fillna(-1).astype(int).to_numpy()
        i = frame.business.astype(str).map(self.business_index).fillna(-1).astype(int).to_numpy()
        result = np.full(len(frame), self.mu, dtype=float)
        known_u, known_i = u >= 0, i >= 0
        result[known_u] += self.node_bias[u[known_u]]
        result[known_i] += self.business_bias[i[known_i]]
        both = known_u & known_i
        if self.node_factors.shape[1]:
            result[both] += np.sum(self.node_factors[u[both]] * self.business_factors[i[both]], axis=1)
        return np.clip(np.expm1(result), 0.0, None)


def fit_mf(train: pd.DataFrame, valid: pd.DataFrame, target: str, factors: int,
           reg: float, lr: float = .015, epochs: int = 220) -> tuple[MFModel, dict]:
    nodes = sorted(train.node_id.unique())
    businesses = sorted(train.business.unique())
    node_index = {value: idx for idx, value in enumerate(nodes)}
    business_index = {value: idx for idx, value in enumerate(businesses)}
    u = train.node_id.map(node_index).to_numpy(int)
    i = train.business.map(business_index).to_numpy(int)
    y = np.log1p(train[target].to_numpy(float))
    rng = np.random.default_rng(RANDOM_STATE + factors + int(reg * 1000))
    mu = float(y.mean())
    bu, bi = np.zeros(len(nodes)), np.zeros(len(businesses))
    p = rng.normal(0, .03, (len(nodes), factors))
    q = rng.normal(0, .03, (len(businesses), factors))
    best_mse, best_epoch, best_state = float("inf"), 0, None
    order = np.arange(len(train))
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        step = lr / (1.0 + epoch / 120.0)
        for idx in order:
            uu, ii = u[idx], i[idx]
            pu = p[uu].copy()
            prediction = mu + bu[uu] + bi[ii] + float(pu @ q[ii])
            error = float(np.clip(y[idx] - prediction, -8.0, 8.0))
            bu[uu] += step * (error - reg * bu[uu])
            bi[ii] += step * (error - reg * bi[ii])
            if factors:
                p[uu] += step * (error * q[ii] - reg * p[uu])
                q[ii] += step * (error * pu - reg * q[ii])
        if epoch % 10 == 0:
            model = MFModel(mu, bu, bi, p, q, node_index, business_index, epoch)
            score = mean_squared_error(valid[target], model.predict(valid))
            if score < best_mse:
                best_mse, best_epoch = score, epoch
                best_state = (bu.copy(), bi.copy(), p.copy(), q.copy())
            elif epoch - best_epoch >= 50:
                break
    assert best_state is not None
    bu, bi, p, q = best_state
    model = MFModel(mu, bu, bi, p, q, node_index, business_index, best_epoch)
    return model, {"factors": factors, "reg": reg, "best_epoch": best_epoch, "validation_mse": best_mse}


def fit_xgb(name: str, config: str, train: pd.DataFrame, valid: pd.DataFrame,
            target: str, features: list[str]) -> tuple[XGBRegressor, np.ndarray]:
    log_target = config == "xgb_log1p"
    y_train = np.log1p(train[target]) if log_target else train[target]
    y_valid = np.log1p(valid[target]) if log_target else valid[target]
    model = XGBRegressor(**model_config(config))
    model.fit(train[features], y_train, eval_set=[(valid[features], y_valid)], verbose=False)
    pred = model.predict(valid[features])
    if log_target:
        pred = np.expm1(pred)
    return model, np.clip(pred, 0.0, None)


def run_target(base_train: pd.DataFrame, base_valid: pd.DataFrame, target: str, short: str) -> tuple[dict, pd.DataFrame, dict]:
    train, valid = attach_history(base_train, base_valid, target, short)
    y = valid[target].to_numpy()
    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    global_mean = float(train[target].mean())
    predictions["global_mean"] = np.full(len(valid), global_mean)
    predictions["business_prior"] = valid[f"{short}_business_prior"].to_numpy()
    predictions["node_history_prior"] = valid[f"{short}_node_prior"].to_numpy()
    for name in list(predictions):
        results[name] = {"validation": metrics(y, predictions[name])}

    mf_candidates: dict[str, MFModel] = {}
    for factors, reg in [(0, .05), (4, .05), (8, .05), (16, .05), (8, .15), (16, .15)]:
        name = f"mf_k{factors}_reg{reg}"
        print(f"{short}: fitting {name}", flush=True)
        model, info = fit_mf(train, valid, target, factors, reg)
        mf_candidates[name] = model
        predictions[name] = model.predict(valid)
        results[name] = {"config": info, "validation": metrics(y, predictions[name])}
    best_mf = min(mf_candidates, key=lambda name: results[name]["validation"]["mse"])

    cat_features = ["node_id"] + [c for c in CAT_FEATURES if c in train]
    category_map = prepare_categories([train, valid], cat_features)
    history_features = [f"{short}_node_prior", f"{short}_node_known_count_log1p",
                        f"{short}_business_prior", f"{short}_business_known_count_log1p"]
    numeric = ([c for c in STATIC_NUM_FEATURES + PRE_RUN_BW_FEATURES if c in train]
               + history_features + ["planned_start_month", "planned_start_day_index"])
    numeric_expanded, medians = prepare_numeric([train, valid], list(dict.fromkeys(numeric)), train)
    xgb_features = cat_features + numeric_expanded
    xgb_models: dict[str, XGBRegressor] = {}
    for name, config in [("xgb_raw_depth4", "xgb_raw_depth4_regularized"),
                         ("xgb_raw_depth8", "xgb_raw_depth8"),
                         ("xgb_log1p", "xgb_log1p"), ("xgb_tweedie", "xgb_tweedie_1_3")]:
        print(f"{short}: fitting {name}", flush=True)
        model, pred = fit_xgb(name, config, train, valid, target, xgb_features)
        xgb_models[name], predictions[name] = model, pred
        results[name] = {"best_iteration": int(model.best_iteration), "validation": metrics(y, pred)}

    blend_candidates = {
        "business_prior": predictions["business_prior"],
        "node_history_prior": predictions["node_history_prior"],
        best_mf: predictions[best_mf],
        **{name: predictions[name] for name in xgb_models},
    }
    weights, blend_pred = best_convex_blend(y, blend_candidates)
    predictions["nonnegative_stack"] = blend_pred
    results["nonnegative_stack"] = {"weights": weights, "validation": metrics(y, blend_pred)}
    selected = min(results, key=lambda name: results[name]["validation"]["mse"])

    artifact = {
        "selected_model": selected, "stack_weights": weights, "best_mf": best_mf,
        "category_map": category_map, "numeric_medians": medians,
        "categorical_features": cat_features, "numeric_features": numeric_expanded,
        "xgb_features": xgb_features,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    for name, model in xgb_models.items():
        model.save_model(ARTIFACT_DIR / f"{short}_{name}.json")
    mf_model = mf_candidates[best_mf]
    np.savez_compressed(
        ARTIFACT_DIR / f"{short}_{best_mf}.npz", mu=mf_model.mu,
        node_bias=mf_model.node_bias, business_bias=mf_model.business_bias,
        node_factors=mf_model.node_factors, business_factors=mf_model.business_factors,
        node_values=np.array(list(mf_model.node_index), dtype=object),
        business_values=np.array(list(mf_model.business_index), dtype=object),
        best_epoch=mf_model.best_epoch,
    )
    output = valid[["node_id", "business", "business_online_day", target]].copy()
    for name, pred in predictions.items():
        output[f"pred_{name}"] = pred
    output["selected_prediction"] = predictions[selected]
    output["absolute_error"] = np.abs(output[target] - output.selected_prediction)
    return {"selected_model": selected, "experiments": results,
            "selected_validation_metrics": results[selected]["validation"]}, output, artifact


def main() -> None:
    frame = load_frame()
    train, valid, split_info = masked_split(frame)
    all_results, outputs, artifacts = {}, [], {}
    started = time.time()
    for short, target in TARGETS.items():
        result, output, artifact = run_target(train, valid, target, short)
        all_results[short], artifacts[short] = result, artifact
        output = output.rename(columns={target: f"actual_{short}_7d", "selected_prediction": f"predicted_{short}_7d",
                                       "absolute_error": f"{short}_absolute_error"})
        keep = ["node_id", "business", "business_online_day", f"actual_{short}_7d",
                f"predicted_{short}_7d", f"{short}_absolute_error"]
        outputs.append(output[keep])
    predictions = outputs[0].merge(outputs[1], on=["node_id", "business", "business_online_day"], validate="one_to_one")
    predictions["actual_profit_7d"] = predictions.actual_revenue_7d - predictions.actual_cost_7d
    predictions["predicted_profit_7d"] = predictions.predicted_revenue_7d - predictions.predicted_cost_7d
    predictions.to_csv(PREDICTIONS_OUT, index=False)
    profit_metrics = metrics(predictions.actual_profit_7d, predictions.predicted_profit_7d, clip=False)

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "validation_design": "90/10 within-node masked observed-business holdout",
        "split": split_info, "targets": all_results, "profit_validation_metrics": profit_metrics,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "validation contains only historically observed businesses artificially hidden from the model",
            "unobserved real counterfactual outcomes remain unknowable without exploration or additional assumptions",
            "only complete seven-day outcomes are used",
        ],
    }
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 反事实金额矩阵补全优化报告", "", "## 验证设计", "",
             f"- 完整样本：{split_info['all_rows']:,}；训练：{split_info['train_rows']:,}（{split_info['train_ratio']:.1%}）；验证：{split_info['validation_rows']:,}（{split_info['validation_ratio']:.1%}）。",
             f"- 验证节点：{split_info['validation_nodes']:,}；所有验证节点及业务均在训练中保留至少一个观测。",
             "- 验证方式：人工隐藏已运行过的业务，模拟同一节点未运行该业务时的金额补全。", ""]
    for short, label in [("cost", "成本"), ("revenue", "收入")]:
        result = all_results[short]
        lines += [f"## {label}结果", "", f"最终模型：`{result['selected_model']}`。", "",
                  "| 方案 | MSE | RMSE | MAE | R² | WAPE | 金额≥10 WAPE |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in result["experiments"].items():
            m = row["validation"]
            lines.append(f"| {name} | {m['mse']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['wape']:.2%} | {m['amount_ge_10_wape']:.2%} |")
        lines.append("")
    lines += ["## 解释边界", "",
              "- 这是暖启动反事实近似：节点至少有一个其他业务金额可供模型校准。",
              "- 验证集参与模型选择，因此这里是验证结果，不是最终独立测试结果。",
              "- 真正从未运行过的业务仍没有真实标签；遮挡验证只能衡量在历史可观测分布内的补全能力。"]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"split": split_info,
                      "cost": all_results["cost"]["selected_validation_metrics"],
                      "revenue": all_results["revenue"]["selected_validation_metrics"],
                      "profit": profit_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
