#!/usr/bin/env python3
"""节点×业务 7 天矿主成本回归：节点级切分、验证集选型、测试集一次评估。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
OUTCOMES = HERE / "multibusiness_outcomes.csv"
NODES = HERE / "multibusiness_nodes.csv"
METRICS_OUT = HERE / "cost_regression_optimization_metrics.json"
PREDICTIONS_OUT = HERE / "cost_regression_test_predictions.csv"
IMPORTANCE_OUT = HERE / "cost_regression_feature_importance.csv"
REPORT_OUT = HERE / "矿主成本预测优化报告.md"
ARTIFACT_DIR = HERE / "cost_regression_artifact"

RANDOM_STATE = 42

CAT_FEATURES = [
    "business", "vendorid", "deliverytype", "resourcetype", "dialtype", "nattype",
    "scheduleisps", "regsource", "customermode", "province", "isp", "city",
    "device_type", "arch_type", "isvm", "qoskiller_status", "os", "arch",
    "hardwaretype", "node_manufacturer", "node_model", "idc_id", "idc_name",
    "analysis_nodecustmertype", "analysis_nodedeliverytype", "analysis_tcpnattype",
    "analysis_udpnattype", "analysis_cgroupversion", "analysis_lastdeploystate",
    "analysis_lastdeployscenario", "analysis_cooperationtype",
    "analysis_supply_side_delivery_type", "analysis_isroot",
    "analysis_issupportipv6", "analysis_upnpstate", "dial_ipv6_enable",
    "dial_on_physical_nic", "dial_lbtype", "dial_lbtype_v6", "join_ismanaged",
    "join_isminorisp", "join_isbantransprov", "join_isipv6schedule",
    "join_natforwardenable", "join_idcbindtype", "join_deploystate",
]

STATIC_NUM_FEATURES = [
    "bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize",
    "systemdisksize", "join_cpu_corenumber", "join_cpu_totalcores",
    "join_cpu_totalphysicals", "join_cpu_totalthreads", "join_memtotal",
    "join_disks_totalsize", "join_disks_total_size", "join_disks_hdd_size",
    "join_disks_ssd_size", "join_disks_system_size",
]

# 这些字段是候选业务启动前的节点快照，作为“近期可跑带宽代理”单独做增量实验。
# 它们不是候选业务未来 7 天真实跑量，因而可以用于预测，但不能解释为结算计量值。
PRE_RUN_BW_FEATURES = [
    "actualbandwidth", "netbenchlimitbandwidth", "analysis_yesterday_p95_bw",
    "analysis_yesterday_snapshot_bw", "analysis_yesterday_snapshot_netbench_bw",
    "analysis_dby_p95_bw", "join_actualbw", "join_limitbw", "join_biz_bw",
    "join_yesterday_avg_peak_ratio", "sevendayavg95ratio",
]

PRIOR_FEATURES = [
    "business_cost_prior", "business_cost_p90", "business_positive_rate",
    "business_train_count_log1p",
]


def finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = finite(y_true)
    yp = np.clip(finite(y_pred), 0.0, None)
    mse = mean_squared_error(yt, yp)
    cutoff = float(np.quantile(yt, 0.99))
    high = yt >= cutoff
    return {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "wape": float(np.abs(yt - yp).sum() / max(np.abs(yt).sum(), 1e-12)),
        "top_1pct_cutoff": cutoff,
        "top_1pct_rmse": float(math.sqrt(mean_squared_error(yt[high], yp[high]))),
    }


def business_stats(frame: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(frame["cum_cost_7d"].mean())
    grouped = frame.groupby("business", observed=True)["cum_cost_7d"]
    stats = grouped.agg(["count", "mean", lambda x: x.quantile(0.9), lambda x: (x > 0).mean()])
    stats.columns = ["count", "mean", "p90", "positive_rate"]
    # 少量样本业务向全局均值收缩，减少稀有业务的过拟合。
    alpha = 10.0
    stats["prior"] = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
    return stats


def attach_prior(frame: pd.DataFrame, stats: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    result = frame.copy()
    key = result["business"].astype(str)
    result["business_cost_prior"] = key.map(stats["prior"]).fillna(global_mean)
    result["business_cost_p90"] = key.map(stats["p90"]).fillna(global_mean)
    result["business_positive_rate"] = key.map(stats["positive_rate"]).fillna(0.5)
    result["business_train_count_log1p"] = np.log1p(key.map(stats["count"]).fillna(0.0))
    return result


def attach_oof_prior(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in PRIOR_FEATURES:
        result[column] = np.nan
    splitter = GroupKFold(n_splits=5)
    for fit_idx, hold_idx in splitter.split(result, groups=result["node_id"]):
        fit = result.iloc[fit_idx]
        stats = business_stats(fit)
        mapped = attach_prior(result.iloc[hold_idx], stats, float(fit["cum_cost_7d"].mean()))
        result.loc[result.index[hold_idx], PRIOR_FEATURES] = mapped[PRIOR_FEATURES].to_numpy()
    return result


def load_frame() -> pd.DataFrame:
    outcome_cols = ["node_id", "business", "business_online_day", "cum_cost_7d", "outcome_days"]
    outcomes = pd.read_csv(OUTCOMES, usecols=outcome_cols, dtype={"node_id": str, "business": str})
    outcomes = outcomes[(outcomes["outcome_days"] >= 7) & outcomes["cum_cost_7d"].notna()].copy()
    node_header = pd.read_csv(NODES, nrows=0).columns
    wanted = ["node_id"] + [c for c in CAT_FEATURES[1:] + STATIC_NUM_FEATURES + PRE_RUN_BW_FEATURES if c in node_header]
    nodes = pd.read_csv(NODES, usecols=list(dict.fromkeys(wanted)), dtype={"node_id": str})
    nodes = nodes.drop_duplicates("node_id")
    frame = outcomes.merge(nodes, on="node_id", how="inner", validate="many_to_one")
    frame["business"] = frame["business"].astype(str)
    date = pd.to_datetime(frame["business_online_day"], errors="coerce")
    frame["planned_start_month"] = date.dt.month.astype(float)
    frame["planned_start_day_index"] = (date - pd.Timestamp("2026-01-01")).dt.days.astype(float)
    return frame


def split_nodes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = np.array(sorted(frame["node_id"].unique()))
    train_nodes, test_nodes = train_test_split(nodes, test_size=0.20, random_state=RANDOM_STATE)
    fit_nodes, val_nodes = train_test_split(train_nodes, test_size=0.20, random_state=RANDOM_STATE + 1)
    return (
        frame[frame["node_id"].isin(fit_nodes)].copy(),
        frame[frame["node_id"].isin(val_nodes)].copy(),
        frame[frame["node_id"].isin(test_nodes)].copy(),
    )


def prepare_categories(parts: list[pd.DataFrame], columns: list[str]) -> dict[str, list[str]]:
    category_map: dict[str, list[str]] = {}
    for column in columns:
        values = pd.concat([p[column].fillna("__MISSING__").astype(str) for p in parts], ignore_index=True)
        categories = pd.Index(values.unique())
        category_map[column] = categories.astype(str).tolist()
        for part in parts:
            part[column] = pd.Categorical(part[column].fillna("__MISSING__").astype(str), categories=categories)
    return category_map


def prepare_numeric(parts: list[pd.DataFrame], columns: list[str], fit: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    expanded = []
    medians: dict[str, float] = {}
    for column in columns:
        for part in parts:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        median = float(fit[column].median()) if fit[column].notna().any() else 0.0
        medians[column] = median
        missing_col = f"{column}__missing"
        for part in parts:
            part[missing_col] = part[column].isna().astype(np.int8)
            part[column] = part[column].fillna(median).astype(np.float32)
        expanded.extend([column, missing_col])
    return expanded, medians


def model_config(name: str) -> dict:
    common = dict(
        n_estimators=1400,
        learning_rate=0.035,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        reg_alpha=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=80,
    )
    variants = {
        "xgb_raw_depth6": dict(objective="reg:squarederror", max_depth=6, min_child_weight=8),
        "xgb_raw_depth4_regularized": dict(objective="reg:squarederror", max_depth=4, min_child_weight=15,
                                              reg_lambda=20.0),
        "xgb_raw_depth8": dict(objective="reg:squarederror", max_depth=8, min_child_weight=12,
                                reg_lambda=20.0),
        "xgb_tweedie_1_3": dict(objective="reg:tweedie", tweedie_variance_power=1.3,
                                 max_depth=6, min_child_weight=8),
        "xgb_log1p": dict(objective="reg:squarederror", max_depth=6, min_child_weight=8),
    }
    common.update(variants[name])
    return common


def fit_predict(name: str, fit: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                feature_columns: list[str]) -> tuple[XGBRegressor, np.ndarray, np.ndarray, float]:
    transform_log = name == "xgb_log1p"
    y_fit = np.log1p(fit["cum_cost_7d"].to_numpy()) if transform_log else fit["cum_cost_7d"].to_numpy()
    y_val = np.log1p(val["cum_cost_7d"].to_numpy()) if transform_log else val["cum_cost_7d"].to_numpy()
    model = XGBRegressor(**model_config(name))
    started = time.time()
    model.fit(
        fit[feature_columns], y_fit,
        eval_set=[(val[feature_columns], y_val)],
        verbose=False,
    )
    val_pred = model.predict(val[feature_columns])
    test_pred = model.predict(test[feature_columns])
    if transform_log:
        val_pred = np.expm1(val_pred)
        test_pred = np.expm1(test_pred)
    return model, np.clip(val_pred, 0, None), np.clip(test_pred, 0, None), time.time() - started


def best_blend(y: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray]:
    names = list(predictions)
    best_weights = {names[0]: 1.0}
    best_pred = predictions[names[0]]
    best_mse = mean_squared_error(y, best_pred)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for weight in np.linspace(0.0, 1.0, 21):
                pred = weight * predictions[left] + (1.0 - weight) * predictions[right]
                score = mean_squared_error(y, pred)
                if score < best_mse:
                    best_mse = score
                    best_pred = pred
                    best_weights = {left: float(weight), right: float(1.0 - weight)}
    return best_weights, best_pred


def best_convex_blend(y: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray]:
    """用非负最小二乘直接学习多模型堆叠权重，目标就是验证集 MSE。"""
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    weights, _ = nnls(matrix, y, maxiter=5000)
    kept = {name: float(weight) for name, weight in zip(names, weights) if weight >= 1e-5}
    return kept, matrix @ weights


def main() -> None:
    frame = load_frame()
    fit, val, test = split_nodes(frame)
    global_mean = float(fit["cum_cost_7d"].mean())
    stats = business_stats(fit)
    fit = attach_oof_prior(fit)
    val = attach_prior(val, stats, global_mean)
    test = attach_prior(test, stats, global_mean)

    all_cat = [c for c in CAT_FEATURES if c in fit]
    category_map = prepare_categories([fit, val, test], all_cat)
    date_features = ["planned_start_month", "planned_start_day_index"]
    static_num = [c for c in STATIC_NUM_FEATURES if c in fit] + PRIOR_FEATURES + date_features
    operational_num = static_num + [c for c in PRE_RUN_BW_FEATURES if c in fit]
    numeric_expanded, numeric_medians = prepare_numeric(
        [fit, val, test], list(dict.fromkeys(operational_num)), fit
    )
    static_expanded = [c for c in numeric_expanded if c.removesuffix("__missing") in static_num]
    operational_expanded = numeric_expanded

    y_val = val["cum_cost_7d"].to_numpy()
    y_test = test["cum_cost_7d"].to_numpy()
    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, XGBRegressor] = {}
    results: dict[str, dict] = {}

    val_predictions["global_mean"] = np.full(len(val), global_mean)
    test_predictions["global_mean"] = np.full(len(test), global_mean)
    val_predictions["business_prior"] = val["business_cost_prior"].to_numpy()
    test_predictions["business_prior"] = test["business_cost_prior"].to_numpy()
    for name in ["global_mean", "business_prior"]:
        results[name] = {
            "feature_set": "baseline",
            "validation": metrics(y_val, val_predictions[name]),
            "test": metrics(y_test, test_predictions[name]),
        }

    experiments = [
        ("xgb_raw_depth6_static", "xgb_raw_depth6", all_cat + static_expanded, "static+business_prior"),
        ("xgb_raw_depth6_operational", "xgb_raw_depth6", all_cat + operational_expanded, "static+pre_run_bandwidth"),
        ("xgb_raw_depth4_regularized", "xgb_raw_depth4_regularized", all_cat + operational_expanded, "static+pre_run_bandwidth"),
        ("xgb_raw_depth8", "xgb_raw_depth8", all_cat + operational_expanded, "static+pre_run_bandwidth"),
        ("xgb_tweedie_1_3", "xgb_tweedie_1_3", all_cat + operational_expanded, "static+pre_run_bandwidth"),
        ("xgb_log1p", "xgb_log1p", all_cat + operational_expanded, "static+pre_run_bandwidth"),
    ]
    experiment_columns = {output_name: columns for output_name, _config, columns, _set in experiments}
    for output_name, config_name, columns, feature_set in experiments:
        print(f"fitting {output_name}: rows={len(fit):,}, features={len(columns)}", flush=True)
        model, val_pred, test_pred, seconds = fit_predict(config_name, fit, val, test, columns)
        fitted_models[output_name] = model
        val_predictions[output_name] = val_pred
        test_predictions[output_name] = test_pred
        results[output_name] = {
            "feature_set": feature_set,
            "feature_count": len(columns),
            "best_iteration": int(model.best_iteration),
            "fit_seconds": float(seconds),
            "validation": metrics(y_val, val_pred),
            "test": metrics(y_test, test_pred),
        }
        print(json.dumps({output_name: results[output_name]["validation"]}, ensure_ascii=False), flush=True)

    candidate_predictions = {k: v for k, v in val_predictions.items() if k.startswith("xgb_")}
    blend_weights, blend_val = best_blend(y_val, candidate_predictions)
    blend_test = sum(blend_weights[name] * test_predictions[name] for name in blend_weights)
    results["validation_selected_blend"] = {
        "weights": blend_weights,
        "validation": metrics(y_val, blend_val),
        "test": metrics(y_test, blend_test),
    }

    convex_candidates = {"business_prior": val_predictions["business_prior"], **candidate_predictions}
    convex_weights, convex_val = best_convex_blend(y_val, convex_candidates)
    convex_test = sum(convex_weights[name] * test_predictions[name] for name in convex_weights)
    results["validation_selected_convex_blend"] = {
        "weights": convex_weights,
        "validation": metrics(y_val, convex_val),
        "test": metrics(y_test, convex_test),
    }

    best_name = min(
        [name for name in results if name.startswith("xgb_")],
        key=lambda name: results[name]["validation"]["mse"],
    )
    selectable = [best_name, "validation_selected_blend", "validation_selected_convex_blend"]
    final_name = min(selectable, key=lambda name: results[name]["validation"]["mse"])
    if final_name == "validation_selected_convex_blend":
        final_test_pred = convex_test
    elif final_name == "validation_selected_blend":
        final_test_pred = blend_test
    else:
        final_test_pred = test_predictions[best_name]

    prediction_frame = test[["node_id", "business", "business_online_day", "cum_cost_7d"]].copy()
    prediction_frame["predicted_cost_7d"] = final_test_pred
    prediction_frame["residual"] = prediction_frame["cum_cost_7d"] - prediction_frame["predicted_cost_7d"]
    for name, values in test_predictions.items():
        prediction_frame[f"pred_{name}"] = values
    prediction_frame.to_csv(PREDICTIONS_OUT, index=False)

    importance_model_name = best_name if best_name in fitted_models else min(
        fitted_models, key=lambda name: results[name]["validation"]["mse"]
    )
    model = fitted_models[importance_model_name]
    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "gain_importance": model.feature_importances_,
    }).sort_values("gain_importance", ascending=False)
    importance.to_csv(IMPORTANCE_OUT, index=False)

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "task": "node_business_to_7d_miner_cost",
        "target": "cum_cost_7d",
        "selection_metric": "validation_mse",
        "split": "node-level 64/16/20 fit/validation/test",
        "rows": {"all": len(frame), "fit": len(fit), "validation": len(val), "test": len(test)},
        "nodes": {"all": frame["node_id"].nunique(), "fit": fit["node_id"].nunique(),
                  "validation": val["node_id"].nunique(), "test": test["node_id"].nunique()},
        "target_summary": frame["cum_cost_7d"].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_dict(),
        "billing_feature_status": {
            "available_in_source_schema": ["cost_priceType", "cost_price", "cost_measure", "cost_guaranteedRate", "peak95"],
            "available_in_local_training_file": [],
            "proxy_used": "OOF business historical cost statistics + pre-run bandwidth snapshot",
        },
        "experiments": results,
        "selected_model": final_name,
        "selected_test_metrics": metrics(y_test, final_test_pred),
        "importance_model": importance_model_name,
    }
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    component_names = [name for name in results[final_name].get("weights", {}) if name in fitted_models]
    for name in component_names:
        fitted_models[name].save_model(ARTIFACT_DIR / f"{name}.json")
    artifact_metadata = {
        "selected_model": final_name,
        "weights": results[final_name].get("weights", {best_name: 1.0}),
        "component_model_files": {name: f"{name}.json" for name in component_names},
        "component_feature_columns": {name: experiment_columns[name] for name in component_names},
        "categorical_features": all_cat,
        "categories": category_map,
        "numeric_medians": numeric_medians,
        "business_stats": stats.reset_index().astype({"business": str}).to_dict(orient="records"),
        "global_mean": global_mean,
        "training_target": "cum_cost_7d",
        "planned_date_origin": "2026-01-01",
        "source_node_file": NODES.name,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(artifact_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = importance.head(20)
    lines = [
        "# 矿主成本预测优化报告", "", "## 结论", "",
        f"- 任务：节点 × 业务 → 预测上线后 7 天矿主成本 `cum_cost_7d`。",
        f"- 模型选择仅使用验证集 MSE；测试集只做最终报告。",
        f"- 最终模型：`{final_name}`。",
        f"- 测试 MSE：{payload['selected_test_metrics']['mse']:.6f}；RMSE：{payload['selected_test_metrics']['rmse']:.6f}；MAE：{payload['selected_test_metrics']['mae']:.6f}；R²：{payload['selected_test_metrics']['r2']:.6f}。",
        "", "## 数据边界", "",
        "- 原始宽表存在计费类型、单价、计量值、保底比例和实际95带宽，但当前本地节点×业务训练文件没有落盘这些列。",
        "- 本轮使用训练折外业务成本统计模拟业务计费先验，并比较上线前节点带宽快照是否有增益；没有把目标窗口内真实跑量作为输入，避免直接泄漏最终成本。",
        "", "## 各方案", "",
        "| 方案 | 验证MSE | 测试MSE | 测试RMSE | 测试MAE | 测试R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        lines.append(f"| {name} | {row['validation']['mse']:.6f} | {row['test']['mse']:.6f} | {row['test']['rmse']:.6f} | {row['test']['mae']:.6f} | {row['test']['r2']:.6f} |")
    lines += ["", "## 前20项特征重要性", "", "| 特征 | Gain重要性 |", "|---|---:|"]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.gain_importance:.6f} |")
    lines += [
        "", "## 下一步进入生产前必须补的数据", "",
        "1. 将 `cost_priceType`、`cost_price`、`cost_priceAfterBonus`、`cost_guaranteedRate`、`measureBase`、`measureCoefficient` 按节点×业务×合同有效期落盘。",
        "2. 将实际结算口径的带宽计量值和单位统一后落盘；`peak95` 是 bps，建设带宽通常是 Mbps，不能直接混用。",
        "3. 上线前推荐场景使用预测跑量；结算预测场景才使用已经发生的真实跑量。",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_model": final_name, "test": payload["selected_test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
