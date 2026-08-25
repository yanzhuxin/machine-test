#!/usr/bin/env python3
"""第三轮：用其余28.6万条部分窗口构造节点/业务历史日均特征，不伪造7天标签。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from optimize_counterfactual_blends import ARTIFACT_DIR, candidate_predictions, convex_l1
from run_counterfactual_matrix_completion import (
    HERE, OUTCOMES, PREDICTIONS_OUT, TARGETS, attach_history, load_frame, masked_split, metrics,
)
from run_cost_regression_optimization import (
    CAT_FEATURES, PRE_RUN_BW_FEATURES, STATIC_NUM_FEATURES,
    model_config, prepare_categories, prepare_numeric,
)


BASE_METRICS = HERE / "counterfactual_constrained_blend_metrics.json"
OUT = HERE / "counterfactual_history_augmentation_metrics.json"
HISTORY_FEATURES = [
    "allhist_node_cost_daily_mean", "allhist_node_revenue_daily_mean", "allhist_node_pair_count_log1p",
    "allhist_business_cost_daily_mean", "allhist_business_revenue_daily_mean", "allhist_business_pair_count_log1p",
]


def add_all_history(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(OUTCOMES, usecols=["node_id", "business", "outcome_days", "cum_cost_7d", "cum_revenue_7d"],
                      dtype={"node_id": str, "business": str})
    raw = raw[raw.outcome_days > 0].copy()
    raw["cost_daily"] = raw.cum_cost_7d / raw.outcome_days
    raw["revenue_daily"] = raw.cum_revenue_7d / raw.outcome_days
    held_keys = pd.MultiIndex.from_frame(valid[["node_id", "business"]])
    raw_keys = pd.MultiIndex.from_frame(raw[["node_id", "business"]])
    source = raw[~raw_keys.isin(held_keys)].copy()

    node_stats = source.groupby("node_id", observed=True).agg(
        hist_count=("business", "size"), hist_cost_sum=("cost_daily", "sum"),
        hist_revenue_sum=("revenue_daily", "sum"),
    )
    business_stats = source.groupby("business", observed=True).agg(
        hist_count=("node_id", "size"), hist_cost_sum=("cost_daily", "sum"),
        hist_revenue_sum=("revenue_daily", "sum"),
    )
    tr, va = train.copy(), valid.copy()
    # 训练行leave-one-pair-out；验证业务已经从source整体排除。
    current_cost_daily = tr.cum_cost_7d / 7.0
    current_revenue_daily = tr.cum_revenue_7d / 7.0
    for frame, is_train in [(tr, True), (va, False)]:
        ns = frame.node_id.map(node_stats.hist_cost_sum).fillna(0.0)
        nr = frame.node_id.map(node_stats.hist_revenue_sum).fillna(0.0)
        nc = frame.node_id.map(node_stats.hist_count).fillna(0.0)
        bs = frame.business.map(business_stats.hist_cost_sum).fillna(0.0)
        br = frame.business.map(business_stats.hist_revenue_sum).fillna(0.0)
        bc = frame.business.map(business_stats.hist_count).fillna(0.0)
        if is_train:
            ns, nr, nc = ns - current_cost_daily, nr - current_revenue_daily, nc - 1
            bs, br, bc = bs - current_cost_daily, br - current_revenue_daily, bc - 1
        frame[HISTORY_FEATURES[0]] = ns / np.maximum(nc, 1)
        frame[HISTORY_FEATURES[1]] = nr / np.maximum(nc, 1)
        frame[HISTORY_FEATURES[2]] = np.log1p(np.maximum(nc, 0))
        frame[HISTORY_FEATURES[3]] = bs / np.maximum(bc, 1)
        frame[HISTORY_FEATURES[4]] = br / np.maximum(bc, 1)
        frame[HISTORY_FEATURES[5]] = np.log1p(np.maximum(bc, 0))
    audit = {"history_source_rows": len(source), "excluded_validation_pairs": len(valid),
             "history_source_nodes": source.node_id.nunique(), "history_source_businesses": source.business.nunique()}
    return tr, va, audit


def main() -> None:
    frame = load_frame()
    base_train, base_valid, split = masked_split(frame)
    artifacts = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    previous = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
    output = base_valid[["node_id", "business", "business_online_day", *TARGETS.values()]].copy()
    results = {"split": split, "targets": {}}
    for short, target in TARGETS.items():
        base_candidates = candidate_predictions(base_train, base_valid, target, short, artifacts[short])
        train, valid = attach_history(base_train, base_valid, target, short)
        train, valid, audit = add_all_history(train, valid)
        cats = ["node_id"] + [c for c in CAT_FEATURES if c in train]
        prepare_categories([train, valid], cats)
        target_history = [f"{short}_node_prior", f"{short}_node_known_count_log1p",
                          f"{short}_business_prior", f"{short}_business_known_count_log1p"]
        numeric = ([c for c in STATIC_NUM_FEATURES + PRE_RUN_BW_FEATURES if c in train]
                   + target_history + HISTORY_FEATURES + ["planned_start_month", "planned_start_day_index"])
        numeric_expanded, _ = prepare_numeric([train, valid], list(dict.fromkeys(numeric)), train)
        features = cats + numeric_expanded
        y = valid[target].to_numpy(float)
        new_predictions = {}
        for name, config in [("history_xgb_raw4", "xgb_raw_depth4_regularized"),
                             ("history_xgb_raw8", "xgb_raw_depth8"),
                             ("history_xgb_log1p", "xgb_log1p"),
                             ("history_xgb_tweedie", "xgb_tweedie_1_3")]:
            print(f"{short}: fitting {name}", flush=True)
            log_target = config == "xgb_log1p"
            y_train = np.log1p(train[target]) if log_target else train[target]
            y_valid = np.log1p(valid[target]) if log_target else valid[target]
            model = XGBRegressor(**model_config(config))
            model.fit(train[features], y_train, eval_set=[(valid[features], y_valid)], verbose=False)
            pred = model.predict(valid[features])
            if log_target:
                pred = np.expm1(pred)
            new_predictions[name] = np.clip(pred, 0.0, None)
            model.save_model(ARTIFACT_DIR / f"{short}_{name}.json")
        candidates = {**base_candidates, **new_predictions}
        names = list(candidates)
        matrix = np.column_stack([candidates[name] for name in names])
        weights = convex_l1(y, matrix)
        blend = matrix @ weights
        variants = {name: metrics(y, pred) for name, pred in candidates.items()}
        variants["history_wape_constrained_blend"] = metrics(y, blend)
        best_wape = min(row["wape"] for row in variants.values())
        eligible = [name for name, row in variants.items() if row["wape"] <= best_wape + .01]
        selected = min(eligible, key=lambda name: variants[name]["mse"])
        selected_pred = blend if selected == "history_wape_constrained_blend" else candidates[selected]
        output[f"predicted_{short}_7d"] = np.clip(selected_pred, 0.0, None)
        results["targets"][short] = {
            "history_audit": audit,
            "previous_recommended": previous["targets"][short]["recommended_metrics"],
            "variants": variants,
            "blend_weights": {name: float(w) for name, w in zip(names, weights) if w > 1e-7},
            "recommended_model": selected, "recommended_metrics": variants[selected],
        }
    output = output.rename(columns={"cum_cost_7d": "actual_cost_7d", "cum_revenue_7d": "actual_revenue_7d"})
    output["actual_profit_7d"] = output.actual_revenue_7d - output.actual_cost_7d
    output["predicted_profit_7d"] = output.predicted_revenue_7d - output.predicted_cost_7d
    output.to_csv(PREDICTIONS_OUT, index=False)
    results["profit_metrics"] = metrics(output.actual_profit_7d, output.predicted_profit_7d, clip=False)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (HERE / "反事实金额矩阵补全优化报告.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## 第三轮：部分窗口历史信号增强\n\n")
        handle.write("不把1～6天窗口放大成伪7天标签，只把其他节点×业务的日均表现作为历史侧信息。\n\n")
        for short, label in [("cost", "成本"), ("revenue", "收入")]:
            row = results["targets"][short]
            m = row["recommended_metrics"]
            handle.write(f"- {label}推荐：`{row['recommended_model']}`；MSE {m['mse']:.2f}，MAE {m['mae']:.2f}，R² {m['r2']:.4f}，WAPE {m['wape']:.2%}，金额≥10 WAPE {m['amount_ge_10_wape']:.2%}。\n")
    print(json.dumps({short: {"model": row["recommended_model"], **row["recommended_metrics"]}
                      for short, row in results["targets"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
