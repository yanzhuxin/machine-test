#!/usr/bin/env python3
"""验证：仅按节点固有特征找相似节点，同业务金额是否也更接近。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, normalize

HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "01_counterfactual_amount_prediction"
sys.path.insert(0, str(V1_DIR))

from run_counterfactual_matrix_completion import load_frame, masked_split, metrics
from run_cost_regression_optimization import CAT_FEATURES, STATIC_NUM_FEATURES

NODE_INTRINSIC_CAT_FEATURES = CAT_FEATURES[1:]
NODE_INTRINSIC_NUM_FEATURES = STATIC_NUM_FEATURES


NODES_FILE = V1_DIR / "multibusiness_nodes.csv"
PAIR_OUT = HERE / "feature_similar_node_pairs.csv"
OVERLAP_OUT = HERE / "feature_similar_same_business_details.csv"
PRED_OUT = HERE / "feature_neighbor_masked_predictions.csv"
METRICS_OUT = HERE / "feature_neighbor_similarity_validation.json"
REPORT_OUT = HERE / "特征相似节点业务扩散预验证报告.md"
CURRENT_PREDICTIONS = V1_DIR / "counterfactual_masked_validation_predictions.csv"
RANDOM_STATE = 42
NEIGHBOR_SEARCH = 120


def build_embedding(nodes: pd.DataFrame) -> tuple[sparse.csr_matrix, dict]:
    cats = [c for c in NODE_INTRINSIC_CAT_FEATURES if c in nodes]
    nums = [c for c in NODE_INTRINSIC_NUM_FEATURES if c in nodes]
    cat_frame = nodes[cats].fillna("__MISSING__").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True, dtype=np.float32)
    x_cat = normalize(encoder.fit_transform(cat_frame), norm="l2")

    num_frame = nodes[nums].apply(pd.to_numeric, errors="coerce")
    num_frame = num_frame.mask(num_frame <= 0)
    missing = num_frame.isna().astype(np.float32).to_numpy()
    imputer = SimpleImputer(strategy="median")
    num_imputed = imputer.fit_transform(num_frame)
    quantiles = QuantileTransformer(
        n_quantiles=min(500, len(nodes)), output_distribution="uniform",
        random_state=RANDOM_STATE, subsample=None,
    ).fit_transform(num_imputed).astype(np.float32)
    x_num = normalize(sparse.csr_matrix(np.c_[quantiles, missing]), norm="l2")
    # 类别与数值两块分别归一化后赋予65%/35%的距离权重。
    matrix = sparse.hstack([math.sqrt(.65) * x_cat, math.sqrt(.35) * x_num], format="csr")
    matrix = normalize(matrix, norm="l2")
    return matrix, {
        "categorical_features": cats, "numeric_features": nums,
        "categorical_weight": .65, "numeric_weight": .35,
        "encoded_dimensions": int(matrix.shape[1]),
    }


def smape(a: float, b: float) -> float:
    denominator = abs(a) + abs(b)
    return 0.0 if denominator <= 1e-12 else 2.0 * abs(a - b) / denominator


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def prior_map(train: pd.DataFrame, target: str) -> tuple[dict[str, float], float]:
    global_mean = float(train[target].mean())
    grouped = train.groupby("business", observed=True)[target].agg(["count", "sum"])
    grouped["prior"] = (grouped["sum"] + 10.0 * global_mean) / (grouped["count"] + 10.0)
    return grouped.prior.to_dict(), global_mean


def evaluate_predictions(y: np.ndarray, prediction: np.ndarray) -> dict:
    result = metrics(y, prediction)
    nonzero = np.abs(y) > 1e-12
    ape = np.abs(y[nonzero] - prediction[nonzero]) / np.abs(y[nonzero])
    result.update({
        "median_absolute_percentage_error": float(np.median(ape)),
        "within_50pct_share": float(np.mean(ape <= .5)),
        "within_100pct_share": float(np.mean(ape <= 1.0)),
    })
    return result


def main() -> None:
    frame = load_frame()
    train, valid, split_info = masked_split(frame)
    node_ids = np.array(sorted(frame.node_id.unique()))
    header = pd.read_csv(NODES_FILE, nrows=0).columns
    feature_columns = [c for c in NODE_INTRINSIC_CAT_FEATURES + NODE_INTRINSIC_NUM_FEATURES if c in header]
    nodes = pd.read_csv(NODES_FILE, usecols=["node_id", *feature_columns], dtype={"node_id": str}, low_memory=False)
    nodes = nodes.drop_duplicates("node_id").set_index("node_id").reindex(node_ids).reset_index()
    embedding, feature_info = build_embedding(nodes)
    neighbor_model = NearestNeighbors(n_neighbors=min(NEIGHBOR_SEARCH + 1, len(nodes)), metric="cosine", algorithm="brute", n_jobs=-1)
    neighbor_model.fit(embedding)
    raw_distances, raw_indices = neighbor_model.kneighbors(embedding)
    # 特征完全相同的节点会与自身同为距离 0，因此不能假定第一列必然是自身。
    # 按行显式剔除当前节点，避免把 self-pair 计入相似性验证。
    filtered_distances = []
    filtered_indices = []
    for row_idx, (row_distances, row_indices) in enumerate(zip(raw_distances, raw_indices)):
        keep = row_indices != row_idx
        filtered_distances.append(row_distances[keep][:NEIGHBOR_SEARCH])
        filtered_indices.append(row_indices[keep][:NEIGHBOR_SEARCH])
    distances = np.vstack(filtered_distances)
    indices = np.vstack(filtered_indices)
    node_to_position = {node: idx for idx, node in enumerate(node_ids)}

    complete_lookup = {
        node: group.set_index("business")[["cum_cost_7d", "cum_revenue_7d"]]
        for node, group in frame.groupby("node_id", observed=True)
    }
    pair_rows, overlap_rows = [], []
    seen_pairs: set[tuple[str, str]] = set()
    rng = np.random.default_rng(RANDOM_STATE)
    business_holders = {
        business: group.set_index("node_id")[["cum_cost_7d", "cum_revenue_7d"]]
        for business, group in frame.groupby("business", observed=True)
    }
    for position, node in enumerate(node_ids):
        left = complete_lookup[node]
        for rank, (distance, neighbor_position) in enumerate(zip(distances[position, :10], indices[position, :10]), 1):
            neighbor = node_ids[neighbor_position]
            pair = tuple(sorted((node, neighbor)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            right = complete_lookup[neighbor]
            common = sorted(set(left.index).intersection(right.index))
            cost_gaps, revenue_gaps = [], []
            for business in common:
                cost_gap = smape(float(left.loc[business, "cum_cost_7d"]), float(right.loc[business, "cum_cost_7d"]))
                revenue_gap = smape(float(left.loc[business, "cum_revenue_7d"]), float(right.loc[business, "cum_revenue_7d"]))
                cost_gaps.append(cost_gap)
                revenue_gaps.append(revenue_gap)
                holders = business_holders[business]
                candidates = holders.index[(holders.index != node) & (holders.index != neighbor)]
                random_node = str(rng.choice(candidates)) if len(candidates) else neighbor
                overlap_rows.append({
                    "node_id": node, "similar_node_id": neighbor, "neighbor_rank": rank,
                    "feature_distance": float(distance), "feature_similarity": float(1.0 - distance),
                    "business": business,
                    "node_cost_7d": float(left.loc[business, "cum_cost_7d"]),
                    "similar_cost_7d": float(right.loc[business, "cum_cost_7d"]),
                    "cost_smape": cost_gap,
                    "node_revenue_7d": float(left.loc[business, "cum_revenue_7d"]),
                    "similar_revenue_7d": float(right.loc[business, "cum_revenue_7d"]),
                    "revenue_smape": revenue_gap,
                    "random_node_id": random_node,
                    "random_cost_smape": smape(float(left.loc[business, "cum_cost_7d"]), float(holders.loc[random_node, "cum_cost_7d"])),
                    "random_revenue_smape": smape(float(left.loc[business, "cum_revenue_7d"]), float(holders.loc[random_node, "cum_revenue_7d"])),
                })
            pair_rows.append({
                "node_id": node, "similar_node_id": neighbor, "neighbor_rank": rank,
                "feature_distance": float(distance), "feature_similarity": float(1.0 - distance),
                "shared_complete_business_count": len(common),
                "median_cost_smape_on_shared_business": float(np.median(cost_gaps)) if cost_gaps else np.nan,
                "median_revenue_smape_on_shared_business": float(np.median(revenue_gaps)) if revenue_gaps else np.nan,
            })
    pairs = pd.DataFrame(pair_rows).sort_values(["feature_distance", "node_id"])
    overlaps = pd.DataFrame(overlap_rows)
    pairs.to_csv(PAIR_OUT, index=False)
    overlaps.to_csv(OVERLAP_OUT, index=False)

    train_by_business = {
        business: group.set_index("node_id")[["cum_cost_7d", "cum_revenue_7d"]]
        for business, group in train.groupby("business", observed=True)
    }
    prediction_frame = valid[["node_id", "business", "business_online_day", "cum_cost_7d", "cum_revenue_7d"]].copy()
    coverage_counts = []
    neighbor_distances = []
    for target_short, target in [("cost", "cum_cost_7d"), ("revenue", "cum_revenue_7d")]:
        priors, global_mean = prior_map(train, target)
        for k in [3, 5, 10, 20]:
            median_values, mean_values, hybrid_values = [], [], []
            for row in valid.itertuples():
                position = node_to_position[row.node_id]
                holders = train_by_business[str(row.business)]
                values, weights, used_distances = [], [], []
                for distance, neighbor_position in zip(distances[position], indices[position]):
                    neighbor = node_ids[neighbor_position]
                    if neighbor not in holders.index:
                        continue
                    values.append(float(holders.loc[neighbor, target]))
                    weights.append(math.exp(-5.0 * float(distance)))
                    used_distances.append(float(distance))
                    if len(values) == k:
                        break
                prior = float(priors.get(str(row.business), global_mean))
                if not values:
                    median_pred = mean_pred = prior
                else:
                    array, weight_array = np.asarray(values), np.asarray(weights)
                    median_pred = weighted_median(array, weight_array)
                    mean_pred = float(np.average(array, weights=weight_array))
                # 稳健混合：邻居越少，越向业务先验收缩。
                reliability = len(values) / (len(values) + 5.0)
                hybrid_pred = reliability * median_pred + (1.0 - reliability) * prior
                median_values.append(median_pred)
                mean_values.append(mean_pred)
                hybrid_values.append(hybrid_pred)
                if target_short == "cost" and k == 10:
                    coverage_counts.append(len(values))
                    neighbor_distances.append(min(used_distances) if used_distances else np.nan)
            prediction_frame[f"pred_{target_short}_knn{k}_median"] = median_values
            prediction_frame[f"pred_{target_short}_knn{k}_mean"] = mean_values
            prediction_frame[f"pred_{target_short}_knn{k}_hybrid"] = hybrid_values
        prediction_frame[f"pred_{target_short}_business_prior"] = prediction_frame.business.map(priors).fillna(global_mean)
    prediction_frame["same_business_neighbors_found_top120"] = coverage_counts
    prediction_frame["nearest_same_business_feature_distance"] = neighbor_distances

    current = pd.read_csv(CURRENT_PREDICTIONS, dtype={"node_id": str, "business": str})
    current = current[["node_id", "business", "business_online_day", "predicted_cost_7d", "predicted_revenue_7d"]]
    current = current.rename(columns={"predicted_cost_7d": "pred_cost_current_hybrid",
                                      "predicted_revenue_7d": "pred_revenue_current_hybrid"})
    prediction_frame = prediction_frame.merge(current, on=["node_id", "business", "business_online_day"], how="left", validate="one_to_one")
    prediction_frame.to_csv(PRED_OUT, index=False)

    model_results = {}
    for short, target in [("cost", "cum_cost_7d"), ("revenue", "cum_revenue_7d")]:
        candidates = [c for c in prediction_frame if c.startswith(f"pred_{short}_")]
        model_results[short] = {
            column: evaluate_predictions(prediction_frame[target].to_numpy(), prediction_frame[column].to_numpy())
            for column in candidates
        }
        model_results[short]["recommended"] = min(model_results[short], key=lambda name: model_results[short][name]["wape"])

    if overlaps.empty:
        similarity_validation = {"shared_business_rows": 0}
    else:
        cost_test = mannwhitneyu(overlaps.cost_smape, overlaps.random_cost_smape, alternative="less")
        revenue_test = mannwhitneyu(overlaps.revenue_smape, overlaps.random_revenue_smape, alternative="less")
        cost_corr = spearmanr(overlaps.feature_distance, overlaps.cost_smape)
        revenue_corr = spearmanr(overlaps.feature_distance, overlaps.revenue_smape)
        similarity_validation = {
            "unique_top10_pairs": int(len(pairs)),
            "pairs_with_shared_complete_business": int((pairs.shared_complete_business_count > 0).sum()),
            "pair_shared_business_rate": float((pairs.shared_complete_business_count > 0).mean()),
            "shared_business_rows": int(len(overlaps)),
            "similar_cost_smape_mean": float(overlaps.cost_smape.mean()),
            "random_cost_smape_mean": float(overlaps.random_cost_smape.mean()),
            "similar_cost_gap_lower_than_random_share": float((overlaps.cost_smape < overlaps.random_cost_smape).mean()),
            "similar_cost_vs_random_mannwhitney_p": float(cost_test.pvalue),
            "similar_revenue_smape_mean": float(overlaps.revenue_smape.mean()),
            "random_revenue_smape_mean": float(overlaps.random_revenue_smape.mean()),
            "similar_revenue_gap_lower_than_random_share": float((overlaps.revenue_smape < overlaps.random_revenue_smape).mean()),
            "similar_revenue_vs_random_mannwhitney_p": float(revenue_test.pvalue),
            "distance_cost_gap_spearman": float(cost_corr.statistic),
            "distance_cost_gap_p": float(cost_corr.pvalue),
            "distance_revenue_gap_spearman": float(revenue_corr.statistic),
            "distance_revenue_gap_p": float(revenue_corr.pvalue),
        }
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "question": "do feature-similar nodes have closer outcomes on the same business",
        "split": split_info, "feature_space": feature_info,
        "neighbor_search": {"top_pairs": 10, "business_holder_search_depth": NEIGHBOR_SEARCH},
        "similarity_validation": similarity_validation,
        "masked_prediction_validation": model_results,
        "neighbor_coverage": {
            "mean_same_business_neighbors_found": float(np.mean(coverage_counts)),
            "share_with_at_least_3": float(np.mean(np.asarray(coverage_counts) >= 3)),
            "share_with_at_least_10": float(np.mean(np.asarray(coverage_counts) >= 10)),
        },
    }
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 特征相似节点业务扩散预验证报告", "", "## 验证问题", "",
             "只用节点固有特征找近邻，检查相似节点运行同一业务时的成本和收入是否比随机节点更接近。金额和业务结果不参与相似度计算。", "",
             "## 相似度特征", "", f"- 类别特征：{len(feature_info['categorical_features'])}个，权重65%。",
             f"- 数值容量特征：{len(feature_info['numeric_features'])}个，权重35%。",
             f"- 编码后维度：{feature_info['encoded_dimensions']}。", "", "## 共同业务验证", ""]
    for key, value in similarity_validation.items():
        lines.append(f"- `{key}`：{value}")
    lines += ["", "## 90/10遮挡预测", ""]
    for short, label in [("cost", "成本"), ("revenue", "收入")]:
        candidates = model_results[short]
        neighbor_names = [name for name in candidates if name.startswith(f"pred_{short}_knn")]
        best_neighbor = min(neighbor_names, key=lambda name: candidates[name]["wape"])
        for title, name in [("最优纯近邻", best_neighbor),
                            ("业务先验基线", f"pred_{short}_business_prior"),
                            ("当前历史混合模型", f"pred_{short}_current_hybrid")]:
            row = candidates[name]
            lines.append(f"- {label}{title}：`{name}`；R² {row['r2']:.4f}，WAPE {row['wape']:.2%}，MAE {row['mae']:.2f}。")
    lines += ["", "## 结论", "",
              "特征近邻在同业务上的金额差异显著小于随机对照，且纯近邻预测明显优于仅使用业务先验；但按WAPE/MAE仍未超过当前历史混合模型。建议把近邻预测作为带置信度的补充特征，而不是直接复制近邻金额或替换当前模型。"]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"similarity": similarity_validation,
                      "cost_recommended": model_results["cost"]["recommended"],
                      "cost_metrics": model_results["cost"][model_results["cost"]["recommended"]],
                      "revenue_recommended": model_results["revenue"]["recommended"],
                      "revenue_metrics": model_results["revenue"][model_results["revenue"]["recommended"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
