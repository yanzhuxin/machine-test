from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from xgboost import XGBRegressor

from predict_similar_node_v3 import NEIGHBOR_FEATURES, SimilarNodePredictor


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "01a0225a-43b2-7991-a2c3-00e9e4064c5d"
ARTIFACT = OUT / "matched_similar_node_v3_model_artifact"
DAILY = OUT / "aggregation_dedicated_daily_outcomes_last30d.csv"
AMOUNT_TOP3 = OUT / "v3_recent7d_amount_profit_top3.csv"
DIRECT_TOP3 = OUT / "v3_recent7d_direct_profit_top3.csv"
COMBINED_TOP5 = OUT / "v3_recent7d_combined_top5.csv"
SUMMARY = OUT / "v3_recent7d_recommendation_summary.json"
KS = (3, 5, 10, 20)

SIM_CAT_FEATURES = [
    "attribute_delivery_type", "resourcetype", "dialtype", "nattype", "scheduleisps",
    "regsource", "customermode", "province", "isp", "device_type", "arch_type", "isvm",
    "hardwaretype", "analysis_nodecustmertype", "analysis_nodedeliverytype",
    "analysis_tcpnattype", "analysis_udpnattype", "analysis_cooperationtype",
    "analysis_supply_side_delivery_type", "analysis_isroot", "analysis_issupportipv6",
]
SIM_NUM_FEATURES = [
    "bw", "corenum", "memtotal", "totaldisksize", "hdddisksize", "ssddisksize",
    "actualbandwidth", "netbenchlimitbandwidth", "analysis_yesterday_p95_bw",
    "analysis_yesterday_snapshot_bw", "analysis_yesterday_snapshot_netbench_bw",
    "analysis_dby_p95_bw", "join_actualbw", "join_limitbw", "join_cpu_totalcores",
    "join_memtotal", "join_disks_total_size",
]


def rebuild_similarity_matrix(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """Rebuild the deterministic similarity transform when a joblib NumPy ABI differs."""
    nodes = frame.drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)
    cats = [column for column in SIM_CAT_FEATURES if column in nodes]
    nums = [column for column in SIM_NUM_FEATURES if column in nodes]
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, dtype=np.float32)
    cat_matrix = encoder.fit_transform(nodes[cats].fillna("__MISSING__").astype(str))
    numeric = np.log1p(nodes[nums].apply(pd.to_numeric, errors="coerce").clip(lower=0))
    numeric = SimpleImputer(strategy="median").fit_transform(numeric)
    numeric = np.clip(StandardScaler().fit_transform(numeric), -4, 4).astype(np.float32)
    matrix = sparse.hstack([cat_matrix * 1.25, sparse.csr_matrix(numeric) * 0.55], format="csr")
    matrix = normalize(matrix, axis=1).tocsr()
    return matrix, dict(zip(nodes["node_id"].astype(str), range(len(nodes))))


def load_predictor() -> SimilarNodePredictor:
    try:
        return SimilarNodePredictor(ARTIFACT)
    except ModuleNotFoundError as error:
        if "numpy._core" not in str(error):
            raise
        predictor = SimilarNodePredictor.__new__(SimilarNodePredictor)
        predictor.root = ARTIFACT
        predictor.manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
        predictor.nodes = pd.read_csv(
            ARTIFACT / predictor.manifest["files"]["node_reference"], dtype={"node_id": str}, low_memory=False
        ).sort_values("node_id").reset_index(drop=True)
        predictor.amounts = pd.read_csv(
            ARTIFACT / predictor.manifest["files"]["amount_reference"],
            dtype={"node_id": str, "business": str}, low_memory=False,
        )
        predictor.node_rows = predictor.nodes.set_index("node_id", drop=False)
        predictor.node_matrix, predictor.node_position = rebuild_similarity_matrix(predictor.nodes)
        predictor.models = {}
        for name, spec in predictor.manifest["models"].items():
            model = XGBRegressor()
            model.load_model(ARTIFACT / spec["file"])
            predictor.models[name] = model
        predictor.business_metadata = predictor.amounts.groupby("business", observed=True).agg(
            business_name=("business_name", "first"), outcome_delivery_type=("outcome_delivery_type", "first")
        )
        return predictor


def weighted_summary(values: np.ndarray, similarities: np.ndarray, k: int) -> tuple[float, float, int]:
    take = min(k, len(values))
    if take == 0:
        return np.nan, np.nan, 0
    values = values[:take]
    similarities = similarities[:take]
    weights = np.maximum(similarities, 0.0) ** 4
    if weights.sum() <= 1e-12:
        weights = np.ones(take)
    mean = float(np.average(values, weights=weights))
    std = float(np.sqrt(max(np.average((values - mean) ** 2, weights=weights), 0.0)))
    return mean, std, take


def attach_neighbor_features(
    predictor: SimilarNodePredictor,
    query: pd.DataFrame,
    business: str,
    target: str,
) -> None:
    reference = predictor.amounts[
        predictor.amounts["business"].eq(business) & predictor.amounts[target].notna()
    ].drop_duplicates("node_id")
    for column in NEIGHBOR_FEATURES:
        query[column] = np.nan
    if reference.empty:
        return
    reference = reference[reference["node_id"].isin(predictor.node_position)].copy()
    candidate_nodes = reference["node_id"].astype(str).to_numpy()
    candidate_rows = np.array([predictor.node_position[n] for n in candidate_nodes], dtype=int)
    query_nodes = query["node_id"].astype(str).to_numpy()
    query_rows = np.array([predictor.node_position[n] for n in query_nodes], dtype=int)
    similarities = (predictor.node_matrix[query_rows] @ predictor.node_matrix[candidate_rows].T).toarray()
    similarities[query_nodes[:, None] == candidate_nodes[None, :]] = -np.inf
    top_n = min(20, similarities.shape[1])
    if top_n == 0:
        return
    positions = np.argpartition(-similarities, kth=top_n - 1, axis=1)[:, :top_n]
    top_sims = np.take_along_axis(similarities, positions, axis=1)
    order = np.argsort(-top_sims, axis=1)
    positions = np.take_along_axis(positions, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    candidate_values = reference[target].to_numpy(float)
    output = {column: np.full(len(query), np.nan, dtype=float) for column in NEIGHBOR_FEATURES}
    for row in range(len(query)):
        finite = np.isfinite(top_sims[row])
        sims = top_sims[row, finite]
        values = candidate_values[positions[row, finite]]
        if len(values) == 0:
            continue
        for k in KS:
            output[f"neighbor_prior_k{k}"][row] = weighted_summary(values, sims, k)[0]
        _, std10, support10 = weighted_summary(values, sims, 10)
        output["neighbor_support"][row] = support10
        output["neighbor_similarity_max"][row] = sims[0]
        output["neighbor_similarity_mean"][row] = np.mean(sims[:support10])
        output["neighbor_std_k10"][row] = std10
    for column, values in output.items():
        query[column] = values


def attach_amount_priors(
    predictor: SimilarNodePredictor,
    query: pd.DataFrame,
    business: str,
    target: str,
    weighted: bool,
) -> None:
    short = target.split("_")[0]
    reliability_column = f"{short}_reliability_weight"
    reference = predictor.amounts[predictor.amounts[target].notna()].copy()
    y = reference[target].to_numpy(float)
    reliability = pd.to_numeric(reference[reliability_column], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    global_sum = float(np.sum(reliability * y))
    global_weight = float(np.sum(reliability))

    # Existing node-business observations are removed row by row, matching the
    # counterfactual masking used during validation and avoiding target leakage.
    pair = reference.set_index(["node_id", "business"])
    own = np.array([
        float(pair.at[(node, business), target]) if (node, business) in pair.index else np.nan
        for node in query["node_id"].astype(str)
    ])
    own_weight = np.array([
        float(pair.at[(node, business), reliability_column]) if (node, business) in pair.index else 0.0
        for node in query["node_id"].astype(str)
    ])
    has_own = np.isfinite(own)
    own_weight = np.where(has_own, np.maximum(own_weight, 0.0), 0.0)
    global_mean = (global_sum - np.where(has_own, own * own_weight, 0.0)) / np.maximum(global_weight - own_weight, 1e-9)

    def stats(column: str, value: str | np.ndarray, alpha: float):
        if weighted:
            temp = reference.assign(_wy=reliability * y, _wy2=reliability * y * y, _w=reliability)
            grouped = temp.groupby(column, observed=True).agg(s=("_wy", "sum"), s2=("_wy2", "sum"), n=("_w", "sum"))
        else:
            temp = reference.assign(_y2=y * y)
            grouped = temp.groupby(column, observed=True).agg(s=(target, "sum"), s2=("_y2", "sum"), n=(target, "count"))
        keys = np.repeat(str(value), len(query)) if np.isscalar(value) else np.asarray(value).astype(str)
        sums = pd.Series(keys).map(grouped["s"]).fillna(0).to_numpy(float)
        sums2 = pd.Series(keys).map(grouped["s2"]).fillna(0).to_numpy(float)
        counts = pd.Series(keys).map(grouped["n"]).fillna(0).to_numpy(float)
        subtract = own_weight if weighted else has_own.astype(float)
        sums = sums - np.where(has_own, own * subtract, 0.0)
        sums2 = sums2 - np.where(has_own, own * own * subtract, 0.0)
        counts = np.maximum(counts - subtract, 0.0)
        prior = (sums + alpha * global_mean) / np.maximum(counts + alpha, 1e-9)
        raw_mean = sums / np.maximum(counts, 1e-9)
        variance = np.maximum(sums2 / np.maximum(counts, 1e-9) - raw_mean * raw_mean, 0.0)
        return prior, np.log1p(counts), np.sqrt(variance)

    business_prior, business_support, business_std = stats("business", business, 12.0)
    node_prior, node_support, node_std = stats("node_id", query["node_id"].astype(str).to_numpy(), 2.0)
    query["business_prior"] = business_prior
    query["business_support_log1p"] = business_support
    query["business_std"] = business_std
    query["node_prior"] = node_prior
    query["node_support_log1p"] = node_support
    query["node_std"] = node_std
    query["node_business_interaction_prior"] = business_prior * node_prior / np.maximum(global_mean, 1e-6)


def model_frame(predictor: SimilarNodePredictor, rows: pd.DataFrame, model_name: str) -> pd.DataFrame:
    spec = predictor.manifest["models"][model_name]
    output = rows.copy()
    for column in spec["feature_columns"]:
        if column not in output:
            output[column] = np.nan
    for column, levels in spec["categories"].items():
        output[column] = pd.Categorical(output[column].fillna("__MISSING__").astype(str), categories=levels)
    for column in (c for c in spec["feature_columns"] if c not in spec["categories"]):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float32)
    return output[spec["feature_columns"]]


def score_business(
    predictor: SimilarNodePredictor,
    nodes: pd.DataFrame,
    business: str,
    ranker_bundle: dict | None = None,
) -> pd.DataFrame:
    rows = nodes.copy()
    rows["business"] = business
    if business in predictor.business_metadata.index:
        rows["business_name"] = predictor.business_metadata.at[business, "business_name"]
        rows["outcome_delivery_type"] = predictor.business_metadata.at[business, "outcome_delivery_type"]
    else:
        rows["business_name"] = ""
        rows["outcome_delivery_type"] = "__MISSING__"

    attach_amount_priors(predictor, rows, business, "cost_target", weighted=False)
    cost_base = predictor.models["cost_base"].predict(model_frame(predictor, rows, "cost_base"))
    attach_neighbor_features(predictor, rows, business, "cost_target")
    cost_similar = predictor.models["cost_similar"].predict(model_frame(predictor, rows, "cost_similar"))

    attach_amount_priors(predictor, rows, business, "revenue_target", weighted=True)
    revenue_base = predictor.models["revenue_base"].predict(model_frame(predictor, rows, "revenue_base"))
    attach_neighbor_features(predictor, rows, business, "revenue_target")
    revenue_similar = predictor.models["revenue_similar"].predict(model_frame(predictor, rows, "revenue_similar"))

    attach_amount_priors(predictor, rows, business, "profit_target", weighted=False)
    attach_neighbor_features(predictor, rows, business, "profit_target")
    direct_profit = predictor.models["profit_similar"].predict(model_frame(predictor, rows, "profit_similar"))
    pairwise_score = np.full(len(rows), np.nan, dtype=float)
    revised_profit = direct_profit.copy()
    if ranker_bundle is not None:
        spec = ranker_bundle["spec"]
        external = rows.copy()
        for column in spec["columns"]:
            if column not in external:
                external[column] = np.nan
        for column, levels in spec["categories"].items():
            external[column] = pd.Categorical(external[column].fillna("__MISSING__").astype(str), categories=levels)
        for column in (c for c in spec["columns"] if c not in spec["categories"]):
            external[column] = pd.to_numeric(external[column], errors="coerce").astype(np.float32)
        pairwise_score = ranker_bundle["model"].predict(external[spec["columns"]])
        calibration = ranker_bundle["calibration"]
        standardized = (pairwise_score - calibration["pairwise_mean"]) / max(calibration["pairwise_std"], 1e-9)
        revised_profit = direct_profit + calibration["beta"] * calibration["direct_std"] * standardized

    cost_weight = predictor.manifest["prediction"]["cost_blend"]["similar_weight"]
    revenue_weight = predictor.manifest["prediction"]["revenue_blend"]["similar_weight"]
    predicted_cost = np.maximum(cost_weight * cost_similar + (1 - cost_weight) * cost_base, 0)
    predicted_revenue = np.maximum(revenue_weight * revenue_similar + (1 - revenue_weight) * revenue_base, 0)
    return pd.DataFrame({
        "node_id": rows["node_id"].astype(str),
        "business": business,
        "business_name": rows["business_name"],
        "attribute_delivery_type": rows["attribute_delivery_type"].astype(str),
        "outcome_delivery_type": rows["outcome_delivery_type"].astype(str),
        "predicted_cost": predicted_cost,
        "predicted_revenue": predicted_revenue,
        "predicted_amount_profit": predicted_revenue - predicted_cost,
        "direct_profit_score": direct_profit,
        "pairwise_profit_score": pairwise_score,
        "revised_profit_score": revised_profit,
        "profit_training_support": int(((predictor.amounts["business"] == business) & predictor.amounts["profit_target"].notna()).sum()),
    })


def top_n(frame: pd.DataFrame, score: str, n: int, rank_name: str) -> pd.DataFrame:
    result = frame.sort_values(["node_id", score, "business"], ascending=[True, False, True]).groupby("node_id", sort=False).head(n).copy()
    result[rank_name] = result.groupby("node_id", sort=False).cumcount() + 1
    columns = ["node_id", rank_name, "business", "business_name", "predicted_cost", "predicted_revenue", "predicted_amount_profit", "direct_profit_score", "profit_training_support"]
    return result[columns]


def combine_rankings(amount: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    left = amount.rename(columns={"amount_profit_rank": "amount_rank"})
    right = direct[["node_id", "business", "direct_profit_rank"]].rename(columns={"direct_profit_rank": "direct_rank"})
    merged = left.merge(right, on=["node_id", "business"], how="outer")
    missing = merged["business_name"].isna()
    if missing.any():
        details = direct.set_index(["node_id", "business"])
        for column in ["business_name", "predicted_cost", "predicted_revenue", "predicted_amount_profit", "direct_profit_score", "profit_training_support"]:
            keys = pd.MultiIndex.from_frame(merged.loc[missing, ["node_id", "business"]])
            merged.loc[missing, column] = details.reindex(keys)[column].to_numpy()
    merged["amount_points"] = np.where(merged["amount_rank"].notna(), 4 - merged["amount_rank"], 0)
    merged["direct_points"] = np.where(merged["direct_rank"].notna(), 4 - merged["direct_rank"], 0)
    merged["combined_points"] = merged["amount_points"] + merged["direct_points"]
    merged["source_count"] = merged[["amount_rank", "direct_rank"]].notna().sum(axis=1)
    merged["selected_by"] = np.select(
        [merged["source_count"].eq(2), merged["amount_rank"].notna()],
        ["both", "amount_profit"], default="direct_profit",
    )
    merged = merged.sort_values(
        ["node_id", "combined_points", "source_count", "direct_profit_score", "predicted_amount_profit", "business"],
        ascending=[True, False, False, False, False, True],
    ).groupby("node_id", sort=False).head(5).copy()
    merged["combined_rank"] = merged.groupby("node_id", sort=False).cumcount() + 1
    return merged[[
        "node_id", "combined_rank", "business", "business_name", "selected_by", "source_count",
        "amount_rank", "direct_rank", "amount_points", "direct_points", "combined_points",
        "predicted_cost", "predicted_revenue", "predicted_amount_profit", "direct_profit_score", "profit_training_support",
    ]]


def main() -> None:
    started = time.time()
    predictor = load_predictor()
    daily = pd.read_csv(DAILY, dtype={"nodeId": str, "customerId": str}, low_memory=False)
    daily["day"] = pd.to_datetime(daily["day"], errors="coerce")
    actual_days = sorted(daily["day"].dropna().dt.normalize().unique())[-7:]
    recent = daily[daily["day"].dt.normalize().isin(actual_days)].copy()
    activity = recent.groupby("nodeId", observed=True).agg(
        recent7d_active_days=("day", lambda s: s.dt.normalize().nunique()),
        recent7d_first_day=("day", "min"),
        recent7d_last_day=("day", "max"),
    ).reset_index().rename(columns={"nodeId": "node_id"})
    known = set(predictor.nodes["node_id"].astype(str))
    matched_activity = activity[activity["node_id"].isin(known)].copy()
    nodes = predictor.nodes[predictor.nodes["node_id"].isin(matched_activity["node_id"])].copy()
    businesses = sorted(predictor.amounts.loc[predictor.amounts["profit_target"].notna(), "business"].astype(str).unique())

    scored = []
    for index, business in enumerate(businesses, start=1):
        scored.append(score_business(predictor, nodes, business))
        if index % 5 == 0 or index == len(businesses):
            print(f"scored {index}/{len(businesses)} businesses", flush=True)
    scores = pd.concat(scored, ignore_index=True)
    amount = top_n(scores, "predicted_amount_profit", 3, "amount_profit_rank")
    direct = top_n(scores, "direct_profit_score", 3, "direct_profit_rank")
    combined = combine_rankings(amount, direct)

    activity_columns = ["node_id", "recent7d_active_days", "recent7d_first_day", "recent7d_last_day"]
    for result, path in [(amount, AMOUNT_TOP3), (direct, DIRECT_TOP3), (combined, COMBINED_TOP5)]:
        result = result.merge(matched_activity[activity_columns], on="node_id", how="left")
        result["recent7d_first_day"] = result["recent7d_first_day"].dt.strftime("%Y-%m-%d")
        result["recent7d_last_day"] = result["recent7d_last_day"].dt.strftime("%Y-%m-%d")
        result.to_csv(path, index=False, encoding="utf-8-sig")

    union_counts = combined.groupby("node_id").size()
    summary = {
        "version": "03_similar_node_amount_and_profit_ranking",
        "date_window": {"start": pd.Timestamp(actual_days[0]).strftime("%Y-%m-%d"), "end": pd.Timestamp(actual_days[-1]).strftime("%Y-%m-%d"), "actual_data_days": len(actual_days)},
        "recent_active_nodes": int(activity["node_id"].nunique()),
        "scored_nodes_complete_static_attributes": int(nodes["node_id"].nunique()),
        "excluded_nodes_missing_or_unknown_static_attributes": int(activity["node_id"].nunique() - nodes["node_id"].nunique()),
        "eligible_businesses": len(businesses),
        "candidate_node_business_scores": int(len(scores)),
        "amount_top3_rows": int(len(amount)),
        "direct_top3_rows": int(len(direct)),
        "combined_rows": int(len(combined)),
        "combined_unique_candidates_per_node": {str(int(k)): int(v) for k, v in union_counts.value_counts().sort_index().items()},
        "combined_rule": "Each source contributes rank points 3/2/1 for ranks 1/2/3; sum the points, prefer appearing in both sources, then direct profit score and amount profit; retain at most 5 unique businesses.",
        "seconds": round(time.time() - started, 3),
        "outputs": [AMOUNT_TOP3.name, DIRECT_TOP3.name, COMBINED_TOP5.name],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
