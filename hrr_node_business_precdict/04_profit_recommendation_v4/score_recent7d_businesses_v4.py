from __future__ import annotations

import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from score_recent7d_all_businesses_v3 import DAILY, OUT, load_predictor, score_business


HERE = Path(__file__).resolve().parent
RANKER_DIR = OUT / "debiased_pairwise_ranker_v4_artifact"
CALIBRATION = RANKER_DIR / "calibration.json"
AMOUNT_TOP3 = OUT / "v4_1_recent7d_feasible_amount_profit_top3.csv"
REVISED_TOP3 = OUT / "v4_1_recent7d_debiased_profit_top3.csv"
COMBINED_TOP5 = OUT / "v4_1_recent7d_combined_top5.csv"
SUMMARY = OUT / "v4_1_recent7d_recommendation_summary.json"
CHECKPOINT_DIR = HERE / ".codex_work" / "v4_1_asymmetric_delivery_score_chunks"
BETA = 0.30
AMOUNT_STABILITY_BETA = 0.10


def load_ranker_bundle() -> dict:
    manifest = json.loads((RANKER_DIR / "manifest.json").read_text(encoding="utf-8"))
    model_name = "pairwise_no_business_prior_d3"
    spec = manifest["models"][model_name]
    model = XGBRanker()
    model.load_model(RANKER_DIR / spec["model_file"])
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return {
        "model": model,
        "spec": spec,
        "calibration": calibration,
    }


def top_n(frame: pd.DataFrame, score: str, rank_name: str) -> pd.DataFrame:
    result = frame.sort_values(["node_id", score, "business"], ascending=[True, False, True]).groupby("node_id", sort=False).head(3).copy()
    result[rank_name] = result.groupby("node_id", sort=False).cumcount() + 1
    return result[[
        "node_id", rank_name, "business", "business_name", "attribute_delivery_type", "outcome_delivery_type",
        "predicted_cost", "predicted_revenue", "predicted_amount_profit", "direct_profit_score",
        "amount_ranking_score", "pairwise_profit_score", "revised_profit_score", "profit_training_support",
    ]]


def combine_rankings(amount: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    left = amount.rename(columns={"amount_profit_rank": "amount_rank"})
    rank = revised[["node_id", "business", "revised_profit_rank"]].rename(columns={"revised_profit_rank": "revised_rank"})
    merged = left.merge(rank, on=["node_id", "business"], how="outer")
    missing = merged["business_name"].isna()
    if missing.any():
        details = revised.set_index(["node_id", "business"])
        keys = pd.MultiIndex.from_frame(merged.loc[missing, ["node_id", "business"]])
        for column in [
            "business_name", "attribute_delivery_type", "outcome_delivery_type", "predicted_cost", "predicted_revenue",
            "predicted_amount_profit", "direct_profit_score", "pairwise_profit_score", "revised_profit_score", "profit_training_support",
            "amount_ranking_score",
        ]:
            merged.loc[missing, column] = details.reindex(keys)[column].to_numpy()
    merged["amount_points"] = np.where(merged["amount_rank"].notna(), 4 - merged["amount_rank"], 0)
    merged["revised_points"] = np.where(merged["revised_rank"].notna(), 4 - merged["revised_rank"], 0)
    merged["combined_points"] = merged["amount_points"] + merged["revised_points"]
    merged["source_count"] = merged[["amount_rank", "revised_rank"]].notna().sum(axis=1)
    merged["selected_by"] = np.select(
        [merged["source_count"].eq(2), merged["amount_rank"].notna()],
        ["both", "amount_profit"], default="debiased_profit",
    )
    merged = merged.sort_values(
        ["node_id", "combined_points", "source_count", "revised_profit_score", "predicted_amount_profit", "business"],
        ascending=[True, False, False, False, False, True],
    ).groupby("node_id", sort=False).head(5).copy()
    merged["combined_rank"] = merged.groupby("node_id", sort=False).cumcount() + 1
    return merged[[
        "node_id", "combined_rank", "business", "business_name", "selected_by", "source_count",
        "amount_rank", "revised_rank", "amount_points", "revised_points", "combined_points",
        "attribute_delivery_type", "outcome_delivery_type", "predicted_cost", "predicted_revenue", "predicted_amount_profit",
        "amount_ranking_score", "direct_profit_score", "pairwise_profit_score", "revised_profit_score", "profit_training_support",
    ]]


def distribution(frame: pd.DataFrame, rank_col: str) -> dict:
    top = frame[frame[rank_col].eq(1)]
    counts = top.groupby(["business", "business_name"], observed=True).size().sort_values(ascending=False)
    top_business, top_name = counts.index[0]
    b_count = int(top[top["business"].eq("10000183")].shape[0])
    return {
        "nodes": int(top["node_id"].nunique()),
        "unique_top1_businesses": int(len(counts)),
        "largest_top1_business": str(top_business),
        "largest_top1_business_name": str(top_name),
        "largest_top1_share": float(counts.iloc[0] / len(top)),
        "business_10000183_top1_count": b_count,
        "business_10000183_top1_share": float(b_count / len(top)),
        "top10": [
            {"business": str(idx[0]), "business_name": str(idx[1]), "count": int(value), "share": float(value / len(top))}
            for idx, value in counts.head(10).items()
        ],
    }


def delivery_compatible(attribute_type: pd.Series, business_type: pd.Series) -> pd.Series:
    """Aggregation nodes only run aggregation; dedicated nodes may run both."""
    attribute_type = attribute_type.astype(str)
    business_type = business_type.astype(str)
    return (
        attribute_type.eq("dedicated") & business_type.isin(["dedicated", "aggregation"])
    ) | (
        attribute_type.eq("aggregation") & business_type.eq("aggregation")
    )


def main() -> None:
    started = time.time()
    predictor = load_predictor()
    ranker = load_ranker_bundle()
    daily = pd.read_csv(DAILY, dtype={"nodeId": str, "customerId": str}, low_memory=False)
    daily["day"] = pd.to_datetime(daily["day"], errors="coerce")
    actual_days = sorted(daily["day"].dropna().dt.normalize().unique())[-7:]
    recent = daily[daily["day"].dt.normalize().isin(actual_days)].copy()
    activity = recent.groupby("nodeId", observed=True).agg(
        recent7d_active_days=("day", lambda s: s.dt.normalize().nunique()),
        recent7d_first_day=("day", "min"), recent7d_last_day=("day", "max"),
    ).reset_index().rename(columns={"nodeId": "node_id"})
    known = set(predictor.nodes["node_id"].astype(str))
    matched = activity[activity["node_id"].isin(known)].copy()
    nodes = predictor.nodes[predictor.nodes["node_id"].isin(matched["node_id"])].copy()
    metadata = predictor.amounts.groupby("business", observed=True).agg(
        outcome_delivery_type=("outcome_delivery_type", "first"),
        profit_training_support=("profit_target", "count"),
    )
    eligible_businesses = metadata[metadata["profit_training_support"].gt(3)].index.astype(str).tolist()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scored_paths = []
    for index, business in enumerate(sorted(eligible_businesses), start=1):
        delivery_type = str(metadata.at[business, "outcome_delivery_type"])
        if delivery_type == "aggregation":
            compatible_nodes = nodes[nodes["attribute_delivery_type"].astype(str).isin(["aggregation", "dedicated"])].copy()
        elif delivery_type == "dedicated":
            compatible_nodes = nodes[nodes["attribute_delivery_type"].astype(str).eq("dedicated")].copy()
        else:
            compatible_nodes = nodes.iloc[0:0].copy()
        if compatible_nodes.empty:
            continue
        checkpoint = CHECKPOINT_DIR / f"{business}.csv"
        if not checkpoint.exists():
            current = score_business(predictor, compatible_nodes, business, ranker_bundle=ranker)
            current.to_csv(checkpoint, index=False, encoding="utf-8-sig")
            del current
            gc.collect()
        scored_paths.append(checkpoint)
        if index % 5 == 0 or index == len(eligible_businesses):
            print(f"scored feasible businesses {index}/{len(eligible_businesses)}", flush=True)
    scores = pd.concat(
        [pd.read_csv(path, dtype={"node_id": str, "business": str}) for path in scored_paths],
        ignore_index=True,
    )
    scores = scores[
        delivery_compatible(scores["attribute_delivery_type"], scores["outcome_delivery_type"])
        & scores["profit_training_support"].gt(3)
    ].copy()
    revised_standard = (scores["revised_profit_score"] - scores["revised_profit_score"].mean()) / max(float(scores["revised_profit_score"].std()), 1e-9)
    scores["amount_ranking_score"] = (
        scores["predicted_amount_profit"]
        + AMOUNT_STABILITY_BETA * float(scores["predicted_amount_profit"].std()) * revised_standard
    )
    amount = top_n(scores, "amount_ranking_score", "amount_profit_rank")
    revised = top_n(scores, "revised_profit_score", "revised_profit_rank")
    combined = combine_rankings(amount, revised)

    activity_columns = ["node_id", "recent7d_active_days", "recent7d_first_day", "recent7d_last_day"]
    for result, path in [(amount, AMOUNT_TOP3), (revised, REVISED_TOP3), (combined, COMBINED_TOP5)]:
        result = result.merge(matched[activity_columns], on="node_id", how="left")
        result["recent7d_first_day"] = result["recent7d_first_day"].dt.strftime("%Y-%m-%d")
        result["recent7d_last_day"] = result["recent7d_last_day"].dt.strftime("%Y-%m-%d")
        result.to_csv(path, index=False, encoding="utf-8-sig")

    covered_nodes = int(scores["node_id"].nunique())
    summary = {
        "version": "04_1_asymmetric_delivery_feasible_debiased_profit_ranking",
        "date_window": {"start": pd.Timestamp(actual_days[0]).strftime("%Y-%m-%d"), "end": pd.Timestamp(actual_days[-1]).strftime("%Y-%m-%d")},
        "rules": {
            "delivery_eligibility": "aggregation node -> aggregation business only; dedicated node -> aggregation or dedicated business",
            "minimum_profit_training_support": 4,
            "profit_score": f"similar-node profit regressor + {BETA} * standardized no-business-prior pairwise score",
            "amount_ranking_score": f"predicted revenue - predicted cost + {AMOUNT_STABILITY_BETA} * standardized revised-profit stability signal",
        },
        "recent_active_nodes": int(activity["node_id"].nunique()),
        "matched_complete_static_nodes": int(nodes["node_id"].nunique()),
        "recommended_nodes_with_feasible_candidates": covered_nodes,
        "nodes_without_feasible_candidates": int(nodes["node_id"].nunique() - covered_nodes),
        "eligible_businesses_support_gt3": len(eligible_businesses),
        "feasible_candidate_scores": int(len(scores)),
        "validation": {
            "old_pairwise_accuracy": 0.6370597243491577,
            "revised_pairwise_accuracy": 0.6477794793261868,
            "old_ndcg_at_3": 0.8660495430267006,
            "revised_ndcg_at_3": 0.8700058856377263,
            "old_mean_top1_regret": 6.390333095747599,
            "revised_mean_top1_regret": 6.350722571450647,
        },
        "v4_amount_top1_distribution": distribution(amount, "amount_profit_rank"),
        "v4_revised_top1_distribution": distribution(revised, "revised_profit_rank"),
        "combined_rows": int(len(combined)),
        "seconds": round(time.time() - started, 3),
        "outputs": [AMOUNT_TOP3.name, REVISED_TOP3.name, COMBINED_TOP5.name],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
