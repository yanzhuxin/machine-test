from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from run_last30d_amount_models import load_frame, smoothed_priors
from run_explicit_similar_node_optimization import NEIGHBOR_FEATURES, attach_neighbor_priors, make_similarity_matrix, prepare_with_neighbors
from run_last30d_profit_ranking import ranking_metrics
from tune_matched_profit_ranking_v2 import ranking_mask

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "01a0225a-43b2-7991-a2c3-00e9e4064c5d"
SPLIT = OUT / "last30d_amount_split.csv"
VALIDATION_OUT = OUT / "v4_3_validation_predictions.csv"
FUTURE_IN = OUT / "v42_future_20260828_20260903_predictions.csv"
FUTURE_OUT = OUT / "v4_3_future_20260828_20260903_predictions.csv"
METRICS_OUT = OUT / "v4_3_soft_pseudo_metrics.json"
REPORT_OUT = OUT / "v4_3_soft_pseudo_report.md"
ARTIFACT_DIR = OUT / "v4_3_soft_pseudo_artifact"
WEIGHTS = (0.1, 0.5, 1.0)
SIMILARITY_THRESHOLDS = (0.99, 0.994, 0.995, 0.996, 0.999)
PSEUDO_TOP_K = 3
SEED = 20260903


def build_pseudo_pool(train: pd.DataFrame, matrix, node_to_row: dict[str, int], min_similarity: float) -> pd.DataFrame:
    train = train.copy()
    businesses = sorted(train["business"].astype(str).unique())
    nodes = train.drop_duplicates("node_id").set_index("node_id", drop=False)
    observed = set(zip(train["node_id"].astype(str), train["business"].astype(str)))
    rows = []
    node_ids = nodes.index.astype(str).tolist()
    node_positions = np.array([node_to_row[n] for n in node_ids if n in node_to_row], dtype=int)
    node_lookup = nodes.loc[[n for n in node_ids if n in node_to_row]]
    for business in businesses:
        refs = train[(train["business"].astype(str) == business) & train["profit_target"].notna()].drop_duplicates("node_id")
        refs = refs[refs["node_id"].astype(str).isin(node_to_row)]
        if len(refs) < 3:
            continue
        ref_positions = np.array([node_to_row[str(x)] for x in refs["node_id"]], dtype=int)
        similarity = (matrix[node_positions] @ matrix[ref_positions].T).toarray()
        order = np.argpartition(-similarity, kth=min(PSEUDO_TOP_K - 1, similarity.shape[1] - 1), axis=1)[:, :min(PSEUDO_TOP_K, similarity.shape[1])]
        top_sims = np.take_along_axis(similarity, order, axis=1)
        top_vals = refs["profit_target"].to_numpy(float)[order]
        for row_pos, node_id in enumerate(node_lookup.index.astype(str)):
            if (node_id, business) in observed:
                continue
            local_order = np.argsort(-top_sims[row_pos])
            sims = top_sims[row_pos][local_order]
            vals = top_vals[row_pos][local_order]
            if sims[0] < min_similarity:
                continue
            weights = np.maximum(sims, 0.0) ** 4
            pseudo = float(np.average(vals, weights=weights)) if weights.sum() > 0 else float(vals.mean())
            row = node_lookup.iloc[row_pos].copy()
            row["business"] = business
            row["profit_target"] = pseudo
            row["cost_target"] = np.nan
            row["revenue_target"] = np.nan
            row["cost_reliability_weight"] = 0.0
            row["revenue_reliability_weight"] = 0.0
            row["is_pseudo"] = 1
            row["pseudo_neighbor_support"] = len(vals)
            row["pseudo_neighbor_similarity_max"] = float(sims[0])
            row["pseudo_neighbor_similarity_mean"] = float(sims.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def add_pseudo_rows(train: pd.DataFrame, pseudo_pool: pd.DataFrame, weight: float, threshold: float) -> tuple[pd.DataFrame, dict]:
    pseudo = pseudo_pool[pseudo_pool["pseudo_neighbor_similarity_max"].ge(threshold)].copy()
    if pseudo.empty:
        return train.assign(is_pseudo=0), {"rows": 0, "nodes": 0, "weight": weight, "threshold": threshold}
    pseudo["is_validation"] = 0
    pseudo["profit_reliability_weight"] = float(weight)
    output = pd.concat([train.assign(is_pseudo=0), pseudo], ignore_index=True, sort=False)
    return output, {"rows": int(len(pseudo)), "nodes": int(pseudo["node_id"].nunique()), "weight": weight, "threshold": threshold, "weight_mode": "xgboost_sample_weight"}


def prepare(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[list[str], dict]:
    train_p, valid_p = train.copy(), valid.copy()
    columns, categories = prepare_with_neighbors(train_p, valid_p)
    for c in columns:
        if c not in train_p:
            train_p[c] = np.nan
        if c not in valid_p:
            valid_p[c] = np.nan
    return columns, categories


def train_one(train: pd.DataFrame, valid: pd.DataFrame, pseudo_pool: pd.DataFrame, weight: float, threshold: float, matrix, node_to_row: dict[str, int]) -> tuple[pd.DataFrame, dict, dict]:
    augmented, pseudo_meta = add_pseudo_rows(train, pseudo_pool, weight, threshold)
    train_aug = augmented[augmented["profit_target"].notna()].copy()
    real = train_aug[train_aug["is_pseudo"].eq(0)].copy()
    pseudo = train_aug[train_aug["is_pseudo"].eq(1)].copy()
    real_n, valid_n, _ = attach_neighbor_priors(real, valid.copy(), "profit_target", matrix, node_to_row, "v43")
    for column in NEIGHBOR_FEATURES:
        pseudo[column] = np.nan
    train_aug = pd.concat([real_n, pseudo], ignore_index=True, sort=False)
    train_aug, valid_work, _ = smoothed_priors(train_aug, valid_n, "profit_target")
    train_p, valid_p = train_aug.copy(), valid_work.copy()
    columns, categories = prepare_with_neighbors(train_p, valid_p)
    counts = train_p.groupby("node_id", observed=True).size()
    rank_nodes = counts[counts >= 2].index
    rank_train = train_p[train_p["node_id"].isin(rank_nodes)].sort_values(["node_id", "business"]).copy()
    y = rank_train.groupby("node_id", observed=True)["profit_target"].rank(method="average", pct=True).to_numpy(float)
    qid = pd.factorize(rank_train["node_id"], sort=True)[0]
    # XGBRanker interprets sample_weight as one weight per query group, not
    # one weight per row. Aggregate the real/pseudo row weights per node.
    group_weights = rank_train.assign(
        _row_weight=rank_train["profit_reliability_weight"].fillna(1.0)
    ).groupby("node_id", sort=False, observed=True)["_row_weight"].mean()
    sample_weight = group_weights.reindex(pd.Index(rank_train["node_id"].drop_duplicates())).to_numpy(float)
    model = XGBRanker(objective="rank:pairwise", eval_metric="ndcg@3", n_estimators=450, learning_rate=0.035, max_depth=3, min_child_weight=8, subsample=0.88, colsample_bytree=0.88, reg_lambda=20.0, reg_alpha=0.2, tree_method="hist", enable_categorical=True, max_cat_to_onehot=8, random_state=SEED, n_jobs=-1)
    started = time.time()
    model.fit(rank_train[columns], y, qid=qid, sample_weight=sample_weight, verbose=False)
    valid_work["v4_3_score"] = model.predict(valid_p[columns])
    result, _ = ranking_metrics(valid_work, "v4_3_score")
    result.update({"similarity_threshold": threshold, "pseudo_weight": weight, "pseudo_rows": pseudo_meta["rows"], "pseudo_nodes": pseudo_meta["nodes"], "train_rows": int(len(rank_train)), "feature_count": len(columns), "fit_seconds": round(time.time() - started, 3)})
    return valid_work, result, {
        "model": model,
        "columns": columns,
        "categories": categories,
        "pseudo_meta": pseudo_meta,
        "real_train": real,
        "prior_train": train_aug,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one v4.3 soft-pseudo candidate.")
    parser.add_argument("--weight", type=float, required=True, choices=WEIGHTS)
    parser.add_argument("--threshold", type=float, required=True, choices=SIMILARITY_THRESHOLDS)
    args = parser.parse_args()

    print("loading training frame", flush=True)
    frame = load_frame()
    frame = frame[frame["attribute_missing_flag"].eq(0)].copy()
    frame["profit_reliability_weight"] = np.minimum(frame["cost_reliability_weight"], frame["revenue_reliability_weight"])
    split = pd.read_csv(SPLIT, dtype={"node_id": str, "business": str})
    frame = frame.merge(split, on=["node_id", "business"], how="left", validate="one_to_one")
    mask = ranking_mask(frame)
    usable = frame["profit_target"].notna()
    train = frame[usable & ~mask].copy()
    valid = frame[usable & mask].copy()
    print("building node similarity matrix", flush=True)
    matrix, node_to_row, similarity = make_similarity_matrix(frame)
    print("building pseudo-sample pool", flush=True)
    pseudo_pool = build_pseudo_pool(train, matrix, node_to_row, min(SIMILARITY_THRESHOLDS))
    print(f"pseudo-sample pool rows: {len(pseudo_pool)}", flush=True)
    similarity_distribution = pseudo_pool["pseudo_neighbor_similarity_max"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.97, 0.99]).to_dict()
    work, metrics, bundle = train_one(train, valid, pseudo_pool, args.weight, args.threshold, matrix, node_to_row)
    results = [metrics]
    predictions = [work[["node_id", "business", "profit_target", "v4_3_score"]].copy()]
    bundles = {(args.threshold, args.weight): bundle}
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    result_frame = pd.DataFrame(results)
    selected = result_frame.sort_values(["pairwise_accuracy", "mean_ndcg_at_3", "mean_top1_regret"], ascending=[False, False, True]).iloc[0]
    selected_weight = float(selected["pseudo_weight"])
    selected_threshold = float(selected["similarity_threshold"])
    selected_valid = next(p for p, r in zip(predictions, results) if float(r["pseudo_weight"]) == selected_weight and float(r["similarity_threshold"]) == selected_threshold)
    selected_valid.to_csv(VALIDATION_OUT, index=False, encoding="utf-8-sig")
    future_result = None
    if FUTURE_IN.exists():
        future = pd.read_csv(FUTURE_IN, dtype={"node_id": str, "business": str}, low_memory=False)
        future = future.rename(columns={"actual_profit": "profit_target"})
        future["is_pseudo"] = 0
        # The selected soft-supervision model is trained on the original train fold;
        # future rows are scored only after the model is fixed.
        bundle = bundles[(selected_threshold, selected_weight)]
        model, columns, categories = bundle["model"], bundle["columns"], bundle["categories"]
        _, future_n, _ = attach_neighbor_priors(
            bundle["real_train"], future.copy(), "profit_target", matrix, node_to_row, "v43_future"
        )
        _, work, _ = smoothed_priors(bundle["prior_train"], future_n, "profit_target")
        missing_columns = [column for column in columns if column not in work]
        if missing_columns:
            work = pd.concat(
                [work, pd.DataFrame(np.nan, index=work.index, columns=missing_columns)], axis=1
            )
        for column, levels in categories.items():
            work[column] = pd.Categorical(work[column].fillna("__MISSING__").astype(str), categories=levels)
        for column in (c for c in columns if c not in categories):
            work[column] = pd.to_numeric(work[column], errors="coerce").astype(np.float32)
        future["v4_3_score"] = model.predict(work[columns])
        future.to_csv(FUTURE_OUT, index=False, encoding="utf-8-sig")
        multi_counts = future.groupby("node_id")["business"].nunique()
        all_multibusiness = future[future["node_id"].isin(multi_counts[multi_counts >= 2].index)].copy()
        unseen_counts = future.groupby("node_id")["is_strict_unseen_business"].sum()
        strict_unseen = future[
            future["node_id"].isin(unseen_counts[unseen_counts >= 2].index)
            & future["is_strict_unseen_business"]
        ].copy()
        mixed_ids = future.groupby("node_id").agg(
            unseen=("is_strict_unseen_business", "sum"), total=("business", "nunique")
        ).query("unseen >= 1 and total >= 2").index
        mixed_unseen = future[future["node_id"].isin(mixed_ids)].copy()
        future_result = {
            "rows": int(len(future)),
            "nodes": int(future["node_id"].nunique()),
            "all_multibusiness": ranking_metrics(all_multibusiness, "v4_3_score")[0],
            "mixed_with_unseen": ranking_metrics(mixed_unseen, "v4_3_score")[0],
            "strict_unseen_only": ranking_metrics(strict_unseen, "v4_3_score")[0],
        }
    payload = {"version": "v4.3_soft_pseudo_samples", "validation_design": "original v4.1 deterministic mask; pseudo rows generated only from training nodes/business outcomes", "similarity": similarity, "pseudo_similarity_distribution": similarity_distribution, "baseline_v41": {"pairwise_accuracy": 0.6477794793261868, "ndcg_at_3": 0.8700058856377263, "mean_top1_regret": 6.350722571450647}, "selected_threshold": selected_threshold, "selected_weight": selected_weight, "candidates": results, "future_test": future_result}
    ARTIFACT_DIR.mkdir(exist_ok=True)
    candidate_dir = ARTIFACT_DIR / f"threshold_{selected_threshold:g}_weight_{selected_weight:g}"
    candidate_dir.mkdir(exist_ok=True)
    (candidate_dir / "manifest.json").write_text(json.dumps({"version": "v4.3", "selected_threshold": selected_threshold, "selected_weight": selected_weight, "pseudo_sample_policy": "low-weight similarity soft labels; validation targets excluded", "selected_metrics": dict(selected)}, ensure_ascii=False, indent=2), encoding="utf-8")
    bundles[(selected_threshold, selected_weight)]["model"].save_model(candidate_dir / "v4_3_ranker.json")
    result_frame.to_csv(candidate_dir / "weight_search.csv", index=False, encoding="utf-8-sig")
    if future_result is not None:
        future.to_csv(candidate_dir / "future_test_predictions.csv", index=False, encoding="utf-8-sig")
        (candidate_dir / "future_test_metrics.json").write_text(
            json.dumps(future_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_OUT.write_text("# v4.3 相似节点低权重软伪样本\n\n" + result_frame.to_csv(index=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
