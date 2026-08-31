from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from xgboost import XGBRegressor


NEIGHBOR_FEATURES = [
    "neighbor_prior_k3", "neighbor_prior_k5", "neighbor_prior_k10", "neighbor_prior_k20",
    "neighbor_support", "neighbor_similarity_max", "neighbor_similarity_mean", "neighbor_std_k10",
]
warnings.filterwarnings("ignore", category=FutureWarning)


class SimilarNodePredictor:
    def __init__(self, artifact_dir: str | Path):
        self.root = Path(artifact_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.bundle = joblib.load(self.root / self.manifest["files"]["similarity_preprocessor"])
        self.nodes = pd.read_csv(self.root / self.manifest["files"]["node_reference"], dtype={"node_id": str}, low_memory=False)
        self.amounts = pd.read_csv(self.root / self.manifest["files"]["amount_reference"], dtype={"node_id": str, "business": str}, low_memory=False)
        self.node_rows = self.nodes.set_index("node_id", drop=False)
        self.node_matrix = self._similarity_transform(self.nodes)
        self.node_position = dict(zip(self.nodes["node_id"], range(len(self.nodes))))
        self.models = {}
        for name, spec in self.manifest["models"].items():
            model = XGBRegressor(); model.load_model(self.root / spec["file"]); self.models[name] = model
        metadata = self.amounts.groupby("business", observed=True).agg(
            business_name=("business_name", "first"), outcome_delivery_type=("outcome_delivery_type", "first")
        )
        self.business_metadata = metadata

    def _similarity_transform(self, rows: pd.DataFrame):
        cats, nums = self.bundle["categorical_features"], self.bundle["numeric_features"]
        cat = self.bundle["encoder"].transform(rows[cats].fillna("__MISSING__").astype(str))
        numeric = np.log1p(rows[nums].apply(pd.to_numeric, errors="coerce").clip(lower=0))
        numeric = self.bundle["imputer"].transform(numeric)
        numeric = np.clip(self.bundle["scaler"].transform(numeric), -4, 4).astype(np.float32)
        matrix = sparse.hstack([
            cat * self.bundle["categorical_weight"],
            sparse.csr_matrix(numeric) * self.bundle["numeric_weight"],
        ], format="csr")
        return normalize(matrix, axis=1).tocsr()

    @staticmethod
    def _weighted(values: np.ndarray, sims: np.ndarray, k: int) -> tuple[float, float, int]:
        n = min(k, len(values))
        if n == 0:
            return np.nan, np.nan, 0
        values, sims = values[:n], sims[:n]
        weights = np.maximum(sims, 0) ** 4
        if weights.sum() <= 1e-12: weights = np.ones(n)
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(max(np.average((values - mean) ** 2, weights=weights), 0)))
        return mean, std, n

    def _neighbor_features(self, node_id: str, business: str, target: str) -> dict:
        candidates = self.amounts[(self.amounts["business"] == business) & self.amounts[target].notna() & (self.amounts["node_id"] != node_id)].drop_duplicates("node_id")
        output = {c: np.nan for c in NEIGHBOR_FEATURES}
        if candidates.empty: return output
        positions = np.array([self.node_position[n] for n in candidates["node_id"] if n in self.node_position], dtype=int)
        candidates = candidates[candidates["node_id"].isin(self.node_position)].copy()
        if len(positions) == 0: return output
        query = self.node_matrix[self.node_position[node_id]]
        sims = (query @ self.node_matrix[positions].T).toarray().ravel()
        order = np.argsort(-sims)[:20]; sims = sims[order]
        values = candidates[target].to_numpy(float)[order]
        for k in [3, 5, 10, 20]:
            output[f"neighbor_prior_k{k}"] = self._weighted(values, sims, k)[0]
        _, std, support = self._weighted(values, sims, 10)
        output.update({
            "neighbor_support": support, "neighbor_similarity_max": float(sims[0]),
            "neighbor_similarity_mean": float(np.mean(sims[:support])), "neighbor_std_k10": std,
        })
        return output

    def _amount_priors(self, node_id: str, business: str, target: str, weighted: bool) -> dict:
        short = target.split("_")[0]
        data = self.amounts[self.amounts[target].notna()].copy()
        # Makes predictions of already-observed pairs behave like counterfactual masking.
        data = data[~((data["node_id"] == node_id) & (data["business"] == business))]
        reliability = pd.to_numeric(data[f"{short}_reliability_weight"], errors="coerce").fillna(0).clip(lower=0)
        y = data[target].astype(float)
        global_mean = float(np.average(y, weights=np.maximum(reliability, 1e-9)))

        def group_prior(column: str, value: str, alpha: float):
            selected = data[data[column].astype(str) == str(value)]
            if selected.empty: return global_mean, 0.0, 0.0
            amounts = selected[target].to_numpy(float)
            if weighted:
                weights = selected[f"{short}_reliability_weight"].to_numpy(float)
                prior = (np.sum(weights * amounts) + alpha * global_mean) / (np.sum(weights) + alpha)
                support = np.sum(weights)
                raw_mean = np.average(amounts, weights=np.maximum(weights, 1e-9))
                std = np.sqrt(np.average((amounts - raw_mean) ** 2, weights=np.maximum(weights, 1e-9)))
            else:
                prior = (np.sum(amounts) + alpha * global_mean) / (len(amounts) + alpha)
                support = len(amounts); std = np.std(amounts, ddof=1) if len(amounts) > 1 else 0.0
            return float(prior), float(np.log1p(support)), float(std)

        bp, bs, bstd = group_prior("business", business, 12.0)
        np_, ns, nstd = group_prior("node_id", node_id, 2.0)
        return {
            "business_prior": bp, "business_support_log1p": bs, "business_std": bstd,
            "node_prior": np_, "node_support_log1p": ns, "node_std": nstd,
            "node_business_interaction_prior": bp * np_ / max(global_mean, 1e-6),
        }

    def _feature_row(self, node_id: str, business: str, target: str, model_name: str, augmented: bool) -> pd.DataFrame:
        if node_id not in self.node_rows.index: raise KeyError(f"Unknown existing node: {node_id}")
        row = self.node_rows.loc[[node_id]].reset_index(drop=True).copy()
        row["business"] = str(business)
        if business in self.business_metadata.index:
            row["outcome_delivery_type"] = self.business_metadata.at[business, "outcome_delivery_type"]
        else:
            row["outcome_delivery_type"] = "__MISSING__"
        short = target.split("_")[0]
        for key, value in self._amount_priors(node_id, business, target, weighted=(short == "revenue")).items(): row[key] = value
        if augmented:
            for key, value in self._neighbor_features(node_id, business, target).items(): row[key] = value
        spec = self.manifest["models"][model_name]
        for column in spec["feature_columns"]:
            if column not in row: row[column] = np.nan
        for column, levels in spec["categories"].items():
            value = row[column].fillna("__MISSING__").astype(str)
            row[column] = pd.Categorical(value, categories=levels)
        numeric = [c for c in spec["feature_columns"] if c not in spec["categories"]]
        for column in numeric: row[column] = pd.to_numeric(row[column], errors="coerce").astype(np.float32)
        return row[spec["feature_columns"]]

    def predict_one(self, node_id: str, business: str) -> dict:
        node_id, business = str(node_id), str(business)
        cost_base = float(self.models["cost_base"].predict(self._feature_row(node_id, business, "cost_target", "cost_base", False))[0])
        cost_sim = float(self.models["cost_similar"].predict(self._feature_row(node_id, business, "cost_target", "cost_similar", True))[0])
        rev_base = float(self.models["revenue_base"].predict(self._feature_row(node_id, business, "revenue_target", "revenue_base", False))[0])
        rev_sim = float(self.models["revenue_similar"].predict(self._feature_row(node_id, business, "revenue_target", "revenue_similar", True))[0])
        weight = self.manifest["prediction"]["cost_blend"]["similar_weight"]
        cost = max(weight * cost_sim + (1 - weight) * cost_base, 0.0)
        weight = self.manifest["prediction"]["revenue_blend"]["similar_weight"]
        revenue = max(weight * rev_sim + (1 - weight) * rev_base, 0.0)
        rank_score = float(self.models["profit_similar"].predict(self._feature_row(node_id, business, "profit_target", "profit_similar", True))[0])
        support = int(((self.amounts["business"] == business) & self.amounts["profit_target"].notna()).sum())
        return {"node_id": node_id, "business": business, "predicted_cost": cost, "predicted_revenue": revenue, "predicted_amount_profit": revenue - cost, "ranking_profit_score": rank_score, "profit_training_support": support, "recommendation_eligible": support > 0}

    def rank_businesses(self, node_id: str, businesses: list[str] | None = None) -> pd.DataFrame:
        businesses = businesses or sorted(self.amounts.loc[self.amounts["profit_target"].notna(), "business"].dropna().astype(str).unique())
        result = pd.DataFrame([self.predict_one(node_id, business) for business in businesses])
        return result.sort_values("ranking_profit_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(Path(__file__).resolve().parent / "matched_similar_node_v3_model_artifact"))
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--business")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    predictor = SimilarNodePredictor(args.artifact)
    if args.business:
        result = predictor.predict_one(args.node_id, args.business)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = predictor.rank_businesses(args.node_id).head(args.top_n)
        print(result.to_string(index=False))
        if args.output: result.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
