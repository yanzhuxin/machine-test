#!/usr/bin/env python3
"""为最终90/10遮挡验证逐条输出金额差额和百分比误差。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "counterfactual_masked_validation_predictions.csv"
OUTPUT = HERE / "counterfactual_validation_error_details.csv"
SUMMARY = HERE / "counterfactual_validation_error_summary.json"


def add_error_columns(frame: pd.DataFrame, target: str) -> dict:
    actual = pd.to_numeric(frame[f"actual_{target}_7d"], errors="coerce")
    predicted = pd.to_numeric(frame[f"predicted_{target}_7d"], errors="coerce")
    signed = predicted - actual
    absolute = signed.abs()
    nonzero = actual.abs() > 1e-12
    frame[f"{target}_signed_error"] = signed
    frame[f"{target}_absolute_error"] = absolute
    frame[f"{target}_actual_is_zero"] = ~nonzero
    frame[f"{target}_signed_percentage_error_pct"] = np.where(nonzero, signed / actual * 100.0, np.nan)
    frame[f"{target}_absolute_percentage_error_pct"] = np.where(nonzero, absolute / actual.abs() * 100.0, np.nan)
    frame[f"{target}_prediction_to_actual_ratio"] = np.where(nonzero, predicted / actual, np.nan)
    denominator = actual.abs() + predicted.abs()
    frame[f"{target}_symmetric_percentage_error_pct"] = np.where(
        denominator > 1e-12, 200.0 * absolute / denominator, 0.0
    )
    ape = frame.loc[nonzero, f"{target}_absolute_percentage_error_pct"]
    ge10 = actual >= 10.0
    ge10_ape = frame.loc[ge10, f"{target}_absolute_percentage_error_pct"]
    ge100 = actual >= 100.0
    ge100_ape = frame.loc[ge100, f"{target}_absolute_percentage_error_pct"]
    return {
        "rows": int(len(frame)), "actual_zero_rows": int((~nonzero).sum()),
        "percentage_error_defined_rows": int(nonzero.sum()),
        "median_absolute_percentage_error_pct": float(ape.median()),
        "p75_absolute_percentage_error_pct": float(ape.quantile(.75)),
        "p90_absolute_percentage_error_pct": float(ape.quantile(.90)),
        "within_20pct_share": float((ape <= 20).mean()),
        "within_50pct_share": float((ape <= 50).mean()),
        "within_100pct_share": float((ape <= 100).mean()),
        "mean_symmetric_percentage_error_pct": float(frame[f"{target}_symmetric_percentage_error_pct"].mean()),
        "actual_ge_10_rows": int(ge10.sum()),
        "actual_ge_10_median_absolute_percentage_error_pct": float(ge10_ape.median()),
        "actual_ge_10_within_50pct_share": float((ge10_ape <= 50).mean()),
        "actual_ge_100_rows": int(ge100.sum()),
        "actual_ge_100_median_absolute_percentage_error_pct": float(ge100_ape.median()),
        "actual_ge_100_within_50pct_share": float((ge100_ape <= 50).mean()),
    }


def main() -> None:
    frame = pd.read_csv(SOURCE, dtype={"node_id": str, "business": str})
    summary = {target: add_error_columns(frame, target) for target in ["cost", "revenue"]}
    ordered = [
        "node_id", "business", "business_online_day",
        "actual_cost_7d", "predicted_cost_7d", "cost_signed_error", "cost_absolute_error",
        "cost_signed_percentage_error_pct", "cost_absolute_percentage_error_pct",
        "cost_prediction_to_actual_ratio", "cost_symmetric_percentage_error_pct", "cost_actual_is_zero",
        "actual_revenue_7d", "predicted_revenue_7d", "revenue_signed_error", "revenue_absolute_error",
        "revenue_signed_percentage_error_pct", "revenue_absolute_percentage_error_pct",
        "revenue_prediction_to_actual_ratio", "revenue_symmetric_percentage_error_pct", "revenue_actual_is_zero",
        "actual_profit_7d", "predicted_profit_7d",
    ]
    frame[ordered].to_csv(OUTPUT, index=False)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
