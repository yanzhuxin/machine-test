#!/usr/bin/env python3
"""基于节点固有配置和宽表业务出现频率的分类推荐模型。

当前分支的目标不是预测成本、收入或利润金额，而是回答：
“具有这类上线前配置的节点，历史上更常出现哪些业务？”

训练标签只来自 test.node_day_ops_wide_full 的业务出现记录：
- 多标签分类：节点是否出现过某个业务；
- 评估标签：节点历史出现天数最多的业务，平局节点不强行指定唯一标签；
- 配置频率：训练节点中，同一配置分组出现该业务的节点占比。

multibusiness_outcomes.csv 是从宽表导出的节点-业务出现结果。本脚本只读取
业务、节点和出现天数列，明确不读取成本、收入或利润列。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
NODES_FILE = HERE / "multibusiness_nodes.csv"
OUTCOMES_FILE = HERE / "multibusiness_outcomes.csv"
OUTPUT_RECOMMENDATIONS = HERE / "classification_recommendations.csv"
OUTPUT_BUSINESS_FREQUENCY = HERE / "classification_business_frequency.csv"
OUTPUT_GROUP_FREQUENCY = HERE / "classification_configuration_business_frequency.csv"
OUTPUT_METRICS = HERE / "classification_metrics.json"
OUTPUT_REPORT = HERE / "配置分类业务推荐报告.md"

RANDOM_SEED = int(os.environ.get("CLASSIFICATION_PRIMARY_SEED", "42"))
VALIDATION_SEEDS = tuple(
    int(value.strip())
    for value in os.environ.get(
        "CLASSIFICATION_VALIDATION_SEEDS", "42,52,62,72,82"
    ).split(",")
    if value.strip()
)
TEST_RATIO = float(os.environ.get("CLASSIFICATION_TEST_RATIO", "0.2"))
MIN_BUSINESS_NODE_SUPPORT = int(
    os.environ.get("CLASSIFICATION_MIN_BUSINESS_SUPPORT", "30")
)
MIN_CONFIG_GROUP_NODES = int(
    os.environ.get("CLASSIFICATION_MIN_CONFIG_GROUP_NODES", "20")
)
BOOTSTRAP_ROUNDS = int(os.environ.get("CLASSIFICATION_BOOTSTRAP_ROUNDS", "1000"))


# 只允许上线前可以确定的固有属性进入模型。运行时健康指标不在这里。
INTRINSIC_CATEGORICAL_FEATURES = [
    "vendorid",
    "deliverytype",
    "resourcetype",
    "dialtype",
    "nattype",
    "scheduleisps",
    "regsource",
    "customermode",
    "province",
    "isp",
    "city",
    "device_type",
    "arch_type",
    "isvm",
    "qoskiller_status",
    "os",
    "arch",
    "hardwaretype",
    "node_manufacturer",
    "node_model",
    "idc_id",
    "idc_name",
    "analysis_nodecustmertype",
    "analysis_nodedeliverytype",
    "analysis_tcpnattype",
    "analysis_udpnattype",
    "analysis_cgroupversion",
    "analysis_lastdeploystate",
    "analysis_lastdeployscenario",
    "analysis_cooperationtype",
    "analysis_supply_side_delivery_type",
    "analysis_isroot",
    "analysis_issupportipv6",
    "analysis_upnpstate",
    "dial_ipv6_enable",
    "dial_on_physical_nic",
    "dial_lbtype",
    "dial_lbtype_v6",
    "join_ismanaged",
    "join_isminorisp",
    "join_isbantransprov",
    "join_isipv6schedule",
    "join_natforwardenable",
    "join_idcbindtype",
    "join_deploystate",
]
INTRINSIC_NUMERIC_FEATURES = [
    "bw",
    "corenum",
    "memtotal",
    "totaldisksize",
    "hdddisksize",
    "ssddisksize",
    "systemdisksize",
    "join_cpu_corenumber",
    "join_cpu_totalcores",
    "join_cpu_totalphysicals",
    "join_cpu_totalthreads",
    "join_memtotal",
    "join_disks_totalsize",
    "join_disks_total_size",
    "join_disks_hdd_size",
    "join_disks_ssd_size",
    "join_disks_system_size",
]

# 配置频率分组采用静态配置和容量档位，不使用节点上线后的观测量。
GROUP_NUMERIC_FEATURES = [
    "bw",
    "corenum",
    "memtotal",
    "totaldisksize",
]
GROUP_LEVELS = {
    "hardware_config": [
        "vendorid",
        "device_type",
        "node_manufacturer",
        "node_model",
        "deliverytype",
        "resourcetype",
        "isvm",
        *GROUP_NUMERIC_FEATURES,
    ],
    "network_config": [
        "deliverytype",
        "resourcetype",
        "dialtype",
        "nattype",
        "province",
        "isp",
        "isvm",
        *GROUP_NUMERIC_FEATURES,
    ],
    "resource_config": [
        "deliverytype",
        "resourcetype",
        "province",
        "isp",
        "isvm",
        *GROUP_NUMERIC_FEATURES,
    ],
}

# 这些字段只用于上线准入和结果说明，不进入分类模型。
HEALTH_RULES = {
    "cpu_load1_per_core": (1.5, ">", "CPU负载/核超过阈值"),
    "mem_used_ratio": (0.95, ">", "内存使用率超过阈值"),
    "disk_used_ratio": (0.95, ">", "磁盘使用率超过阈值"),
    "retrans": (5.0, ">", "node_analysis重传超过阈值"),
    "v4pingloss": (5.0, ">", "IPv4丢包超过阈值"),
    "v6pingloss": (5.0, ">", "IPv6丢包超过阈值"),
    "quality_retransrate": (5.0, ">", "node_join重传超过阈值"),
    "quality_pinglossrate": (5.0, ">", "node_join丢包超过阈值"),
    "maxioutil": (95.0, ">", "磁盘IO利用率超过阈值"),
    "prom_retrans_ratio": (5.0, ">", "Prometheus重传超过阈值"),
    "analysis_retrans_abnormal_ratio": (0.5, ">", "历史重传异常比例超过阈值"),
}
HEALTH_MIN_RULES = {
    "smart_available_spare": (10.0, "SMART可用spare低于阈值"),
}
HEALTH_ZERO_RULES = {
    "smart_bad_block_count": "SMART坏块超过阈值",
    "smart_critical_warning": "SMART critical warning超过阈值",
}
HEALTH_REQUIRED_RULES = {
    "dial_is_normal": (1.0, "拨号状态异常"),
    "zfs_pool_online": (1.0, "ZFS池非online"),
}
HEALTH_MANUAL_ONLY = {
    "smart_case_temperature",
    "zfs_dataset_read",
    "zfs_dataset_write",
}
HEALTH_MISSING_FIELDS = set(HEALTH_RULES) | set(HEALTH_MIN_RULES) | set(
    HEALTH_ZERO_RULES
) | set(HEALTH_REQUIRED_RULES) | HEALTH_MANUAL_ONLY

SAFE_OUTCOME_COLUMNS = [
    "node_id",
    "business",
    "business_active_days",
    "business_name",
]


def normalize_category(value: Any) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"} or text == r"\N":
        return "__MISSING__"
    return text


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not NODES_FILE.exists() or not OUTCOMES_FILE.exists():
        raise FileNotFoundError(
            f"缺少本地宽表导出数据: {NODES_FILE} 或 {OUTCOMES_FILE}"
        )
    nodes = pd.read_csv(NODES_FILE, dtype={"node_id": "string"}, low_memory=False)
    outcomes = pd.read_csv(
        OUTCOMES_FILE,
        usecols=SAFE_OUTCOME_COLUMNS,
        dtype={"node_id": "string", "business": "string"},
        low_memory=False,
    )
    required_nodes = {"node_id", "attribute_day"}
    missing_nodes = required_nodes - set(nodes.columns)
    if missing_nodes:
        raise RuntimeError(f"节点属性文件缺少字段: {sorted(missing_nodes)}")
    required_outcomes = {"node_id", "business", "business_active_days"}
    missing_outcomes = required_outcomes - set(outcomes.columns)
    if missing_outcomes:
        raise RuntimeError(f"宽表业务结果缺少字段: {sorted(missing_outcomes)}")

    nodes["node_id"] = nodes["node_id"].astype(str)
    outcomes["node_id"] = outcomes["node_id"].astype(str)
    outcomes["business"] = outcomes["business"].astype(str)
    outcomes["business_active_days"] = pd.to_numeric(
        outcomes["business_active_days"], errors="coerce"
    ).fillna(0.0)
    outcomes = outcomes.drop_duplicates(["node_id", "business"]).reset_index(drop=True)
    if nodes["node_id"].duplicated().any():
        raise RuntimeError("节点属性文件存在重复 node_id，拒绝静默选择属性行。")
    common_nodes = set(nodes["node_id"]) & set(outcomes["node_id"])
    if not common_nodes:
        raise RuntimeError("节点属性与宽表业务结果没有可交集节点。")
    nodes = nodes[nodes["node_id"].isin(common_nodes)].copy()
    outcomes = outcomes[outcomes["node_id"].isin(common_nodes)].copy()
    return nodes.reset_index(drop=True), outcomes.reset_index(drop=True)


def build_targets(
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], set[str], dict[str, set[str]]]:
    occurrences = outcomes[["node_id", "business", "business_active_days"]].copy()
    max_days = occurrences.groupby("node_id")["business_active_days"].transform("max")
    top = occurrences[occurrences["business_active_days"].eq(max_days)].copy()
    tie_count = top.groupby("node_id")["business"].transform("size")
    unique_top = top[tie_count.eq(1)].drop_duplicates("node_id")
    dominant = dict(zip(unique_top["node_id"], unique_top["business"]))
    ambiguous = set(top.loc[tie_count.gt(1), "node_id"].astype(str))
    observed = {
        str(node_id): set(group["business"].astype(str))
        for node_id, group in occurrences.groupby("node_id")
    }
    return occurrences, dominant, ambiguous, observed


def split_node_ids(node_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    values = np.asarray(sorted(set(node_ids)), dtype=object)
    if len(values) < 2:
        raise RuntimeError("可用于节点级切分的节点数不足。")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    test_count = max(1, int(round(len(values) * TEST_RATIO)))
    test_ids = sorted(values[:test_count].astype(str).tolist())
    train_ids = sorted(values[test_count:].astype(str).tolist())
    return train_ids, test_ids


def effective_features(nodes: pd.DataFrame) -> tuple[list[str], list[str], dict[str, float]]:
    categorical = [
        column
        for column in INTRINSIC_CATEGORICAL_FEATURES
        if column in nodes
        and nodes[column].map(normalize_category).ne("__MISSING__").any()
    ]
    numeric = []
    coverage: dict[str, float] = {}
    for column in INTRINSIC_NUMERIC_FEATURES:
        if column not in nodes:
            coverage[column] = 0.0
            continue
        values = pd.to_numeric(nodes[column], errors="coerce")
        coverage[column] = float(values.notna().mean())
        if values.notna().any():
            numeric.append(column)
    for column in INTRINSIC_CATEGORICAL_FEATURES:
        if column in nodes:
            coverage[column] = float(
                nodes[column].map(normalize_category).ne("__MISSING__").mean()
            )
        else:
            coverage[column] = 0.0
    if not categorical and not numeric:
        raise RuntimeError("没有任何可用节点固有属性，拒绝训练分类模型。")
    return categorical, numeric, coverage


def make_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical:
        category_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                    ),
                ),
                ("onehot", make_one_hot()),
            ]
        )
        transformers.append(("categorical", category_pipeline, categorical))
    if numeric:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def prepare_model_frame(
    frame: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    output = frame[categorical + numeric].copy()
    for column in categorical:
        output[column] = output[column].map(normalize_category)
    return output


def labels_for_nodes(
    node_ids: list[str],
    occurrences: pd.DataFrame,
    businesses: list[str],
) -> np.ndarray:
    candidate_set = set(businesses)
    grouped = (
        occurrences[occurrences["business"].isin(candidate_set)]
        .groupby("node_id")["business"]
        .agg(lambda values: sorted(set(values.astype(str))))
    )
    labels = [grouped.get(node_id, []) for node_id in node_ids]
    encoder = MultiLabelBinarizer(classes=businesses)
    return encoder.fit_transform(labels)


class BusinessOccurrenceClassifier:
    """用节点固有配置预测节点是否会出现每个候选业务。"""

    def __init__(
        self,
        categorical: list[str],
        numeric: list[str],
        seed: int,
    ) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.seed = seed
        self.preprocessor = make_preprocessor(categorical, numeric)
        base = LogisticRegression(
            C=float(os.environ.get("CLASSIFICATION_LOGISTIC_C", "1.0")),
            max_iter=int(
                os.environ.get("CLASSIFICATION_LOGISTIC_MAX_ITER", "600")
            ),
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        self.model = OneVsRestClassifier(
            base,
            n_jobs=int(os.environ.get("CLASSIFICATION_MODEL_JOBS", "-1")),
        )
        self.businesses: list[str] = []

    def fit(
        self,
        nodes: pd.DataFrame,
        train_ids: list[str],
        occurrences: pd.DataFrame,
        businesses: list[str],
    ) -> "BusinessOccurrenceClassifier":
        self.businesses = list(businesses)
        train_nodes = nodes[nodes["node_id"].isin(train_ids)].copy()
        train_nodes = train_nodes.set_index("node_id").loc[train_ids].reset_index()
        train_frame = prepare_model_frame(
            train_nodes,
            self.categorical,
            self.numeric,
        )
        x = self.preprocessor.fit_transform(train_frame)
        y = labels_for_nodes(train_ids, occurrences, self.businesses)
        if y.shape[1] < 2:
            raise RuntimeError("候选业务类别少于2个，无法训练分类器。")
        if not np.all(y.sum(axis=0) > 0):
            raise RuntimeError("存在没有正样本的候选业务，拒绝训练。")
        self.model.fit(x, y)
        return self

    def predict_proba(self, nodes: pd.DataFrame) -> np.ndarray:
        frame = prepare_model_frame(nodes, self.categorical, self.numeric)
        x = self.preprocessor.transform(frame)
        probabilities = np.asarray(self.model.predict_proba(x), dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        return np.clip(probabilities, 0.0, 1.0)


class ConfigurationFrequency:
    """统计相似静态配置节点在宽表中出现业务的频率。"""

    def __init__(
        self,
        min_group_nodes: int,
        numeric_features: list[str],
    ) -> None:
        self.min_group_nodes = min_group_nodes
        self.numeric_features = numeric_features
        self.bin_edges: dict[str, np.ndarray] = {}
        self.group_sizes: dict[str, dict[str, int]] = {}
        self.group_counts: dict[str, dict[tuple[str, str], int]] = {}
        self.global_counts: dict[str, int] = {}
        self.train_node_count = 0

    def fit(
        self,
        nodes: pd.DataFrame,
        train_ids: list[str],
        occurrences: pd.DataFrame,
        businesses: list[str],
    ) -> "ConfigurationFrequency":
        train_nodes = nodes[nodes["node_id"].isin(train_ids)].copy()
        train_nodes = train_nodes.set_index("node_id").loc[train_ids].reset_index()
        self.train_node_count = len(train_nodes)
        for column in self.numeric_features:
            values = pd.to_numeric(train_nodes[column], errors="coerce").dropna()
            if values.empty:
                self.bin_edges[column] = np.array([], dtype=float)
                continue
            edges = np.unique(values.quantile([0.25, 0.5, 0.75]).to_numpy())
            self.bin_edges[column] = edges

        keys = self._keys(train_nodes)
        candidate_set = set(businesses)
        train_occurrences = (
            occurrences[
                occurrences["node_id"].isin(train_ids)
                & occurrences["business"].isin(candidate_set)
            ][["node_id", "business"]]
            .drop_duplicates()
            .merge(keys, on="node_id", how="inner")
        )
        for level in GROUP_LEVELS:
            sizes = keys.groupby(level)["node_id"].nunique()
            self.group_sizes[level] = sizes.astype(int).to_dict()
            counts = (
                train_occurrences.groupby([level, "business"])["node_id"]
                .nunique()
                .astype(int)
            )
            self.group_counts[level] = counts.to_dict()

        self.global_counts = (
            train_occurrences.groupby("business")["node_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        return self

    def _bin_value(self, column: str, value: Any) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "__MISSING__"
        edges = self.bin_edges.get(column, np.array([], dtype=float))
        return f"q{int(np.searchsorted(edges, float(numeric), side='right')) + 1}"

    def _keys(self, nodes: pd.DataFrame) -> pd.DataFrame:
        output = nodes[["node_id"]].copy()
        for level, fields in GROUP_LEVELS.items():
            parts: list[pd.Series] = []
            for field in fields:
                if field in self.numeric_features:
                    source = (
                        nodes[field]
                        if field in nodes
                        else pd.Series(np.nan, index=nodes.index)
                    )
                    numeric = pd.to_numeric(source, errors="coerce")
                    edges = self.bin_edges.get(field, np.array([], dtype=float))
                    bucket = pd.Series("__MISSING__", index=nodes.index, dtype=object)
                    valid = numeric.notna()
                    bucket.loc[valid] = (
                        "q"
                        + (
                            np.searchsorted(
                                edges,
                                numeric.loc[valid].to_numpy(dtype=float),
                                side="right",
                            )
                            + 1
                        ).astype(str)
                    )
                    parts.append(bucket)
                else:
                    if field in nodes:
                        parts.append(nodes[field].map(normalize_category))
                    else:
                        parts.append(
                            pd.Series(
                                "__MISSING__",
                                index=nodes.index,
                                dtype=object,
                            )
                        )
            output[level] = pd.concat(parts, axis=1).astype(str).agg("|".join, axis=1)
        return output

    def _rank_from_key(
        self,
        key_row: pd.Series,
        businesses: list[str],
    ) -> tuple[list[tuple[str, float, int]], str, str, int]:
        selected_level = "global"
        selected_key = "__GLOBAL__"
        selected_size = self.train_node_count
        counts = self.global_counts
        for level in GROUP_LEVELS:
            key = str(key_row[level])
            size = self.group_sizes.get(level, {}).get(key, 0)
            if size >= self.min_group_nodes:
                selected_level = level
                selected_key = key
                selected_size = size
                counts = {
                    business: self.group_counts.get(level, {}).get(
                        (key, business), 0
                    )
                    for business in businesses
                }
                break
        ranked = sorted(
            (
                (
                    str(business),
                    float(counts.get(business, 0) / max(selected_size, 1)),
                    int(counts.get(business, 0)),
                )
                for business in businesses
            ),
            key=lambda item: (
                -item[1],
                -item[2],
                -self.global_counts.get(item[0], 0),
                item[0],
            ),
        )
        return ranked, selected_level, selected_key, selected_size

    def rank(
        self,
        node: pd.Series,
        businesses: list[str],
    ) -> tuple[list[tuple[str, float, int]], str, str, int]:
        row = pd.DataFrame([node])
        row.insert(0, "node_id", str(node.name))
        key_frame = self._keys(row)
        return self._rank_from_key(key_frame.iloc[0], businesses)

    def rank_nodes(
        self,
        nodes: pd.DataFrame,
        businesses: list[str],
    ) -> tuple[
        dict[str, list[tuple[str, float, int]]],
        dict[str, tuple[str, str, int]],
    ]:
        keys = self._keys(nodes).set_index("node_id")
        rankings: dict[str, list[tuple[str, float, int]]] = {}
        metadata: dict[str, tuple[str, str, int]] = {}
        for node_id, key_row in keys.iterrows():
            ranked, level, key, group_size = self._rank_from_key(
                key_row,
                businesses,
            )
            node_id = str(node_id)
            rankings[node_id] = ranked
            metadata[node_id] = (level, key, group_size)
        return rankings, metadata

    def profile_rows(
        self,
        businesses: list[str],
        business_names: dict[str, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for level, sizes in self.group_sizes.items():
            for key, group_size in sizes.items():
                for business in businesses:
                    count = self.group_counts.get(level, {}).get((key, business), 0)
                    if count <= 0:
                        continue
                    rows.append(
                        {
                            "group_level": level,
                            "group_key": key,
                            "group_node_count": int(group_size),
                            "business": business,
                            "business_name": business_names.get(business, ""),
                            "business_node_count": int(count),
                            "business_node_frequency": float(count / group_size),
                        }
                    )
        return rows


def scalar_number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if not pd.isna(value) else np.nan


def health_check(row: pd.Series) -> tuple[str, list[str]]:
    blocked: list[str] = []
    unknown: list[str] = []
    for column, (threshold, _comparator, label) in HEALTH_RULES.items():
        value = scalar_number(row, column)
        if np.isnan(value):
            unknown.append(column)
        elif value > threshold:
            blocked.append(f"{label}({value:.4g}>{threshold:g})")
    for column, (threshold, label) in HEALTH_MIN_RULES.items():
        value = scalar_number(row, column)
        if np.isnan(value):
            unknown.append(column)
        elif value < threshold:
            blocked.append(f"{label}({value:.4g}<{threshold:g})")
    for column, label in HEALTH_ZERO_RULES.items():
        value = scalar_number(row, column)
        if np.isnan(value):
            unknown.append(column)
        elif value > 0:
            blocked.append(f"{label}({value:.4g}>0)")
    for column, (threshold, label) in HEALTH_REQUIRED_RULES.items():
        value = scalar_number(row, column)
        if column == "zfs_pool_online" and np.isnan(value):
            state = normalize_category(row.get("zfs_pool_state"))
            if state.lower() in {"online", "true", "1"}:
                value = threshold
        if np.isnan(value):
            unknown.append(column)
        elif value != threshold:
            blocked.append(f"{label}({value:.4g}!={threshold:g})")
    for column in sorted(HEALTH_MANUAL_ONLY):
        value = scalar_number(row, column)
        if np.isnan(value):
            unknown.append(column)
        else:
            unknown.append(f"{column}_threshold_not_configured")
    if blocked:
        return "blocked", sorted(set(blocked))
    if unknown:
        return "needs_manual_check", [
            f"指标缺失或缺少可审计阈值:{','.join(sorted(set(unknown)))}"
        ]
    return "pass", []


def format_model_rank(
    ranked: list[tuple[str, float]],
    business_names: dict[str, str],
) -> str:
    return ";".join(
        f"{business}:{business_names.get(business, '')}:{probability:.6f}"
        for business, probability in ranked
    )


def format_group_rank(
    ranked: list[tuple[str, float, int]],
    business_names: dict[str, str],
) -> str:
    return ";".join(
        f"{business}:{business_names.get(business, '')}:{frequency:.6f}:{count}"
        for business, frequency, count in ranked
    )


def summarize_hits(hits: list[bool]) -> dict[str, Any]:
    if not hits:
        return {"n": 0, "rate": None}
    return {"n": len(hits), "rate": float(np.mean(hits))}


def bootstrap_difference(
    model_hits: list[bool],
    baseline_hits: list[bool],
    seed: int,
) -> dict[str, Any]:
    if len(model_hits) != len(baseline_hits) or not model_hits:
        return {"difference": None, "ci95_low": None, "ci95_high": None}
    model_array = np.asarray(model_hits, dtype=float)
    baseline_array = np.asarray(baseline_hits, dtype=float)
    observed = float(np.mean(model_array - baseline_array))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(model_array), size=(BOOTSTRAP_ROUNDS, len(model_array)))
    samples = np.mean((model_array - baseline_array)[index], axis=1)
    return {
        "difference": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def evaluate(
    test_ids: list[str],
    model_rankings: dict[str, list[tuple[str, float]]],
    group_rankings: dict[str, list[tuple[str, float, int]]],
    global_business: str,
    observed: dict[str, set[str]],
    dominant: dict[str, str],
    candidates: set[str],
    seed: int,
) -> dict[str, Any]:
    model_observed: list[bool] = []
    group_observed: list[bool] = []
    global_observed: list[bool] = []
    model_observed_top3: list[bool] = []
    group_observed_top3: list[bool] = []
    model_dominant: list[bool] = []
    group_dominant: list[bool] = []
    global_dominant: list[bool] = []
    model_dominant_top3: list[bool] = []
    group_dominant_top3: list[bool] = []
    observed_recall_at_3: list[float] = []

    for node_id in test_ids:
        actual = observed.get(node_id, set()) & candidates
        if not actual:
            continue
        model_top = [item[0] for item in model_rankings[node_id]]
        group_top = [item[0] for item in group_rankings[node_id]]
        model_observed.append(model_top[0] in actual)
        group_observed.append(group_top[0] in actual)
        global_observed.append(global_business in actual)
        model_observed_top3.append(bool(set(model_top[:3]) & actual))
        group_observed_top3.append(bool(set(group_top[:3]) & actual))
        observed_recall_at_3.append(len(set(model_top[:3]) & actual) / len(actual))
        target = dominant.get(node_id)
        if target in candidates:
            model_dominant.append(model_top[0] == target)
            group_dominant.append(group_top[0] == target)
            global_dominant.append(global_business == target)
            model_dominant_top3.append(target in model_top[:3])
            group_dominant_top3.append(target in group_top[:3])

    return {
        "test_nodes": len(test_ids),
        "observed_business_evaluable_nodes": len(model_observed),
        "dominant_business_evaluable_nodes": len(model_dominant),
        "model_observed_hit_rate_at_1": summarize_hits(model_observed),
        "configuration_frequency_observed_hit_rate_at_1": summarize_hits(
            group_observed
        ),
        "global_observed_hit_rate_at_1": summarize_hits(global_observed),
        "model_observed_hit_rate_at_3": summarize_hits(model_observed_top3),
        "configuration_frequency_observed_hit_rate_at_3": summarize_hits(
            group_observed_top3
        ),
        "model_observed_recall_at_3": float(np.mean(observed_recall_at_3))
        if observed_recall_at_3
        else None,
        "model_dominant_hit_rate_at_1": summarize_hits(model_dominant),
        "configuration_frequency_dominant_hit_rate_at_1": summarize_hits(
            group_dominant
        ),
        "global_dominant_hit_rate_at_1": summarize_hits(global_dominant),
        "model_dominant_hit_rate_at_3": summarize_hits(model_dominant_top3),
        "configuration_frequency_dominant_hit_rate_at_3": summarize_hits(
            group_dominant_top3
        ),
        "model_minus_global_observed": bootstrap_difference(
            model_observed, global_observed, seed
        ),
        "model_minus_configuration_observed": bootstrap_difference(
            model_observed, group_observed, seed + 1
        ),
        "model_minus_global_dominant": bootstrap_difference(
            model_dominant, global_dominant, seed + 2
        ),
        "model_minus_configuration_dominant": bootstrap_difference(
            model_dominant, group_dominant, seed + 3
        ),
    }


def run_seed(
    seed: int,
    nodes: pd.DataFrame,
    occurrences: pd.DataFrame,
    dominant: dict[str, str],
    observed: dict[str, set[str]],
    categorical: list[str],
    numeric: list[str],
    business_names: dict[str, str],
    ambiguous: set[str],
    build_recommendations: bool,
) -> dict[str, Any]:
    node_ids = sorted(nodes["node_id"].astype(str).unique())
    train_ids, test_ids = split_node_ids(node_ids, seed)
    train_occurrences = occurrences[
        occurrences["node_id"].isin(train_ids)
    ].drop_duplicates(["node_id", "business"])
    support = train_occurrences.groupby("business")["node_id"].nunique()
    businesses = sorted(
        support[support >= MIN_BUSINESS_NODE_SUPPORT].index.astype(str).tolist()
    )
    if len(businesses) < 2:
        raise RuntimeError(
            f"seed={seed} 的高频业务类别不足2个，实际为 {len(businesses)}"
        )
    candidate_set = set(businesses)
    classifier = BusinessOccurrenceClassifier(categorical, numeric, seed)
    classifier.fit(nodes, train_ids, occurrences, businesses)
    all_nodes = nodes.set_index("node_id").loc[node_ids].reset_index()
    probabilities = classifier.predict_proba(all_nodes)
    model_rankings: dict[str, list[tuple[str, float]]] = {}
    for node_id, row in zip(node_ids, probabilities):
        model_rankings[node_id] = sorted(
            [(business, float(row[index])) for index, business in enumerate(businesses)],
            key=lambda item: (-item[1], item[0]),
        )

    frequency = ConfigurationFrequency(
        MIN_CONFIG_GROUP_NODES,
        [column for column in GROUP_NUMERIC_FEATURES if column in nodes],
    )
    frequency.fit(nodes, train_ids, occurrences, businesses)
    node_index = nodes.set_index("node_id")
    group_rankings, group_meta = frequency.rank_nodes(
        node_index.reset_index(),
        businesses,
    )
    global_business = max(
        businesses,
        key=lambda business: (
            frequency.global_counts.get(business, 0),
            business,
        ),
    )
    metrics = evaluate(
        test_ids,
        model_rankings,
        group_rankings,
        global_business,
        observed,
        dominant,
        candidate_set,
        seed,
    )
    result: dict[str, Any] = {
        "seed": seed,
        "train_nodes": len(train_ids),
        "test_nodes": len(test_ids),
        "train_occurrence_rows": int(len(train_occurrences)),
        "candidate_business_count": len(businesses),
        "candidate_businesses": businesses,
        "candidate_business_support": {
            business: int(support.get(business, 0)) for business in businesses
        },
        "global_business": global_business,
        "metrics": metrics,
    }
    if build_recommendations:
        recommendation_rows: list[dict[str, Any]] = []
        split = {node_id: "train" for node_id in train_ids}
        split.update({node_id: "test" for node_id in test_ids})
        for node_id in node_ids:
            node_row = node_index.loc[node_id]
            model_rank = model_rankings[node_id]
            group_rank = group_rankings[node_id]
            level, key, group_size = group_meta[node_id]
            status, reasons = health_check(node_row)
            provisional = model_rank[0][0]
            recommended = provisional if status == "pass" else ""
            row: dict[str, Any] = {
                "node_id": node_id,
                "split": split[node_id],
                "attribute_day": str(node_row.get("attribute_day", "")),
                "recommended_business": recommended,
                "recommended_business_provisional": provisional,
                "model_probability": model_rank[0][1],
                "model_top5": format_model_rank(model_rank[:5], business_names),
                "configuration_frequency_business": group_rank[0][0],
                "configuration_frequency": group_rank[0][1],
                "configuration_business_node_count": group_rank[0][2],
                "configuration_group_level": level,
                "configuration_group_key": key,
                "configuration_group_nodes": group_size,
                "configuration_top5": format_group_rank(
                    group_rank[:5], business_names
                ),
                "recommendation_status": status,
                "health_reasons": "|".join(reasons),
                "target_dominant_business": dominant.get(node_id, ""),
                "target_is_ambiguous": node_id in ambiguous,
            }
            for column in categorical + numeric:
                row[column] = node_row.get(column)
            recommendation_rows.append(row)
        result["recommendations"] = pd.DataFrame(recommendation_rows)
        result["frequency_profiles"] = pd.DataFrame(
            frequency.profile_rows(businesses, business_names)
        )
    return result


def build_business_frequency(
    occurrences: pd.DataFrame,
    business_names: dict[str, str],
) -> pd.DataFrame:
    distinct = occurrences.drop_duplicates(["node_id", "business"])
    frequency = (
        distinct.groupby("business")
        .agg(
            node_count=("node_id", "nunique"),
            active_days_sum=("business_active_days", "sum"),
        )
        .reset_index()
    )
    frequency["business_name"] = frequency["business"].map(business_names).fillna("")
    frequency["node_share"] = frequency["node_count"] / occurrences["node_id"].nunique()
    return frequency.sort_values(
        ["node_count", "active_days_sum", "business"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def pct(value: Any) -> str:
    return "无" if value is None else f"{float(value):.2%}"


def write_report(metrics: dict[str, Any]) -> None:
    primary = metrics["primary"]
    test = primary["metrics"]
    status_text = "、".join(
        f"{status} {count:,}个"
        for status, count in metrics["recommendation_status_counts"].items()
    )
    lines = [
        "# 配置分类业务推荐报告",
        "",
        "## 结论",
        "",
        "当前分支只做业务出现分类，不预测成本、收入或利润金额。",
        f"测试集节点数为 {test['test_nodes']:,}，以宽表中节点历史出现天数最多的业务作为评估标签；"
        f"唯一主业务可评估节点为 {test['dominant_business_evaluable_nodes']:,}。",
        f"分类模型对节点历史主业务的命中率@1为 "
        f"{pct(test['model_dominant_hit_rate_at_1']['rate'])}，命中率@3为 "
        f"{pct(test['model_dominant_hit_rate_at_3']['rate'])}。",
        f"对节点历史出现业务集合的命中率@1为 "
        f"{pct(test['model_observed_hit_rate_at_1']['rate'])}，命中率@3为 "
        f"{pct(test['model_observed_hit_rate_at_3']['rate'])}。",
        f"训练集中的全局高频业务基线为 {primary['global_business']}；"
        "配置频率规则按相似配置节点的业务出现节点占比排序。",
        "",
        "## 数据口径",
        "",
        f"- 节点属性：{NODES_FILE.name}，来源为节点上线日前一天的属性快照。",
        f"- 业务来源：{OUTCOMES_FILE.name}，来源表为 test.node_day_ops_wide_full。",
        "- 业务频率口径：同一节点-业务只计一次，按宽表中该业务出现的不同天数排序；"
        "配置分组频率按相似配置节点中出现该业务的节点数 / 分组节点数计算。",
        f"- 节点总数：{metrics['data']['node_count']:,}；节点-业务记录："
        f"{metrics['data']['occurrence_rows']:,}；历史业务数：{metrics['data']['business_count']:,}。",
        f"- 每个节点历史业务数最少 {metrics['data']['min_businesses_per_node']} 个，"
        f"少于3个的节点 {metrics['data']['nodes_below_three_businesses']:,}。",
        f"- 高频候选业务：训练集中至少出现在 {MIN_BUSINESS_NODE_SUPPORT} 个不同节点，"
        f"主种子候选数为 {primary['candidate_business_count']}。",
        f"- 业务主标签平局节点：{metrics['data']['ambiguous_node_count']:,}，"
        "这些节点不参与唯一主业务命中率，但仍参与多标签业务出现分类。",
        "",
        "## 分类模型",
        "",
        "- 算法：One-vs-Rest LogisticRegression 多标签分类器。",
        f"- 参数：C={os.environ.get('CLASSIFICATION_LOGISTIC_C', '1.0')}，"
        f"max_iter={os.environ.get('CLASSIFICATION_LOGISTIC_MAX_ITER', '600')}，"
        "class_weight=balanced，solver=liblinear。",
        "- 输入：节点固有类别属性、CPU 核数、内存容量、总盘/HDD/SSD/系统盘容量、名义带宽；"
        "不输入成本、收入、利润、在线时长、在线率、带宽利用率、运行时 CPU/内存/磁盘、"
        "RTT、丢包、重传、SMART、ZFS 或业务流量。",
        f"- 有效类别特征数：{len(metrics['features']['effective_categorical'])}；"
        f"有效数值特征数：{len(metrics['features']['effective_numeric'])}。",
        "- 推荐结果的 model_top5 是分类概率排序；configuration_top5 是相似配置节点的"
        "历史业务出现频率排序，二者同时输出用于解释。",
        "",
        "## 测试集效果",
        "",
        "| 方法 | 观察到业务命中@1 | 观察到业务命中@3 | 主业务命中@1 | 主业务命中@3 |",
        "|---|---:|---:|---:|---:|",
        f"| 分类模型 | {pct(test['model_observed_hit_rate_at_1']['rate'])} | "
        f"{pct(test['model_observed_hit_rate_at_3']['rate'])} | "
        f"{pct(test['model_dominant_hit_rate_at_1']['rate'])} | "
        f"{pct(test['model_dominant_hit_rate_at_3']['rate'])} |",
        f"| 配置频率规则 | {pct(test['configuration_frequency_observed_hit_rate_at_1']['rate'])} | "
        f"{pct(test['configuration_frequency_observed_hit_rate_at_3']['rate'])} | "
        f"{pct(test['configuration_frequency_dominant_hit_rate_at_1']['rate'])} | "
        f"{pct(test['configuration_frequency_dominant_hit_rate_at_3']['rate'])} |",
        f"| 全局高频基线 | {pct(test['global_observed_hit_rate_at_1']['rate'])} | "
        "不适用 | "
        f"{pct(test['global_dominant_hit_rate_at_1']['rate'])} | 不适用 |",
        "",
        "差值的 bootstrap 95% 区间：",
        "",
        f"- 分类模型 - 全局基线，观察到业务命中@1："
        f"{pct(test['model_minus_global_observed']['difference'])}，"
        f"95% CI [{pct(test['model_minus_global_observed']['ci95_low'])}, "
        f"{pct(test['model_minus_global_observed']['ci95_high'])}]。",
        f"- 分类模型 - 配置频率规则，观察到业务命中@1："
        f"{pct(test['model_minus_configuration_observed']['difference'])}，"
        f"95% CI [{pct(test['model_minus_configuration_observed']['ci95_low'])}, "
        f"{pct(test['model_minus_configuration_observed']['ci95_high'])}]。",
        f"- 分类模型 - 全局基线，主业务命中@1："
        f"{pct(test['model_minus_global_dominant']['difference'])}，"
        f"95% CI [{pct(test['model_minus_global_dominant']['ci95_low'])}, "
        f"{pct(test['model_minus_global_dominant']['ci95_high'])}]。",
        f"- 分类模型 - 配置频率规则，主业务命中@1："
        f"{pct(test['model_minus_configuration_dominant']['difference'])}，"
        f"95% CI [{pct(test['model_minus_configuration_dominant']['ci95_low'])}, "
        f"{pct(test['model_minus_configuration_dominant']['ci95_high'])}]。",
        "",
        "## 多随机种子",
        "",
        "| seed | 候选业务数 | 观察到业务命中@1 | 主业务命中@1 | 配置频率主业务命中@1 |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {result['seed']} | {result['candidate_business_count']} | "
            f"{pct(result['metrics']['model_observed_hit_rate_at_1']['rate'])} | "
            f"{pct(result['metrics']['model_dominant_hit_rate_at_1']['rate'])} | "
            f"{pct(result['metrics']['configuration_frequency_dominant_hit_rate_at_1']['rate'])} |"
            for result in metrics["multi_seed_validation"]
        ],
        f"主业务命中率均值 {pct(metrics['multi_seed_summary']['dominant_at_1_mean'])}，"
        f"标准差 {pct(metrics['multi_seed_summary']['dominant_at_1_std'])}，"
        f"范围 [{pct(metrics['multi_seed_summary']['dominant_at_1_min'])}, "
        f"{pct(metrics['multi_seed_summary']['dominant_at_1_max'])}]；"
        f"观察到业务命中率均值 {pct(metrics['multi_seed_summary']['observed_at_1_mean'])}，"
        f"范围 [{pct(metrics['multi_seed_summary']['observed_at_1_min'])}, "
        f"{pct(metrics['multi_seed_summary']['observed_at_1_max'])}]。",
        "",
        "上述是节点级留出回测，不是线上因果效果；配置频率与节点历史业务出现存在选择偏差，"
        "不能解释为把业务切换到推荐业务后一定会产生相同结果。",
        "",
        "## 健康准入",
        "",
        "CPU、内存、磁盘、重传、丢包、SMART、ZFS 等字段仍保留在节点快照中，"
        "只用于推荐前的健康状态检查。指标不合格时 recommended_business 为空并标记 blocked；"
        "指标缺失或没有可审计阈值时标记 needs_manual_check，同时保留 "
        "recommended_business_provisional 供人工查看。当前状态分布为："
        f"{status_text}。",
        "",
        "## 产物",
        "",
        f"- {OUTPUT_RECOMMENDATIONS.name}：节点配置、分类推荐、配置频率证据和健康状态。",
        f"- {OUTPUT_BUSINESS_FREQUENCY.name}：宽表历史业务的节点覆盖频率。",
        f"- {OUTPUT_GROUP_FREQUENCY.name}：训练集配置分组-业务出现频率。",
        f"- {OUTPUT_METRICS.name}：机器可读数据口径、字段、分类指标和多随机种子结果。",
        "",
    ]
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc)
    nodes, outcomes = load_data()
    occurrences, dominant, ambiguous, observed = build_targets(outcomes)
    businesses_per_node = occurrences.groupby("node_id")["business"].nunique()
    if (businesses_per_node < 3).any():
        raise RuntimeError("输入数据存在历史业务少于3个的节点，拒绝改变筛选口径。")
    categorical, numeric, coverage = effective_features(nodes)
    business_names: dict[str, str] = {}
    if "business_name" in outcomes:
        names = outcomes[["business", "business_name"]].dropna()
        for row in names.itertuples(index=False):
            name = normalize_category(row.business_name)
            if name != "__MISSING__":
                business_names.setdefault(str(row.business), name)

    node_ids = sorted(nodes["node_id"].astype(str).unique())
    if not VALIDATION_SEEDS:
        raise RuntimeError("CLASSIFICATION_VALIDATION_SEEDS 不能为空。")
    primary_seed = (
        RANDOM_SEED if RANDOM_SEED in VALIDATION_SEEDS else VALIDATION_SEEDS[0]
    )
    seed_results: list[dict[str, Any]] = []
    primary_result: dict[str, Any] | None = None
    for seed in VALIDATION_SEEDS:
        print(f"fit classification seed={seed}")
        result = run_seed(
            seed,
            nodes,
            occurrences,
            dominant,
            observed,
            categorical,
            numeric,
            business_names,
            ambiguous,
            build_recommendations=seed == primary_seed,
        )
        seed_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"recommendations", "frequency_profiles"}
            }
        )
        if seed == primary_seed:
            primary_result = result
    if primary_result is None:
        raise RuntimeError("没有生成主随机种子的分类结果。")

    recommendations = primary_result["recommendations"]
    recommendations.to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    primary_result["frequency_profiles"].to_csv(
        OUTPUT_GROUP_FREQUENCY, index=False
    )
    global_frequency = build_business_frequency(outcomes, business_names)
    global_frequency.to_csv(OUTPUT_BUSINESS_FREQUENCY, index=False)
    recommendation_status_counts = (
        recommendations["recommendation_status"].value_counts().astype(int).to_dict()
    )
    observed_rates = [
        result["metrics"]["model_observed_hit_rate_at_1"]["rate"]
        for result in seed_results
    ]
    dominant_rates = [
        result["metrics"]["model_dominant_hit_rate_at_1"]["rate"]
        for result in seed_results
    ]

    metrics = {
        "generated_at": dt.datetime.now(
            dt.timezone(dt.timedelta(hours=8))
        ).isoformat(timespec="seconds"),
        "source": {
            "database": "yzh-starrocks",
            "schema": "test",
            "table": "node_day_ops_wide_full",
            "local_nodes_export": NODES_FILE.name,
            "local_business_export": OUTCOMES_FILE.name,
        },
        "target": {
            "type": "business_occurrence_classification",
            "positive_label": "节点历史上出现过该业务",
            "dominant_evaluation_label": "节点历史业务出现天数最多的业务",
            "tie_policy": "主业务平局不强行指定唯一标签",
            "amount_prediction": False,
            "amount_columns_loaded": [],
        },
        "data": {
            "node_count": int(len(node_ids)),
            "occurrence_rows": int(len(occurrences)),
            "business_count": int(outcomes["business"].nunique()),
            "ambiguous_node_count": int(len(ambiguous)),
            "dominant_node_count": int(len(dominant)),
            "min_businesses_per_node": int(businesses_per_node.min()),
            "nodes_below_three_businesses": int((businesses_per_node < 3).sum()),
            "health_fields_retained": sorted(HEALTH_MISSING_FIELDS),
        },
        "features": {
            "declared_categorical": INTRINSIC_CATEGORICAL_FEATURES,
            "declared_numeric": INTRINSIC_NUMERIC_FEATURES,
            "effective_categorical": categorical,
            "effective_numeric": numeric,
            "coverage": coverage,
            "excluded_from_model": sorted(HEALTH_MISSING_FIELDS),
            "policy": "只使用上线前节点固有配置；运行时指标只做健康准入",
        },
        "configuration_frequency": {
            "group_levels": GROUP_LEVELS,
            "min_group_nodes": MIN_CONFIG_GROUP_NODES,
            "frequency_denominator": "相似配置分组中的不同节点数",
        },
        "model": {
            "algorithm": "OneVsRestClassifier(LogisticRegression)",
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": float(os.environ.get("CLASSIFICATION_LOGISTIC_C", "1.0")),
            "max_iter": int(
                os.environ.get("CLASSIFICATION_LOGISTIC_MAX_ITER", "600")
            ),
            "primary_seed": primary_seed,
            "validation_seeds": list(VALIDATION_SEEDS),
        },
        "primary": {
            key: value
            for key, value in primary_result.items()
            if key not in {"recommendations", "frequency_profiles"}
        },
        "multi_seed_validation": seed_results,
        "multi_seed_summary": {
            "observed_at_1_mean": float(np.mean(observed_rates)),
            "observed_at_1_std": float(np.std(observed_rates, ddof=1)),
            "observed_at_1_min": float(np.min(observed_rates)),
            "observed_at_1_max": float(np.max(observed_rates)),
            "dominant_at_1_mean": float(np.mean(dominant_rates)),
            "dominant_at_1_std": float(np.std(dominant_rates, ddof=1)),
            "dominant_at_1_min": float(np.min(dominant_rates)),
            "dominant_at_1_max": float(np.max(dominant_rates)),
        },
        "recommendation_status_counts": recommendation_status_counts,
        "outputs": [
            OUTPUT_RECOMMENDATIONS.name,
            OUTPUT_BUSINESS_FREQUENCY.name,
            OUTPUT_GROUP_FREQUENCY.name,
            OUTPUT_METRICS.name,
            OUTPUT_REPORT.name,
        ],
        "elapsed_seconds": (
            dt.datetime.now(dt.timezone.utc) - started
        ).total_seconds(),
    }
    OUTPUT_METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(metrics)
    print(
        json.dumps(
            {
                "recommendations": str(OUTPUT_RECOMMENDATIONS),
                "business_frequency": str(OUTPUT_BUSINESS_FREQUENCY),
                "group_frequency": str(OUTPUT_GROUP_FREQUENCY),
                "metrics": str(OUTPUT_METRICS),
                "report": str(OUTPUT_REPORT),
                "primary_test": primary_result["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
