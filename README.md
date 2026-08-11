# 配置分类业务推荐

当前 yzx-01 分支将推荐任务调整为分类问题：

给定节点上线前可确定的固有配置，推荐历史上相似配置节点在
test.node_day_ops_wide_full 中经常出现的业务。

当前版本不预测成本、收入或利润金额。

## 数据口径

- 业务来源：Superset 数据库 yzh-starrocks 的 test.node_day_ops_wide_full。
- 节点属性：multibusiness_nodes.csv，取节点上线日前一天的属性快照。
- 业务出现结果：multibusiness_outcomes.csv，只使用 node_id、business、
  business_active_days 和 business_name。
- 节点筛选：已上线且历史至少出现 3 个不同业务；当前数据校验为 26,623 个节点，
  少于 3 个业务的节点为 0。
- 训练/测试：按节点随机切分 80/20，验证随机种子为 42、52、62、72、82。
- 高频候选业务：训练集中至少出现在 30 个不同节点的业务。
- 业务频率：同一节点-业务只计一次，配置分组频率为相似配置节点中出现该业务的节点数
  除以配置分组节点数。
- 主业务评估标签：节点历史出现天数最多的业务；并列时不强行指定唯一标签。

## 模型和特征

- 模型：OneVsRestClassifier(LogisticRegression) 多标签分类器。
- 分类正例：节点历史上出现过该业务。
- 输入字段：地域、运营商、设备/型号、交付/资源/拨号/NAT 配置、IPv6/UPnP
  等固有配置，以及名义带宽、CPU 核数、内存容量、总盘/HDD/SSD/系统盘容量。
- 不进入模型：成本、收入、利润、历史在线时长、在线率、带宽利用率、运行时
  CPU/内存/磁盘、RTT、丢包、重传、SMART、ZFS 和业务流量。
- CPU、内存、磁盘、重传、SMART、ZFS 等运行指标仍保留在节点快照中，只用于健康准入；
  不合格时阻断，缺失时转人工检查。

## 运行

使用项目环境执行：

    /Users/nany/.workbuddy/binaries/python/envs/default/bin/python rebuild_multibusiness_model.py

兼容入口 rebuild_multibusiness_model.py 当前只调用
classification_business_recommender.py，不再执行金额回归。

## 主要产物

- classification_recommendations.csv：节点配置、分类推荐、相似配置频率证据和健康状态。
- classification_business_frequency.csv：宽表历史业务的节点覆盖频率。
- classification_configuration_business_frequency.csv：训练集配置分组-业务出现频率。
- classification_metrics.json：数据口径、字段边界、分类指标和多随机种子结果。
- 配置分类业务推荐报告.md：中文分析报告。

仓库中原有 multibusiness_* 金额/利润结果文件保留作为历史数据快照，不属于当前
yzx-01 分支的训练目标或当前推荐入口。
