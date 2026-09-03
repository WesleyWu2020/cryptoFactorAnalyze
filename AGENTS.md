# AGENTS.md

## Repo purpose
该项目用于加密货币因子研究与回测分析，围绕 Binance 交易对数据进行因子构建、因子挖掘与报告生成。  
当前主要目标是持续扩展高质量因子（含规则因子与 ML 因子），并稳定产出可比较的因子分析报告（`reports/`）。

## Repo layout
- `data/`: 数据获取与预处理脚本（如币对列表、K 线抓取、因子数据目录）
- `factor_analyse/`: 因子分析主流程、因子配置、因子挖掘、ML/GA 模块
- `reports/`: 生成的 HTML 因子分析报告
- `README.md`: 基础运行顺序说明

## How to run
- install: `pip install -r requirements.txt`
- dev:
  - `python data/list-from-binance.py`
  - `python data/capture_data_binance.py`
  - `python factor_analyse/main.py --list`
  - `python factor_analyse/main.py <factor_type> [rebalance_period]`
- test: 当前仓库未配置统一测试入口；如新增逻辑需补充最小可复现验证脚本或测试用例
- lint: 当前仓库未配置统一 lint 工具；提交前至少保证改动脚本可运行并通过基础语法检查
- build: 无独立构建步骤，执行 `factor_analyse/main.py` 后在 `reports/` 产出报告
- 命令规范：默认在仓库根目录执行，并优先使用 `./.venv/bin/python` 运行 Python 命令，避免相对路径导致的解释器错误

## Engineering conventions
- 优先复用已有 `factor_analyse/factor_mining/util_factor.py` 与现有分析流程
- 不新增重复 util，已有能力优先扩展
- 新增因子需遵循现有命名与注册方式：
  - 因子脚本放在 `factor_analyse/factor_mining/`
  - 在 `factor_analyse/factor_config.py` 注册配置
- 因子数据格式保持一致：至少包含 `date`、`instrument`、`factor` 三列
- 报告输出路径与命名遵循 `main.py` 现有规范（输出到 `reports/`）

## 因子开发固定流程（强制）
- 当用户给出新因子公式时，必须按以下顺序执行，不跳步：
1. 在 `factor_analyse/factor_mining/` 新增因子脚本（复用 `util_factor.py` 与 `operator_utils.py`，并保持无未来函数）。
2. 在 `factor_analyse/factor_config.py` 添加对应 `factor_type` 配置（至少包含 `file_prefix`、`factor_name`、`factor_direction`、`factor_desc`、`rebalance_period`）。
3. 先运行因子脚本生成因子数据（输出到 `data/factor_data/`）。
4. 再运行 `python factor_analyse/main.py <factor_type>` 生成报告（输出到 `reports/`）。
5. 返回结果时必须包含：变更文件列表、执行命令、核心输出路径、未来函数检查结论与验证步骤。

## Do-not rules
- 不修改 `factor_miner` 的输入输出契约（列约定、核心调用方式），除非明确说明
- 不引入新的大型框架或重型依赖，优先沿用现有技术栈
- 不为“看起来更好”做大规模重构，避免影响历史脚本可复现性
- 因子值计算阶段严禁未来函数（前视偏差）

## Future-leak（未来函数）强制规则
- 适用范围：所有 `factor_analyse/factor_mining/` 下新增或修改的因子脚本。
- 硬性要求：`factor` 列只能由 `t` 时点及历史数据（`<= t`）计算，不得直接或间接使用 `t+1` 及之后信息。
- 唯一允许使用未来数据的字段：标签列（如 `future_ret`），且仅用于评估/回测统计，不得参与 `factor` 构造、标准化、分组排序或选币逻辑。
- 禁止模式（出现在因子值链路即违规）：
  - `shift(-k)`（`k>0`）、`pct_change(...).shift(-k)`、任何负位移前瞻操作
  - `rolling(..., center=True)`（会引入未来窗口）
  - `bfill` / 反向填充在特征列上的使用
  - `merge_asof(..., direction=\"forward\")` 或等价“向前匹配未来时间”
- 可用性池/成分池约束：某日可交易标的仅可由当日或历史已知信息决定，不得使用未来日期成分。
- 提交前必须给出未来函数检查结果（至少包含）：
  - 静态检查：列出是否存在上述禁止模式，以及出现位置与用途说明
  - 动态反证：对比“全量数据计算”与“按 cutoff 截断计算”在 `date <= cutoff` 区间的因子值一致性（`max_abs_diff` 应为 0 或在浮点容忍范围内）

## Definition of done
- 功能可运行（至少完成一次端到端执行或关键路径验证）
- 相关测试通过；若无现成测试，补充“未覆盖原因 + 手工验证步骤”
- 基础检查通过（语法/导入/关键脚本运行）
- 输出变更文件列表与验证步骤

## PR / output expectations
- 先说明改了什么
- 再说明为什么这么改
- 最后给验证步骤和风险点
