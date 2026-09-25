# Crypto Barra 风格暴露分析

使用本项目 factor_common.DataProvider 和 data/crypto_quant.h5，分析日频 alpha
与九类风格的相关性、联合回归和中性化残差。这是暴露诊断，不是完整 Barra 风险模型。

## Python / Notebook

从仓库根目录运行：

```python
from factor_common import FactorManager
from barra.crypto_barra_exposure import BarraConfig, analyze_and_write

fm = FactorManager()
result = fm.evaluate(
    "factor_analyse/factor_mining/example_momentum.py",
    params={"start": "2024-02-01", "end": "2024-03-15", "n_groups": 5},
)
barra = analyze_and_write(result, cfg=BarraConfig(min_count=20), label="example_momentum")
print(barra["style_summary"])
```

默认输出 reports/barra/<label>/。自定义数据源时，给两个入口传相同 h5_path。
支持日期×币种宽表和 date/instrument/factor 长表；拒绝重复键、非 UTC 日频日期。
jupyter_barra_example.py 提供完整函数，比较共同样本上的原始因子与残差；
导入该示例不会执行回测。

## CLI

```bash
./.venv/bin/python -m barra.crypto_barra_exposure \
  --factor-parquet data/factor_results/example_momentum.parquet \
  --h5 data/crypto_quant.h5 --start 2024-02-01 --end 2024-03-15 \
  --out-dir reports/barra/example_momentum --min-count 20
```

支持直接执行脚本；--result-pkl 是兼容入口，只应读取自己信任的 pickle，
与 --factor-parquet 互斥。没有有效回归日时仍输出诊断，退出码为 2。

## 计算口径

| 风格 | 定义 |
|---|---|
| size | 60 日平均 quote_volume 的对数；成交额代理，不是真实市值 |
| liquidity | 20 日平均 quote_volume 的对数 |
| beta | 60 日市场 beta；协方差/方差使用相同有效日期与 ddof=0 |
| momentum_5d / momentum_20d | 5 / 20 日价格涨跌幅 |
| reversal_1d | 当日收益的相反数 |
| volatility | 20 日收益标准差 × sqrt(365) |
| residual_volatility | 当日收益减当日滚动 beta × 市场收益，再计算 20 日波动 |
| funding | 日内结算费率平均值的 3 日滚动均值，不同于最后一笔费率 |

所有窗口为自然日。市场收益使用历史当日 Top50 有效收益等权平均，
不依赖 alpha 覆盖范围。暴露与 alpha 仅保留当日成分。
缺失价格不填充，资金费缺失不填零；零成交额按不可用处理，避免 log(0)。
预热涵盖 beta 与残差波动的嵌套窗口。

窗口可通过 BarraConfig 设置。beta/size 至少半窗口且默认至少 10 个值，
流动性/波动至少半窗口且默认至少 5 个值（均不超过窗口长度），funding 至少 1 个值。
当前接口缺少 open_interest/premium_close，开启 include_crypto_optional
时明确警告并记录跳过字段，不吞掉其他读取异常。

每日在 alpha 有效样本内按 1% 截尾并标准化。相关性使用成对完整样本，
联合回归包含截距，门槛为 max(min_count, 风格数+3)。
样本不足或秩不足时保留诊断，系数与残差为缺失。
条件数用于识别共线性，尤其注意 size/liquidity。
系数 t 值未经时间自相关修正，仅作描述统计。

## 输出

- style_summary.csv：相关性与系数摘要。
- daily_style_corr.csv：每日样本数、Pearson/Spearman。
- daily_barra_regression.csv：n、alpha_n、R²、系数、秩、条件数和 status。
- alpha_daily.parquet：过滤并对齐后的原始 alpha。
- alpha_barra_residual.parquet：标准化 alpha 的回归残差宽表。
- alpha_barra_residual_long.parquet：有限残差的 date/instrument/factor 长表。
- metadata.json：配置、数据源快照、口径和有效回归覆盖。

残差可传给 FactorManager.evaluate(..., factor_name=...)。对比需要保持
相同有效样本、日期、方向、费用和调仓参数。外部 DataFrame 在 FactorManager
中仍标记因果性未验证；Barra 截断验证不能证明输入 alpha 本身没有未来信息。
数据读取使用 as_of；H5 在运行期间发生变化会中止输出。

## 验证

```bash
./.venv/bin/python -m pytest tests/test_barra_exposure.py tests/test_barra_integration.py -q
./.venv/bin/python -m compileall -q barra
```

真实数据验收（生成暴露、cutoff 证据，以及共同样本的原始/残差 HTML 报告）：

```bash
./.venv/bin/python -m barra.verify --factor-parquet data/factor_results/example_momentum.parquet --out-dir reports/barra/verification
```

结果记录在 verification.json。换用负向因子时指定 --factor-direction -1。
验证包含所有九个暴露和残差的两个 cutoff；输入缓存本身的因果性单独标注未验证。
