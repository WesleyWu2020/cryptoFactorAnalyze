"""特异波动率折价因子（Idiosyncratic Volatility / Ang et al. 2006 IVOL）。

公式（Iter1 版，ACCEPTED 2026-09-14；Iter2~5 全部 REJECTED 后回滚至此）：

    ret_i(t)  = close.pct_change(1)
    mkt(t)    = 全面板等权市场收益（截面均值，只用 t 日及历史）
    IVOL_i(t) = sqrt( var_60(ret_i) - cov_60(ret_i,mkt)^2 / var_60(mkt) )
    factor    = IVOL_i(t)   （factor_direction=-1：高特异波动做空）

经济机制（Ang, Hodrick, Xing, Zhang 2006）：高特异波动资产是彩票型投机标的，
被博彩偏好资金哄抬至高估，后续收益系统性更低；剔除市场共变成分后，剩下的
纯异质风险仍被错误定价。crypto 永续的散户结构放大该折价。

计算只使用当日及历史数据，无未来函数。截面 MAD 去极值与按日 rank 由
factor_common 框架的 ``mad_rank`` 预处理完成，这里返回原始因子值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "idio_vol", "statement": "60d 市场模型残差 std，高=剔除共变后的纯彩票型投机波动"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [],
    "direction": {"id": "premium", "statement": "IVOL 折价：高特异波动被高估做空、低特异波动做多（factor_direction=-1）"},
    "output":    {"id": "rank", "statement": "截面 rank（mad_rank 预处理）"},
    "mechanism": "博彩偏好资金哄抬高特异波动标的至高估；总波动中的市场共变成分是"
                 "正常风险定价，残差成分才是纯彩票性错误定价。",
    "fields":    ["close"],
    "expected_horizon": "1d",
    "invalidation": "(1) IS |rankIC| < 0.02 或符号与 direction 相反；"
                    "(2) trading_net sharpe < 1；"
                    "(3) 与 range_level_premium_20d 演化到 |rho| >= 0.85（残差化无增量）；"
                    "(4) OOS 反号或衰减 >50%",
    "search_mode": "explore",
    "semantic_key": "idio_vol|-|-|premium|rank",
}

ITER_NOTE: dict = {
    "op_type":    "modify_factor",
    "parent_iter": "Iter5（REJECTED 2026-09-15：MAD 稳健估计量，ICIR 0.2768 与 Iter1 持平、"
                  "net sharpe 0.951/maxdd 0.260 双 FAIL）——已回滚至 Iter1 ACCEPTED 版本",
    "hypothesis": "Iter2~5 四轮单点迭代（EWMA hl20/hl40、window 120、MAD 估计量）遍历了"
                  "时间加权×窗口×估计量三个维度的局部邻域，ICIR 上限被锁定在 0.31 附近"
                  "（EWMA hl20：0.309 但 sharpe 0.75 死）；IC 噪声是信号内生的，"
                  "非估计误差。结论：维持 Iter1 版本为最终 best，不再消耗 OOS 预算。",
    "change":     "回滚：恢复 Iter1 恒等式版 sqrt(var_60 - cov_60^2/var_60(mkt))，"
                  "window=60，warmup_bars=70（与本文件当前内容一致，无代码变更）。",
    "expected":   "维持 Iter1 已验证指标：IS rankIC -0.062/ICIR -0.276/t -2.89，"
                  "net +22.9%/yr、sharpe 1.011、maxdd 0.209、turnover 0.118。",
    "reasoning":  "迭代轨迹审查：四轮改动均为估计技术（非经济机制），无一构成身份漂移；"
                  "ICIR≥0.5 的目标在 level 型 IVOL 上不可达（信号内生噪声），"
                  "继续迭代只会滑向全样本过拟合。OOS 预算剩 2 次冻结不动。",
    "semantic_mutation": None,
    "oos_result": "（Iter1）pass：IS -0.0619 → OOS -0.0559（同号，衰减 9.7%），"
                  "OOS net +3.83%/yr > 0。",
    "corr_note":  "Iter1 正式门控 max|rho|=0.726 vs market_coupling_factor（WARN 区已论证）。",
}


TYPE = "regular"

META = {
    "factor_name": "Idio_Vol_Factor",
    "author": "local",
    "level": "daily",
    "category": "volatility",
    "description": "60d market-model residual volatility, IVOL discount (direction -1)",
}

SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 70,
    "preprocessing": "mad_rank",
    "params": {"window": 60},
    "factor_direction": -1,
}


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回逐日市场模型残差波动率矩阵。

    ``data_ctx`` 为 date(升序) × instrument 的矩阵字典；``rolling`` 与
    ``pct_change`` 沿时间轴（axis 0）进行，每个值只依赖当日及之前的数据。
    """

    window = SETTING["params"]["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("SETTING.params.window must be an integer >= 2")

    close = data_ctx["close"].astype("float64")
    ret = close.pct_change(periods=1)
    mkt = ret.mean(axis=1)
    var_i = ret.rolling(window=window, min_periods=window).var()
    cov = ret.rolling(window=window, min_periods=window).cov(mkt)
    mvar = mkt.rolling(window=window, min_periods=window).var()
    resid_var = var_i - cov.pow(2).div(mvar.replace(0.0, np.nan), axis=0)
    # 数值地板：浮点误差下可能出现微小负值
    return np.sqrt(resid_var.clip(lower=0.0))
