import os

import numpy as np
import pandas as pd


SEMANTIC_PLAN: dict = {
    "event":     {"id": "venue_share_shift", "statement": "bybit 量/(bybit 量+binance 量) 份额 20d/120d 慢速抬升=散户场地定价权 regime 上行"},
    "context":   {"id": "-", "statement": "-"},
    "qualities": [
        {"id": "persistence", "statement": "份额双长窗均值比（20d/120d）是持续数周的迁移趋势，非单日脉冲"},
    ],
    "direction": {"id": "premium", "statement": "散户场地定价权溢价：固定符号空份额上移（散户拥挤=过度定价）多份额下移（定价权回归主流场地=neglect）"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "C1(币安 taker 买盘)与 Q1(bybit 总活动)都是单所绝对水平，共享市场级量能周期污染。"
                 "份额=相对口径：bybit/(bybit+binance) 把'投机多不多'换成'投机在哪儿定价'——"
                 "bybit 散户占比高，份额上移=该币定价权被散户场地接管=过热前兆；份额下移=定价回归币安"
                 "(机构/套利主导)=neglect。比值归一化天然剔除市场级共模，与 C1/Q1 的相关应结构性偏低。",
    "fields":    ["dollar_volume", "xbinance_quote_volume", "close"],
    "expected_horizon": "1d",
    "invalidation": "(1) prod_corr >= 0.6（与 C1 族 0.62 / Q1 的 0.5615 撞车=份额口径无增量）；"
                    "(2) |RankIC| < 0.01；"
                    "(3) ls_netir < 0.5；"
                    "(4) turnover(组合换手) > 0.25；"
                    "(5) coverage < 0.85（xbinance 字段覆盖略低于纯 bybit, 预注册略降）",
    "search_mode": "explore",
    "semantic_key": "venue_share_shift|-|persistence|premium|rank",
}

ITER_NOTE: dict = {
    "op_type": "add_factor",
    "factor_family": "sentiment",
    "parent_factor_id": None,
    "semantic_mutation": None,
    "hypothesis": "投机热度资产的'定价权位置'切面：份额迁移与绝对水平正交（比值剔除共模）。"
                  "bybit 份额上移=散户定价=未来跑输。",
    "change": "新因子。share=bybit_dollar_volume/(bybit_dollar_volume+binance_quote_volume) 日级; "
              "mig=mean(share,20d)/mean(share,120d)；factor=-(rank(mig)-0.5)，shift(1) 广播分钟级。"
              "评估走 pkl 预计算路径（时间分片瘦身）。",
    "expected": "|RankIC| 0.01~0.02, ls_netir 0.3~0.9, 换手 0.04~0.10, coverage>0.85, "
                "核心悬念=prod_corr 是否显著低于 C1 的 0.62（份额 vs 绝对水平的正交性检验）。",
    "result": "REJECTED (STATISTICAL_REJECT, 正交但薄): prod_corr=0.3378 全场最低(份额 vs 绝对水平正交性确立, "
              "vs gp_auto_20260605_045002_seed394_r34); 但 |RankIC|=0.0126 刚过门, ls_netir=0.488, 净+5.35%/yr, "
              "coverage=0.588 破 0.6 门(xbinance 字段可用性漂移, 2023 初 0.55→0.75 跳变), 2023 -23.2% 深负 "
              "(当年 IC 仍正——IC/组合背离+宇宙扩张扭曲)。2022 +18.5% 为正(与热度家族熊负相反, regime 形态不同)。"
              "六图: 组0 空腿 demeaned -0.75 干净; 组5>组9=长腿 payoff 中段化第三次出现(C2/C3 同构)。"
              "share 日级中位 0.197(bybit 占两所量 ~20%), 口径健康。结论: 信号真实但 IC 厚度只有 Q1 的 1/3, "
              "乐观上限 ~10-12% 净到不了 0.2 门, 不迭代; 归档为组合层低相关分散项素材(ρ0.34+干净空腿)。",
}


TYPE = "regular"

MIG_FAST_DAYS = 20
MIG_SLOW_DAYS = 120
MINUTES_PER_DAY = 1440
WARMUP_BARS = (MIG_SLOW_DAYS + MIG_FAST_DAYS + 10) * MINUTES_PER_DAY
CHUNK_SIZE = int(os.getenv("FACTOR_CHUNK_SIZE", "8"))
EPS = 1e-12

META = {
    "factor_name": "crypto_alpha_venue_share_shift_20_120",
    "author": "wesleywu",
    "level": "minutes",
    "tag": "venue_share_shift",
    "category": "sentiment",
}

SETTING = {
    "data_needed": ["dollar_volume", "xbinance_quote_volume", "close"],
    "universe": "tradable_mask",
    "pasteurization": True,
    "warmup_bars": WARMUP_BARS,
}


def _chunk_columns(columns: list[str], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for i in range(0, len(columns), chunk_size):
        yield columns[i:i + chunk_size]


def calc_factor(data_ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bybit_dv = data_ctx["dollar_volume"]
    binance_qv = data_ctx["xbinance_quote_volume"]
    close = data_ctx["close"]
    day = close.index.normalize()

    # 分块做分钟级日聚合（内存控制），份额在小型日频帧上算
    bb_parts, bn_parts = [], []
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        bb_parts.append(bybit_dv.loc[:, cols].clip(lower=0).groupby(day).sum())
        bn_parts.append(binance_qv.loc[:, cols].clip(lower=0).groupby(day).sum())

    bb_daily = pd.concat(bb_parts, axis=1)[close.columns]
    bn_daily = pd.concat(bn_parts, axis=1)[close.columns]
    del bb_parts, bn_parts

    # event venue_share_shift: bybit 量能份额
    share = bb_daily / (bb_daily + bn_daily + EPS)
    share = share.where((bb_daily > 0) & (bn_daily > 0))  # 双所均有成交才算份额

    # 慢速迁移 regime = 20d/120d 双窗比
    mig_fast = share.rolling(MIG_FAST_DAYS, min_periods=MIG_FAST_DAYS).mean()
    mig_slow = share.rolling(MIG_SLOW_DAYS, min_periods=MIG_SLOW_DAYS).mean()
    mig = mig_fast / (mig_slow + EPS)

    # direction premium: 空份额上移（散户场地拥挤）多份额下移（neglect），固定符号
    factor_daily = -(mig.rank(axis=1, pct=True) - 0.5)
    factor_daily = factor_daily.shift(1)  # 只用已完成交易日

    # 分块广播回分钟级
    factor = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float32)
    for cols in _chunk_columns(list(close.columns), CHUNK_SIZE):
        part = factor_daily[cols].reindex(day)
        part.index = close.index
        factor.loc[:, cols] = part.astype(np.float32)

    return factor.replace([np.inf, -np.inf], np.nan)
