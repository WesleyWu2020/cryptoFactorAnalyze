"""Compare Path B NAV vs BTC buy-hold on OOS."""
from __future__ import annotations
import pathlib
import pandas as pd


def metrics(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    n = len(rets)
    if n == 0 or nav.iloc[0] == 0:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "mdd": 0.0, "cum_ret": 0.0}
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (365.0 / n) - 1.0
    vol = rets.std() * (365.0 ** 0.5)
    sharpe = ann / vol if vol > 0 else float("nan")
    mdd = ((nav / nav.cummax()) - 1).min()
    return {"ann": float(ann), "vol": float(vol), "sharpe": float(sharpe),
            "mdd": float(mdd), "cum_ret": float(nav.iloc[-1] / nav.iloc[0] - 1.0)}


def main():
    base = pathlib.Path("portfolio/output/pathB")
    nav_csv = base / "pathB_nav.csv"
    pathB = pd.read_csv(nav_csv, index_col=0, parse_dates=True).iloc[:, 0]

    kl = pd.read_csv("data/kline_data/binance_daily_klines_20260417.csv")
    btc = kl[kl["symbol"].str.upper() == "BTCUSDT"].copy()
    btc["date"] = pd.to_datetime(btc["date"])
    btc = btc.sort_values("date").set_index("date")["close"]
    btc = btc.loc[pathB.index[0]:pathB.index[-1]]
    btc_nav = btc / btc.iloc[0]

    m_b = metrics(pathB); m_btc = metrics(btc_nav)
    report = f"""# Path B vs BTC — OOS Report

Period: {pathB.index[0].strftime('%Y-%m-%d')} → {pathB.index[-1].strftime('%Y-%m-%d')}

| Metric | Path B | BTC Buy-Hold |
|---|---|---|
| Cum Return | {m_b['cum_ret']:.2%} | {m_btc['cum_ret']:.2%} |
| Annualized | {m_b['ann']:.2%} | {m_btc['ann']:.2%} |
| Volatility | {m_b['vol']:.2%} | {m_btc['vol']:.2%} |
| Sharpe     | {m_b['sharpe']:.2f} | {m_btc['sharpe']:.2f} |
| MaxDD      | {m_b['mdd']:.2%} | {m_btc['mdd']:.2%} |

**Target check:** MaxDD {'✅' if m_b['mdd']>-0.20 else '❌'} (<20%). Ann vs BTC {'✅' if m_b['ann']>m_btc['ann'] else '❌'}.
"""
    out = base / "report.md"
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
