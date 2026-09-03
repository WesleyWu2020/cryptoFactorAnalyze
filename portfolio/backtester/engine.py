"""Drift-adjusted portfolio backtester."""
from __future__ import annotations

import pandas as pd
import numpy as np

from portfolio.backtester.fees import compute_fee


def _compute_daily_returns(kline: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns per instrument from kline close prices.

    Returns DataFrame with [date, instrument, ret] columns.
    """
    kline = kline.copy()
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    kline = kline.sort_values(["symbol", "date"])
    kline["ret"] = kline.groupby("symbol")["close"].pct_change()
    return kline.rename(columns={"symbol": "instrument"})[["date", "instrument", "ret"]]


def run_backtest(
    kline: pd.DataFrame,
    weights: dict[pd.Timestamp, dict[str, float]],
    fee_rate: float = 0.001,
) -> pd.Series:
    """Run drift-adjusted backtest supporting intra-rebalance hold days.

    Args:
        kline: DataFrame with [symbol, date, close]
        weights: dict mapping rebalance date t → target weights.
                 Dates NOT in weights are hold days (no rebalancing, no fee).
        fee_rate: one-way transaction fee rate

    Returns:
        pd.Series of daily portfolio returns indexed by date (after fees).
        Covers all kline trading days from the first rebalance date onward.
    """
    daily_ret = _compute_daily_returns(kline)
    # Pivot to wide: index=date, columns=instrument
    ret_wide = daily_ret.pivot_table(index="date", columns="instrument", values="ret")
    ret_wide = ret_wide.sort_index()

    rebalance_dates = set(weights.keys())
    if not rebalance_dates:
        return pd.Series(dtype=float)

    first_reb = min(rebalance_dates)

    port_returns: dict[pd.Timestamp, float] = {}
    current_weights: dict[str, float] = {}

    for t in ret_wide.index:
        if t < first_reb:
            continue

        is_rebalance = t in rebalance_dates

        # Initialize on first rebalance date
        if not current_weights:
            if is_rebalance:
                target = weights[t]
                fee = compute_fee({}, target, fee_rate)
                port_returns[t] = -fee
                current_weights = dict(target)
            continue

        rets_t = ret_wide.loc[t]

        # Portfolio return before any rebalancing (drift)
        port_ret = sum(
            current_weights.get(ins, 0.0) * float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)
            for ins in current_weights
        )

        # Drift current weights with today's returns
        denom = 1 + port_ret if abs(1 + port_ret) > 1e-12 else 1e-12
        drifted = {
            ins: w * (1 + float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)) / denom
            for ins, w in current_weights.items()
        }

        if is_rebalance:
            # Rebalance day: pay fee to transition from drifted to target
            target = weights[t]
            fee = compute_fee(drifted, target, fee_rate)
            port_returns[t] = port_ret - fee
            current_weights = dict(target)
        else:
            # Hold day: earn return, let weights drift, no rebalancing
            port_returns[t] = port_ret
            current_weights = drifted

    return pd.Series(port_returns).sort_index()


from portfolio.backtester.fees import compute_perp_fee
from portfolio.backtester.funding import daily_funding_cost


def run_hedged_backtest(
    kline: pd.DataFrame,
    weights: dict,
    alt_exposure: "pd.Series",
    hedge_ratio: "pd.Series",
    alt_fee: float = 0.001,
    perp_fee: float = 0.0005,
    funding_annual: float = 0.1095,
    btc_symbol: str = "BTCUSDT",
) -> "pd.Series":
    """Run backtest with Alt long + BTC perp short hedge.

    daily_ret_t = alt_exposure_t * port_ret_alt_t
                - hedge_ratio_t * btc_ret_t
                - alt_fee_t (on Alt rebalance days)
                - perp_fee_t (on hedge-change days)
                - funding_cost_t (every day)

    Args:
        kline: [symbol, date, close]
        weights: {date: {instrument: weight}} for Alt basket (unit-sum).
                 Only dates in weights are Alt rebalance days.
        alt_exposure: pd.Series[date] -> [0, 1.0] Alt long notional fraction.
        hedge_ratio: pd.Series[date] -> [0, 1.2*alt_exp] BTC short notional fraction.
        alt_fee: one-way spot fee (default 10bps)
        perp_fee: one-way perp fee (default 5bps)
        funding_annual: annualized funding (default 10.95%)
    """
    daily = _compute_daily_returns(kline)
    ret_wide = daily.pivot_table(index="date", columns="instrument", values="ret").sort_index()
    if btc_symbol not in ret_wide.columns:
        raise ValueError(f"Missing {btc_symbol} in kline")
    btc_ret = ret_wide[btc_symbol]

    if not weights:
        return pd.Series(dtype=float)

    rebalance_dates = set(weights.keys())
    first_reb = min(rebalance_dates)

    # Align exposure/hedge series to ret_wide dates
    alt_exp = alt_exposure.reindex(ret_wide.index).ffill().fillna(0.0)
    hedge = hedge_ratio.reindex(ret_wide.index).ffill().fillna(0.0)

    port_returns = {}
    current_weights = {}
    prev_alt_exp = 0.0
    prev_hedge = 0.0
    initialized = False

    for t in ret_wide.index:
        if t < first_reb:
            continue
        
        rets_t = ret_wide.loc[t]
        
        # Initialize on first date >= first_reb
        if not initialized:
            # Use the latest rebalance date <= t
            reb_dates_le_t = [d for d in rebalance_dates if d <= t]
            if not reb_dates_le_t:
                continue  # Not yet at a rebalance date
            reb_date = max(reb_dates_le_t)
            target = weights[reb_date]
            
            alt_fee_t = sum(abs(target.get(k, 0.0)) for k in target) * alt_exp.loc[t] * alt_fee / 2
            perp_fee_t = compute_perp_fee(prev_hedge, hedge.loc[t], perp_fee)
            funding_t = daily_funding_cost(hedge.loc[t], funding_annual)
            port_returns[t] = -alt_fee_t - perp_fee_t - funding_t
            current_weights = dict(target)
            prev_alt_exp = float(alt_exp.loc[t])
            prev_hedge = float(hedge.loc[t])
            initialized = True
            continue

        # Alt internal return
        alt_port_ret = sum(
            current_weights.get(ins, 0.0) * float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)
            for ins in current_weights
        )

        # Drift weights
        denom = 1 + alt_port_ret if abs(1 + alt_port_ret) > 1e-12 else 1e-12
        drifted = {
            ins: w * (1 + float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)) / denom
            for ins, w in current_weights.items()
        }

        btc_r = float(btc_ret.loc[t]) if pd.notna(btc_ret.loc[t]) else 0.0

        # P&L before costs
        pnl_long = float(alt_exp.loc[t]) * alt_port_ret
        pnl_hedge = -float(hedge.loc[t]) * btc_r

        # Costs
        is_rebalance = t in rebalance_dates
        if is_rebalance:
            target = weights[t]
            instruments = set(drifted) | set(target)
            alt_turnover = sum(
                abs(target.get(k, 0.0) * float(alt_exp.loc[t])
                    - drifted.get(k, 0.0) * prev_alt_exp)
                for k in instruments
            )
            alt_fee_t = alt_turnover * alt_fee / 2
            current_weights = dict(target)
        else:
            alt_fee_t = 0.0
            current_weights = drifted

        perp_fee_t = compute_perp_fee(prev_hedge, float(hedge.loc[t]), perp_fee)
        funding_t = daily_funding_cost(float(hedge.loc[t]), funding_annual)

        port_returns[t] = pnl_long + pnl_hedge - alt_fee_t - perp_fee_t - funding_t
        prev_alt_exp = float(alt_exp.loc[t])
        prev_hedge = float(hedge.loc[t])

    return pd.Series(port_returns).sort_index()


def run_btc_core_short_backtest(
    prices: pd.DataFrame,
    btc_weights: pd.Series,
    shorts_by_date: dict,
    funding: pd.DataFrame,
    alt_short_exposure: float = -0.3,
    fee_rate: float = 0.001,
    btc_symbol: str = "BTCUSDT",
) -> pd.Series:
    """Run backtest with dynamic BTC long + equal-weight alt shorts strategy.

    Strategy mechanics:
    - Long BTC with time-varying weight (btc_weights)
    - Short a basket of alts with total notional = alt_short_exposure (negative)
    - Shorts rebalanced only when new list provided in shorts_by_date
    - Turnover fee charged on short basket changes
    - Funding earned on short positions (positive funding → profit for shorts)

    No lookahead: Uses prices through date d and funding through date d — no future data.

    Args:
        prices: Wide DataFrame, index=date(datetime), columns=[BTCUSDT, AUSDT, ...], values=close
        btc_weights: Series indexed by date, values in [0,1] for BTC long weight
        shorts_by_date: Dict {date_str "YYYY-MM-DD" -> list[instrument]} defining short basket changes
        funding: Long DataFrame [date (str YYYY-MM-DD), instrument, funding_daily]
        alt_short_exposure: Total NEGATIVE notional for shorts (default -0.3)
        fee_rate: One-way fee rate applied on turnover (default 0.1%)
        btc_symbol: BTC symbol in prices columns (default "BTCUSDT")

    Returns:
        NAV series indexed by prices.index, starting at 1.0
    """
    # Initialize NAV
    nav = pd.Series(index=prices.index, dtype=float)
    nav.iloc[0] = 1.0

    # Build funding lookup map: (date_str, instrument) -> funding_daily
    funding_map = {}
    if not funding.empty and "date" in funding.columns:
        for _, row in funding.iterrows():
            funding_map[(str(row["date"]), str(row["instrument"]))] = float(row["funding_daily"])

    # Track current shorts
    current_shorts = []

    for i in range(1, len(prices)):
        d = prices.index[i]
        d_prev = prices.index[i - 1]
        d_str = d.strftime("%Y-%m-%d")
        d_prev_str = d_prev.strftime("%Y-%m-%d")

        # Check for short basket rebalancing (happens at d_prev close, held through d)
        turnover_fee = 0.0
        if d_prev_str in shorts_by_date:
            new_shorts = shorts_by_date[d_prev_str]
            # Compute turnover: symmetric difference
            old_set = set(current_shorts)
            new_set = set(new_shorts)
            turnover_syms = old_set.symmetric_difference(new_set)
            if len(new_shorts) > 0:
                turnover_fee = fee_rate * len(turnover_syms) * abs(alt_short_exposure) / len(new_shorts)
            current_shorts = list(new_shorts)

        # BTC long PnL
        w_btc = float(btc_weights.get(d_prev, 0.0))
        p_btc_prev = prices.loc[d_prev, btc_symbol]
        p_btc = prices.loc[d, btc_symbol]
        if pd.notna(p_btc_prev) and pd.notna(p_btc) and abs(p_btc_prev) > 1e-12:
            r_btc = (p_btc / p_btc_prev) - 1.0
        else:
            r_btc = 0.0
        pnl_btc = w_btc * r_btc

        # Alt shorts PnL
        r_short = 0.0
        funding_pnl = 0.0
        if len(current_shorts) > 0:
            w_each = alt_short_exposure / len(current_shorts)  # negative
            for ins in current_shorts:
                if ins not in prices.columns:
                    continue
                p_prev = prices.loc[d_prev, ins]
                p_now = prices.loc[d, ins]
                if pd.notna(p_prev) and pd.notna(p_now) and abs(p_prev) > 1e-12:
                    r_ins = (p_now / p_prev) - 1.0
                    r_short += w_each * r_ins
                # Funding PnL: earn funding when positive (short position)
                fund_val = funding_map.get((d_str, ins), 0.0)
                funding_pnl += (-w_each) * fund_val

        # Total return for the day
        r_total = pnl_btc + r_short + funding_pnl - turnover_fee
        nav.iloc[i] = nav.iloc[i - 1] * (1.0 + r_total)

    return nav
