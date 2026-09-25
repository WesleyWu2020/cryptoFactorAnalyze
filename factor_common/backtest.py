"""Fixed-quantity daily portfolio accounting with explicit event ordering.

``run_backtest(values, opens, events, quality, profile, *, signal_start,
signal_end)`` simulates the 50/50 long-short portfolio (the ``"long_short"``
key of ``grouping.target_weights``) on the daily UTC boundary grid and returns
a plain dict::

    {
        "status": "complete" | "incomplete",   # all scenarios certified
        "portfolio": "long_short",
        "profile_id": str,
        "scenarios": {"gross": {...}, "trading_net": {...}, "all_costs": {...}},
        "diagnostics": {...},                  # shared calendar/signal details
    }

Each scenario dict carries:

- ``"status"``: ``"complete"`` when the portfolio was fully liquidated with
  every cash flow resolved, else ``"incomplete"``.
- ``"ledger"``: DataFrame indexed by date with columns ``equity``, ``return``,
  ``price_pnl``, ``funding_cashflow``, ``fee``, ``slippage``,
  ``trade_notional``. The row for
  date ``D`` values holdings at ``D``'s open and includes orders executed at
  that open plus every funding cash flow with a timestamp in ``[D, D+1)`` that
  belongs to held quantities. The first row's ``return`` is relative to
  ``profile.initial_equity``, so the inception fee is visible in the series.
- ``"orders"``: DataFrame with columns ``date``, ``instrument``, ``side``,
  ``target_weight``, ``target_quantity``, ``order_quantity``, ``price``,
  ``notional``, ``fee``, ``slippage``, ``status`` (``filled``/``failed``),
  ``reason``. ``fee`` is commission (``notional * fee_rate``) and ``slippage``
  is the market-impact cost (``notional * slippage``); both are per-fill
  one-sided charges and zero in the ``gross`` scenario.
- ``"positions"``: date x instrument DataFrame of post-trade quantities.
- ``"valuation_prices"``: date x instrument DataFrame of the last boundary
  open price used to value each holding.
- ``"funding"``: settlement rows (``funding.SETTLEMENT_COLUMNS``); empty for
  ``gross``/``trading_net``.
- ``"funding_coverage"``: coverage rows (``funding.COVERAGE_COLUMNS``); only
  populated for ``all_costs`` with ``profile.include_funding``.
- ``"diagnostics"``: status/halt/tail details with ISO dates, failed-order and
  blocked-order counts, retained final quantities.

Scenarios are independent accountings sharing signals and schedule: ``gross``
(zero fees, no funding), ``trading_net`` (profile fees, no funding) and
``all_costs`` (fees plus funding when ``profile.include_funding``). Each sizes
orders from its own equity; no scenario's costs touch another's NAV.

Boundary processing order per UTC day ``D``:

1. Value pre-existing holdings at ``D``'s open against the last valuation
   price. A held instrument without a valid open halts certified valuation;
   the position stays in diagnostics and never disappears.
2. Settle funding events exactly at ``D 00:00 UTC`` on pre-trade quantities,
   so a first entry owns no boundary funding and the final boundary settles
   before liquidation.
3. If ``D`` is a scheduled execution date, trade toward the target computed
   from the signal row at ``D - signal_delay_days`` (the exact prior signal,
   never a forward-filled arbitrary row). Orders size against equity after
   the already-due funding of step 2; fees reduce equity after sizing.
4. Settle intraday funding events (``D 00:00 < t < (D+1) 00:00``) on the
   post-trade quantities, which stay unchanged until the next boundary.
5. If an exchange lifecycle table supplies a terminal price for ``D``, settle
   that contract at the supplied price, record an explicit
   ``contract_settlement`` order, and prevent later re-entry.
6. For ``all_costs``, check funding coverage of day ``D`` for every held
   instrument. An unresolved event or unaccepted coverage day halts certified
   continuation: known events and quantities through the first unresolved
   cash flow are retained, later exact sizes are never computed from
   assumed-zero funding, while the gross/trading-only paths may continue.

Scheduled execution dates satisfy ``(date - anchor_date).days %
rebalance_days == 0`` over the data calendar; the plot window
(``signal_start``/``signal_end``) only gates which signal rows are usable, it
never shifts the calendar. Ending liquidation occurs one holding period after
the last executed entry/rebalance; when that boundary is beyond the data the
actual evaluable tail is recorded (``missing_tail``). Failed entry prices are
explicit failed orders — rejected weights are never redistributed. Prior-bar
ineligibility (zero-volume placeholder at ``D-1``) blocks new entries only;
exits and reductions are always permitted. A placeholder flag on the trade
day itself is known only at that day's close, so it never excludes the same
day's opening trade; such fills are reported separately as retrospective
nonexecution evidence. Nonpositive equity halts accounting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .funding import (
    COVERAGE_COLUMNS,
    SETTLEMENT_COLUMNS,
    check_funding_coverage,
    settle_funding,
)
from .grouping import target_weights
from .profiles import BacktestProfile
from .value_engine import _validate_axes

PORTFOLIO = "long_short"
SCENARIOS = ("gross", "trading_net", "all_costs")
LEDGER_COLUMNS = ("equity", "return", "price_pnl", "funding_cashflow", "fee", "slippage", "trade_notional")
ORDER_COLUMNS = (
    "date",
    "instrument",
    "side",
    "target_weight",
    "target_quantity",
    "order_quantity",
    "price",
    "notional",
    "fee",
    "slippage",
    "status",
    "reason",
)

_ONE_DAY = pd.Timedelta(days=1)
_EVENT_REQUIRED = {"funding_time", "instrument", "funding_rate", "mark_price"}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_price(value) -> bool:
    return _is_number(value) and value > 0


def _iso(day) -> str:
    return pd.Timestamp(day).date().isoformat()


def _signal_day(value, name: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(f"{name} must be a daily UTC-naive midnight date")
    return day


def _as_naive_day(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


class _Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=list(LEDGER_COLUMNS), index=pd.DatetimeIndex([], name="date"))


def _empty_positions(instruments) -> pd.DataFrame:
    return pd.DataFrame(columns=list(instruments), index=pd.DatetimeIndex([], name="date"))


def _empty_scenario(instruments) -> dict:
    return {
        "status": "complete",
        "ledger": _empty_ledger(),
        "orders": pd.DataFrame(columns=list(ORDER_COLUMNS)),
        "positions": _empty_positions(instruments),
        "valuation_prices": _empty_positions(instruments),
        "funding": pd.DataFrame(columns=list(SETTLEMENT_COLUMNS)),
        "funding_coverage": pd.DataFrame(columns=list(COVERAGE_COLUMNS)),
        "diagnostics": {
            "status": "complete",
            "halt_reason": None,
            "halt_date": None,
            "halt_detail": None,
            "filled_orders": 0,
            "failed_orders": 0,
            "first_order_date": None,
            "last_order_date": None,
            "liquidation_date": None,
            "liquidation_reached": False,
            "missing_tail": False,
            "tail_evaluated_through": None,
            "final_quantities": {},
            "blocked_orders": [],
            "retrospective_nonexecution": [],
            "contract_settlements": [],
            "funding_total": 0.0,
        },
    }


def _run_scenario(ctx: _Context, *, fees: bool, funding: bool) -> dict:
    profile = ctx.profile
    fee_rate = profile.fee_rate if fees else 0.0
    slippage_rate = profile.slippage if fees else 0.0
    equity = profile.initial_equity
    prev_equity = profile.initial_equity
    quantities = {instrument: 0.0 for instrument in ctx.instruments}
    last_price: dict = {}
    ledger_rows: list[dict] = []
    order_rows: list[dict] = []
    position_rows: dict = {}
    price_rows: dict = {}
    funding_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    blocked: list[dict] = []
    retrospective: list[dict] = []
    forced_exits: list[dict] = []
    contract_settlements: list[dict] = []
    failed_orders = 0
    halt = None

    for day in ctx.loop_dates:
        open_row = ctx.opens.loc[day]
        held_pre = {inst: qty for inst, qty in quantities.items() if qty != 0.0}

        # 1. Value old holdings at this boundary's open.
        price_pnl = 0.0
        fee = 0.0
        slippage = 0.0
        trade_notional = 0.0
        for inst, qty in held_pre.items():
            price = open_row[inst]
            if not _valid_price(price):
                if ctx.strict_target:
                    halt = {"reason": "unpriceable_held_position", "date": day,
                            "detail": {"instrument": str(inst), "quantity": qty}}
                    break
                exit_price = last_price[inst]
                notional = abs(qty) * exit_price
                order_fee = notional * fee_rate
                order_slippage = notional * slippage_rate
                fee += order_fee
                slippage += order_slippage
                trade_notional += notional
                order_rows.append({
                    "date": day,
                    "instrument": inst,
                    "side": "buy" if qty < 0 else "sell",
                    "target_weight": 0.0,
                    "target_quantity": 0.0,
                    "order_quantity": -qty,
                    "price": exit_price,
                    "notional": notional,
                    "fee": order_fee,
                    "slippage": order_slippage,
                    "status": "filled",
                    "reason": "unpriceable_forced_exit",
                })
                quantities[inst] = 0.0
                forced_exits.append({
                    "date": _iso(day),
                    "instrument": str(inst),
                    "price": exit_price,
                })
                continue
            price_pnl += qty * (float(price) - last_price[inst])
            last_price[inst] = float(price)
        if halt is not None:
            break
        equity += price_pnl
        equity -= fee + slippage
        held_pre = {inst: qty for inst, qty in quantities.items() if qty != 0.0}

        # 2. Settle funding exactly at this boundary on pre-trade quantities.
        funding_cash = 0.0
        if funding and held_pre and day in ctx.boundary_events:
            rows, _ = settle_funding(
                {str(inst): qty for inst, qty in held_pre.items()},
                ctx.boundary_events[day],
                profile=profile,
                opens=ctx.opens,
            )
            funding_frames.append(rows)
            if not rows.empty and not rows["resolved"].all():
                reason = rows.loc[~rows["resolved"], "unresolved_reason"].iloc[0]
                halt = {"reason": "unresolved_funding", "date": day, "detail": str(reason)}
                break
            cash = float(rows["cashflow"].sum()) if not rows.empty else 0.0
            funding_cash += cash
            equity += cash

        if equity <= 0.0:
            halt = {"reason": "nonpositive_equity", "date": day, "detail": repr(equity)}
            break

        # 3. Execute scheduled orders against equity after already-due funding.
        if day in ctx.scheduled_set:
            signal = ctx.signal_row(day)
            equity_before_trade = equity
            for inst in ctx.instruments:
                weight = 0.0
                if signal is not None and _is_number(signal[inst]):
                    weight = float(signal[inst])
                if ctx.contract_ended(day, inst):
                    weight = 0.0
                current = quantities[inst]
                if weight == 0.0 and current == 0.0:
                    continue
                price = open_row[inst]
                if not _valid_price(price):
                    failed_orders += 1
                    order_rows.append({
                        "date": day,
                        "instrument": inst,
                        "side": "buy" if weight > 0 else "sell",
                        "target_weight": weight,
                        "target_quantity": np.nan,
                        "order_quantity": np.nan,
                        "price": np.nan,
                        "notional": 0.0,
                        "fee": 0.0,
                        "slippage": 0.0,
                        "status": "failed",
                        "reason": "invalid_price",
                    })
                    continue
                target_qty = weight * equity_before_trade / float(price)
                increase = abs(target_qty - current) > abs(current) + 1e-15
                if weight != 0.0 and increase and not ctx.eligible(day, inst):
                    failed_orders += 1
                    reason = ctx.eligibility_reason(day, inst)
                    blocked_entry = {"date": _iso(day), "instrument": str(inst)}
                    if ctx.strict_target:
                        blocked_entry["reason"] = reason
                    blocked.append(blocked_entry)
                    order_rows.append({
                        "date": day,
                        "instrument": inst,
                        "side": "buy" if weight > 0 else "sell",
                        "target_weight": weight,
                        "target_quantity": target_qty,
                        "order_quantity": np.nan,
                        "price": float(price),
                        "notional": 0.0,
                        "fee": 0.0,
                        "slippage": 0.0,
                        "status": "failed",
                        "reason": reason,
                    })
                    continue
                delta = target_qty - current
                if delta == 0.0:
                    continue
                notional = abs(delta) * float(price)
                order_fee = notional * fee_rate
                order_slippage = notional * slippage_rate
                trade_notional += notional
                fee += order_fee
                slippage += order_slippage
                order_rows.append({
                    "date": day,
                    "instrument": inst,
                    "side": "buy" if delta > 0 else "sell",
                    "target_weight": weight,
                    "target_quantity": target_qty,
                    "order_quantity": delta,
                    "price": float(price),
                    "notional": notional,
                    "fee": order_fee,
                    "slippage": order_slippage,
                    "status": "filled",
                    "reason": None,
                })
                quantities[inst] = target_qty
                last_price[inst] = float(price)
                if ctx.placeholder_same_day(day, inst):
                    retrospective.append({"date": _iso(day), "instrument": str(inst)})
            equity -= fee + slippage

        held_post = {inst: qty for inst, qty in quantities.items() if qty != 0.0}

        # 4. Settle intraday funding on the unchanged post-trade quantities.
        if funding and held_post and day in ctx.intraday_events:
            rows, _ = settle_funding(
                {str(inst): qty for inst, qty in held_post.items()},
                ctx.intraday_events[day],
                profile=profile,
                opens=ctx.opens,
            )
            funding_frames.append(rows)
            if not rows.empty and not rows["resolved"].all():
                reason = rows.loc[~rows["resolved"], "unresolved_reason"].iloc[0]
                halt = {"reason": "unresolved_funding", "date": day, "detail": str(reason)}
                break
            cash = float(rows["cashflow"].sum()) if not rows.empty else 0.0
            funding_cash += cash
            equity += cash

        # 5. Contracts with an explicit terminal price settle on their final
        # trading day.  This is an exchange lifecycle event, not an inferred
        # fill for a later missing candle.
        settlement_row = ctx.settlement_prices.loc[day]
        for inst, qty in list(quantities.items()):
            settlement_price = settlement_row[inst]
            if qty == 0.0 or not _valid_price(settlement_price):
                continue
            settlement_price = float(settlement_price)
            settlement_pnl = qty * (settlement_price - last_price[inst])
            notional = abs(qty) * settlement_price
            order_fee = notional * fee_rate
            order_slippage = notional * slippage_rate
            price_pnl += settlement_pnl
            equity += settlement_pnl - order_fee - order_slippage
            fee += order_fee
            slippage += order_slippage
            trade_notional += notional
            order_rows.append({
                "date": day,
                "instrument": inst,
                "side": "buy" if qty < 0 else "sell",
                "target_weight": 0.0,
                "target_quantity": 0.0,
                "order_quantity": -qty,
                "price": settlement_price,
                "notional": notional,
                "fee": order_fee,
                "slippage": order_slippage,
                "status": "filled",
                "reason": "contract_settlement",
            })
            quantities[inst] = 0.0
            last_price[inst] = settlement_price
            contract_settlements.append({
                "date": _iso(day),
                "instrument": str(inst),
                "price": settlement_price,
            })

        # 6. Coverage of this held day gates certified continuation.
        if funding:
            held_today = sorted(set(held_pre) | set(held_post), key=str)
            if held_today:
                intervals = [(str(inst), day, day + _ONE_DAY) for inst in held_today]
                rows, coverage_diag = check_funding_coverage(intervals, ctx.quality)
                coverage_frames.append(rows)
                if coverage_diag["status"] != "complete":
                    halt = {
                        "reason": "unresolved_funding_coverage",
                        "date": day,
                        "detail": coverage_diag["unresolved"][0],
                    }
                    break

        # 7. Record the certified boundary.
        ledger_rows.append({
            "date": day,
            "equity": equity,
            "return": equity / prev_equity - 1.0,
            "price_pnl": price_pnl,
            "funding_cashflow": funding_cash,
            "fee": fee,
            "slippage": slippage,
            "trade_notional": trade_notional,
        })
        prev_equity = equity
        position_rows[day] = dict(quantities)
        price_rows[day] = {inst: last_price.get(inst, np.nan) for inst in ctx.instruments}

    if ledger_rows:
        ledger = pd.DataFrame(ledger_rows, columns=["date", *LEDGER_COLUMNS]).set_index("date")
        ledger.index.name = "date"
    else:
        ledger = _empty_ledger()
    orders = pd.DataFrame(order_rows, columns=list(ORDER_COLUMNS))
    if position_rows:
        positions = pd.DataFrame.from_dict(position_rows, orient="index").reindex(
            columns=ctx.instruments
        )
        valuation_prices = pd.DataFrame.from_dict(price_rows, orient="index").reindex(
            columns=ctx.instruments
        )
        positions.index.name = "date"
        valuation_prices.index.name = "date"
    else:
        positions = _empty_positions(ctx.instruments)
        valuation_prices = _empty_positions(ctx.instruments)
    funding_out = (
        pd.concat(funding_frames, ignore_index=True)
        if funding_frames
        else pd.DataFrame(columns=list(SETTLEMENT_COLUMNS))
    )
    coverage_out = (
        pd.concat(coverage_frames, ignore_index=True)
        if coverage_frames
        else pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    )

    final_quantities = {
        str(inst): qty for inst, qty in quantities.items() if qty != 0.0
    }
    liquidation_reached = halt is None and not ctx.missing_tail
    if halt is None and ctx.missing_tail:
        halt = {
            "reason": "missing_tail",
            "date": ctx.liquidation,
            "detail": f"evaluable through {_iso(ctx.loop_dates[-1])}",
        }
    elif halt is None and final_quantities:
        halt = {"reason": "unliquidated_positions", "date": ctx.loop_dates[-1], "detail": None}
    status = "complete" if halt is None else "incomplete"

    filled_dates = (
        sorted(orders.loc[orders["status"] == "filled", "date"].unique())
        if not orders.empty
        else []
    )
    diagnostics = {
        "status": status,
        "halt_reason": halt["reason"] if halt else None,
        "halt_date": _iso(halt["date"]) if halt else None,
        "halt_detail": halt.get("detail") if halt else None,
        "filled_orders": int((orders["status"] == "filled").sum()) if not orders.empty else 0,
        "failed_orders": failed_orders,
        "first_order_date": _iso(filled_dates[0]) if filled_dates else None,
        "last_order_date": _iso(filled_dates[-1]) if filled_dates else None,
        "liquidation_date": _iso(ctx.liquidation),
        "liquidation_reached": bool(liquidation_reached),
        "missing_tail": bool(ctx.missing_tail),
        "tail_evaluated_through": _iso(ctx.loop_dates[-1]),
        "final_quantities": final_quantities,
        "blocked_orders": blocked,
        "retrospective_nonexecution": retrospective,
        "unpriceable_forced_exits": forced_exits,
        "contract_settlements": contract_settlements,
        "funding_total": float(funding_out["cashflow"].sum()) if not funding_out.empty else 0.0,
    }
    return {
        "status": status,
        "ledger": ledger,
        "orders": orders,
        "positions": positions,
        "valuation_prices": valuation_prices,
        "funding": funding_out,
        "funding_coverage": coverage_out,
        "diagnostics": diagnostics,
    }


def run_target_backtest(
    targets: pd.DataFrame,
    opens: pd.DataFrame,
    events: pd.DataFrame,
    quality: pd.DataFrame,
    profile: BacktestProfile,
    *,
    signal_start,
    signal_end,
    portfolio_name: str = PORTFOLIO,
    settlement_prices: pd.DataFrame | None = None,
) -> dict:
    """Backtest explicit targets and settle terminal contracts at known prices."""
    if portfolio_name != PORTFOLIO:
        raise ValueError(f"unsupported portfolio_name: {portfolio_name}")
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    _validate_axes(targets, name="targets")
    _validate_axes(opens, name="opens")
    if not targets.index.equals(opens.index) or not targets.columns.equals(opens.columns):
        raise ValueError("targets axes must exactly match opens axes")
    if settlement_prices is None:
        settlement_prices = pd.DataFrame(np.nan, index=targets.index, columns=targets.columns)
    else:
        _validate_axes(settlement_prices, name="settlement_prices")
        if not targets.index.equals(settlement_prices.index) or not targets.columns.equals(settlement_prices.columns):
            raise ValueError("settlement_prices axes must exactly match targets axes")
    if not isinstance(events, pd.DataFrame) or (_EVENT_REQUIRED - set(events.columns)):
        raise ValueError(f"events missing columns: {sorted(_EVENT_REQUIRED - set(events.columns))}")
    if not isinstance(quality, pd.DataFrame) or not isinstance(quality.index, pd.MultiIndex) or quality.index.nlevels != 2:
        raise ValueError("quality must be indexed by (date, instrument)")
    required_quality = {"has_complete_kline", "has_placeholder_kline"}
    missing_quality = required_quality - set(quality.columns)
    if missing_quality:
        raise ValueError(f"quality missing columns: {sorted(missing_quality)}")
    signal_start, signal_end = _signal_day(signal_start, "signal_start"), _signal_day(signal_end, "signal_end")
    if signal_start > signal_end:
        raise ValueError("signal_start must be on or before signal_end")
    dates, instruments = targets.index, list(targets.columns)
    base_diagnostics = {
        "anchor_date": profile.anchor_date,
        "rebalance_days": profile.rebalance_days,
        "signal_delay_days": profile.signal_delay_days,
        "signal_start": _iso(signal_start),
        "signal_end": _iso(signal_end),
    }
    signal_days = [d for d in dates if signal_start <= d <= signal_end and not targets.loc[d].isna().all()]
    if not signal_days:
        return {"status": "complete", "portfolio": portfolio_name, "profile_id": profile.profile_id,
                "scenarios": {n: _empty_scenario(instruments) for n in SCENARIOS},
                "diagnostics": {**base_diagnostics,
                                "scheduled_dates": [], "usable_execution_dates": [], "executed_dates": [],
                                "no_usable_signals": True}}
    anchor = pd.Timestamp(profile.anchor_date)
    recurring_schedule = [
        day for day in dates
        if (day - anchor).days % profile.rebalance_days == 0
    ]
    signal_day_set = set(signal_days)
    execution_days = [
        day for day in recurring_schedule
        if day - _ONE_DAY in signal_day_set
    ]
    if not execution_days:
        raise ValueError("targets contain no signal with a following execution open")
    # A target panel may include an explicit all-zero row immediately after
    # the requested signal window.  Treat its next-open execution as the
    # liquidation boundary, while ignoring any post-window non-zero targets.
    post_signal_zero_days = [
        d for d in dates
        if d > signal_end
        and d + _ONE_DAY in dates
        and targets.loc[d].notna().all()
        and (targets.loc[d] == 0.0).all()
    ]
    explicit_zero_execution_days = [d + _ONE_DAY for d in post_signal_zero_days]
    execution_days = sorted(set(execution_days) | set(explicit_zero_execution_days))
    first, last = execution_days[0], execution_days[-1]
    liquidation = (
        explicit_zero_execution_days[-1]
        if explicit_zero_execution_days
        else last + pd.Timedelta(days=profile.rebalance_days)
    )
    loop_dates = dates[(dates >= first) & (dates <= min(liquidation, dates[-1]))]
    boundary_events, intraday_events = {}, {}
    if not events.empty:
        ef = events.copy(); times = pd.to_datetime(ef["funding_time"], utc=True)
        naive = times.dt.tz_convert("UTC").dt.tz_localize(None); edays = naive.dt.normalize()
        for mask, bucket in ((naive == edays, boundary_events), (naive != edays, intraday_events)):
            for d, group in ef[mask].groupby(edays[mask]): bucket[d] = group
    qmap = {( _as_naive_day(d), str(i)): row for (d, i), row in quality.iterrows()}
    def eligibility_reason(day, inst):
        row = qmap.get((day - _ONE_DAY, str(inst)))
        if row is None: return "prior_bar_quality_unknown"
        if type(row["has_complete_kline"]) not in (bool, np.bool_) or not bool(row["has_complete_kline"]): return "prior_bar_incomplete_kline"
        if bool(row["has_placeholder_kline"]): return "prior_bar_placeholder_kline"
        return None
    def eligible(day, inst): return eligibility_reason(day, inst) is None
    def signal_row(day):
        d = day - _ONE_DAY
        if d not in targets.index:
            return None
        if d in post_signal_zero_days:
            return targets.loc[d]
        if not (signal_start <= d <= signal_end):
            return None
        return targets.loc[d]
    def placeholder_same_day(day, inst):
        row = qmap.get((day, str(inst)))
        return row is not None and bool(row["has_placeholder_kline"])
    settlement_dates = {
        inst: settlement_prices.index[settlement_prices[inst].map(_valid_price)].min()
        for inst in instruments
        if settlement_prices[inst].map(_valid_price).any()
    }
    def contract_ended(day, inst):
        terminal_day = settlement_dates.get(inst)
        return terminal_day is not None and day > terminal_day
    ctx = _Context(profile=profile, opens=opens, quality=quality, instruments=instruments,
                   loop_dates=loop_dates, scheduled_set=set(execution_days), signal_row=signal_row,
                   boundary_events=boundary_events, intraday_events=intraday_events, eligible=eligible,
                   eligibility_reason=eligibility_reason,
                   placeholder_same_day=placeholder_same_day,
                   settlement_prices=settlement_prices, contract_ended=contract_ended,
                   liquidation=liquidation, missing_tail=liquidation > dates[-1], strict_target=True)
    scenarios = {"gross": _run_scenario(ctx, fees=False, funding=False),
                 "trading_net": _run_scenario(ctx, fees=True, funding=False),
                 "all_costs": _run_scenario(ctx, fees=True, funding=profile.include_funding)}
    status = "complete" if all(s["status"] == "complete" for s in scenarios.values()) else "incomplete"
    orders = scenarios["gross"]["orders"]
    executed = [_iso(d) for d in sorted(orders.loc[orders.status == "filled", "date"].unique())] if not orders.empty else []
    return {"status": status, "portfolio": portfolio_name, "profile_id": profile.profile_id,
            "scenarios": scenarios,
            "diagnostics": {**base_diagnostics,
                            "scheduled_dates": [_iso(d) for d in execution_days],
                            "usable_execution_dates": [_iso(d) for d in execution_days],
                            "executed_dates": executed, "no_usable_signals": False}}


def run_backtest(
    values: pd.DataFrame,
    opens: pd.DataFrame,
    events: pd.DataFrame,
    quality: pd.DataFrame,
    profile: BacktestProfile,
    *,
    signal_start,
    signal_end,
) -> dict:
    """Simulate fixed-quantity daily accounting; see the module docstring."""
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    _validate_axes(values, name="values")
    _validate_axes(opens, name="opens", expected=(values.index, values.columns))
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a DataFrame of funding events")
    missing = _EVENT_REQUIRED - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    if (
        not isinstance(quality, pd.DataFrame)
        or not isinstance(quality.index, pd.MultiIndex)
        or quality.index.nlevels != 2
    ):
        raise ValueError("quality must be indexed by (date, instrument)")
    signal_start = _signal_day(signal_start, "signal_start")
    signal_end = _signal_day(signal_end, "signal_end")
    if signal_start > signal_end:
        raise ValueError("signal_start must be on or before signal_end")

    dates = values.index
    instruments = list(values.columns)
    weights = target_weights(values, profile)[PORTFOLIO]

    anchor = pd.Timestamp(profile.anchor_date)
    delay = pd.Timedelta(days=profile.signal_delay_days)
    scheduled = [day for day in dates if (day - anchor).days % profile.rebalance_days == 0]

    def signal_row(day: pd.Timestamp):
        signal_date = day - delay
        if signal_date not in weights.index:
            return None
        if not (signal_start <= signal_date <= signal_end):
            return None
        row = weights.loc[signal_date]
        if row.isna().all():
            return None
        return row

    usable = [day for day in scheduled if signal_row(day) is not None]
    base_diagnostics = {
        "anchor_date": profile.anchor_date,
        "rebalance_days": profile.rebalance_days,
        "signal_delay_days": profile.signal_delay_days,
        "signal_start": _iso(signal_start),
        "signal_end": _iso(signal_end),
    }
    if not usable:
        return {
            "status": "complete",
            "portfolio": PORTFOLIO,
            "profile_id": profile.profile_id,
            "scenarios": {name: _empty_scenario(instruments) for name in SCENARIOS},
            "diagnostics": {
                **base_diagnostics,
                "scheduled_dates": [],
                "usable_execution_dates": [],
                "executed_dates": [],
                "no_usable_signals": True,
            },
        }

    first, last = usable[0], usable[-1]
    liquidation = last + pd.Timedelta(days=profile.rebalance_days)
    missing_tail = liquidation > dates[-1]
    last_loop_day = min(liquidation, dates[-1])
    loop_dates = dates[(dates >= first) & (dates <= last_loop_day)]
    scheduled_set = set(scheduled)

    boundary_events: dict[pd.Timestamp, pd.DataFrame] = {}
    intraday_events: dict[pd.Timestamp, pd.DataFrame] = {}
    if not events.empty:
        events_frame = events.copy()
        times = pd.to_datetime(events_frame["funding_time"], utc=True)
        naive = times.dt.tz_convert("UTC").dt.tz_localize(None)
        event_days = naive.dt.normalize()
        for mask, bucket in ((naive == event_days, boundary_events),
                             (naive != event_days, intraday_events)):
            part = events_frame[mask]
            for event_day, group in part.groupby(event_days[mask]):
                bucket[event_day] = group

    placeholder: dict[tuple[pd.Timestamp, str], bool] = {}
    if "has_placeholder_kline" in quality.columns:
        for (quality_day, instrument), flag in zip(
            quality.index, quality["has_placeholder_kline"]
        ):
            placeholder[(_as_naive_day(quality_day), str(instrument))] = (
                bool(flag) if not pd.isna(flag) else False
            )

    def eligible(day: pd.Timestamp, instrument) -> bool:
        return not placeholder.get((day - _ONE_DAY, str(instrument)), False)

    def placeholder_same_day(day: pd.Timestamp, instrument) -> bool:
        return placeholder.get((day, str(instrument)), False)

    def eligibility_reason(day: pd.Timestamp, instrument) -> str | None:
        return "prior_bar_ineligible" if not eligible(day, instrument) else None

    ctx = _Context(
        profile=profile,
        opens=opens,
        quality=quality,
        instruments=instruments,
        loop_dates=loop_dates,
        scheduled_set=scheduled_set,
        signal_row=signal_row,
        boundary_events=boundary_events,
        intraday_events=intraday_events,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        placeholder_same_day=placeholder_same_day,
        settlement_prices=pd.DataFrame(np.nan, index=opens.index, columns=opens.columns),
        contract_ended=lambda day, instrument: False,
        liquidation=liquidation,
        missing_tail=missing_tail,
        strict_target=False,
    )

    scenarios = {
        "gross": _run_scenario(ctx, fees=False, funding=False),
        "trading_net": _run_scenario(ctx, fees=True, funding=False),
        "all_costs": _run_scenario(ctx, fees=True, funding=profile.include_funding),
    }
    status = (
        "complete"
        if all(scenario["status"] == "complete" for scenario in scenarios.values())
        else "incomplete"
    )
    gross_orders = scenarios["gross"]["orders"]
    executed_dates = (
        [_iso(day) for day in sorted(gross_orders.loc[gross_orders["status"] == "filled", "date"].unique())]
        if not gross_orders.empty
        else []
    )
    diagnostics = {
        **base_diagnostics,
        "scheduled_dates": [
            _iso(day) for day in scheduled if first <= day <= last_loop_day
        ],
        "usable_execution_dates": [_iso(day) for day in usable],
        "executed_dates": executed_dates,
        "no_usable_signals": False,
    }
    return {
        "status": status,
        "portfolio": PORTFOLIO,
        "profile_id": profile.profile_id,
        "scenarios": scenarios,
        "diagnostics": diagnostics,
    }


__all__ = ["LEDGER_COLUMNS", "ORDER_COLUMNS", "PORTFOLIO", "SCENARIOS", "run_backtest", "run_target_backtest"]
