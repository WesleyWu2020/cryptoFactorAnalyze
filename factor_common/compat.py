"""Legacy ``factor_miner`` adapter over the common factor framework.

``LegacyFactorMiner`` preserves the constructor argument names/order of the
old ``factor_analyse.factor_analyse_custom.factor_miner`` class and its
representative public return shapes (``IC()`` 9-tuple, ``performance()``
table columns), while delegating all analytics to ``FactorManager`` results
over the H5 store. The full old-to-new method/return mapping lives in
``.superpowers/sdd/2026-09-05-factor-common-h5-implementation/task-13-compat-map.md``.

Contract notes:

- The caller's ``factor_data`` DataFrame is copied and never mutated; the
  ``factor_name`` column is renamed to the standard ``factor`` internally.
- ``api_key``/``api_secret`` are accepted for signature compatibility and
  ignored — the Binance client is removed from the evaluation path.
- A provided legacy ``future_ret`` column is treated strictly as an EXTERNAL,
  evaluation-only label feeding the IC family (IC/RankIC, decay, autocorr).
  It is never a source of claimed funding-adjusted PnL: every return,
  turnover, and drawdown number comes from the common backtest ledger over
  H5 next-open execution prices and real funding events.
- Missing metrics surface as ``None`` (never NaN/0-fill artifacts), matching
  the ``metrics`` module conventions.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .grouping import assign_groups
from .labels import make_labels
from .manager import FactorManager
from .metrics import (
    _daily_ic,
    _half_life,
    _ic_decay,
    _ic_summary,
    _rank_ic_autocorr,
    summarize_returns,
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PERIODS_PER_YEAR = 365

_SAMPLE_ALIASES = {
    "样本内": "in_sample",
    "in_sample": "in_sample",
    "样本外": "out_of_sample",
    "out_of_sample": "out_of_sample",
}

_PERFORMANCE_COLUMNS = [
    "group",
    "return",
    "turnover",
    "annual_return",
    "sharp",
    "IC",
    "volatility",
    "annual_volatility",
    "max_drawback",
    "md_period_days",
    "recovery_period_days",
    "win_percent",
    "profit-loss ratio",
]

_STATS_KEYS = ("annualized_return", "sharpe_ratio", "max_drawdown", "win_rate")


def _round4(value):
    if value is None:
        return None
    return round(float(value), 4)


def _reject_custom_data(data):
    if data is not None:
        raise ValueError(
            "custom data slices are no longer accepted by the common adapter; "
            "use sample_type to select 全样本/样本内/样本外"
        )


class LegacyFactorMiner:
    """Drop-in adapter keeping the legacy ``factor_miner`` public contract.

    Constructor signature matches the legacy class exactly; framework context
    (H5 store, artifact/report directories, profile options) is keyword-only
    and optional, defaulting to the project-root H5 and artifact locations.
    """

    def __init__(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        factor_direction: int,
        render_path: str = None,
        api_key: str = None,
        api_secret: str = None,
        n_groups: int = 5,
        rebalance_period: int = 1,
        out_of_sample_days: int = 180,
        *,
        h5_path=None,
        base_dir=None,
        reports_dir=None,
        profile_id: str = "perp_1d",
        fee_rate: float = 0.0003,
        funding_price_mode: str = "strict",
        include_funding: bool = True,
        start=None,
        end=None,
    ):
        self.factor_name = factor_name
        self.factor_direction = factor_direction
        self.render_path = render_path
        self.api_key = api_key
        self.api_secret = api_secret
        self.n_groups = n_groups
        self.rebalance_period = rebalance_period
        self.out_of_sample_days = out_of_sample_days
        # The Binance client is obsolete; retained as a None attribute only.
        self.client = None

        self._profile_id = profile_id
        self._fee_rate = fee_rate
        self._funding_price_mode = funding_price_mode
        self._include_funding = include_funding
        self._start = start
        self._end = end

        self.factor_data = self._standardize(factor_data)
        self._manager = FactorManager(
            h5_path=h5_path, base_dir=base_dir, reports_dir=reports_dir
        )
        self._results: dict[float, dict] = {}
        self._data_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # input standardization (never mutates the caller's frame)
    # ------------------------------------------------------------------

    def _standardize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(factor_data, pd.DataFrame):
            raise TypeError("factor_data must be a pandas DataFrame")
        missing = {"date", "instrument"} - set(factor_data.columns)
        if missing:
            raise ValueError(f"factor_data is missing required columns: {sorted(missing)}")
        data = factor_data.copy(deep=True)
        if self.factor_name != "factor" and self.factor_name in data.columns:
            data = data.rename(columns={self.factor_name: "factor"})
        elif "factor" not in data.columns:
            raise ValueError(
                f"factor_data has no factor column: expected {self.factor_name!r} "
                "or the standard 'factor'"
            )
        keep = ["date", "instrument", "factor"]
        if "future_ret" in data.columns:
            keep.append("future_ret")
        data = data.loc[:, keep]
        data["date"] = pd.to_datetime(data["date"])
        data["instrument"] = data["instrument"].astype(str)
        data = data.sort_values("date").reset_index(drop=True)
        data = data.drop_duplicates(["date", "instrument"], keep="first")
        return data

    def _factor_id(self) -> str:
        if _IDENTIFIER_RE.fullmatch(str(self.factor_name)):
            return str(self.factor_name)
        slug = re.sub(r"[^A-Za-z0-9_]", "_", str(self.factor_name))
        if not slug or not slug[0].isalpha():
            slug = f"factor_{slug}"
        return slug

    # ------------------------------------------------------------------
    # evaluation plumbing
    # ------------------------------------------------------------------

    def _evaluation(self, fee_rate=None) -> dict:
        fee = self._fee_rate if fee_rate is None else float(fee_rate)
        if fee in self._results:
            return self._results[fee]
        params = {
            "rebalance_days": self.rebalance_period,
            "n_groups": self.n_groups,
            "factor_direction": self.factor_direction,
            "out_of_sample_days": self.out_of_sample_days,
            "fee_rate": fee,
            "funding_price_mode": self._funding_price_mode,
            "include_funding": self._include_funding,
        }
        if self._start is not None:
            params["start"] = self._start
        if self._end is not None:
            params["end"] = self._end
        source = self.factor_data.loc[:, ["date", "instrument", "factor"]]
        result = self._manager.evaluate(
            source,
            factor_name=self._factor_id(),
            profile_id=self._profile_id,
            params=params,
            plot=False,
        )
        self._results[fee] = result
        return result

    def _performance_block(self, result: dict) -> dict:
        return result["factor_performance"]

    def _split_date(self, result: dict) -> pd.Timestamp:
        return pd.Timestamp(self._performance_block(result)["split"]["split_date"])

    @staticmethod
    def _sample_key(sample_type) -> str:
        return _SAMPLE_ALIASES.get(str(sample_type), "full")

    def _slice_sample(self, frame: pd.DataFrame, result: dict, sample: str):
        if sample == "full":
            return frame
        split = self._split_date(result)
        if sample == "in_sample":
            return frame[frame.index <= split]
        return frame[frame.index > split]

    def _signal_sample_mask(self, dates: pd.DatetimeIndex, result: dict, sample: str):
        """``evaluate_metrics`` purge semantics for signal-dated slices.

        Entry executes at the ``t+1`` open, exit at the
        ``t+1+rebalance_period`` open. In-sample keeps signals whose exit is
        on or before the split; out-of-sample admits signals whose entry is
        strictly after the split; a signal whose holding crosses the split is
        purged from both slices (kept only in the full sample).
        """
        if sample == "full":
            return pd.Series(True, index=dates)
        split = self._split_date(result)
        entry = dates + pd.Timedelta(days=1)
        exit_ = dates + pd.Timedelta(days=1 + self.rebalance_period)
        if sample == "in_sample":
            return pd.Series(exit_ <= split, index=dates)
        return pd.Series(entry > split, index=dates)

    def _slice_signal_sample(self, frame: pd.DataFrame, result: dict, sample: str):
        if sample == "full":
            return frame
        mask = self._signal_sample_mask(frame.index, result, sample)
        return frame[mask.to_numpy()]

    def _external_labels(self, values: pd.DataFrame):
        """Provided legacy future_ret as an external evaluation-only label matrix."""
        if "future_ret" not in self.factor_data.columns:
            return None
        table = self.factor_data.loc[:, ["date", "instrument", "future_ret"]]
        labels = table.pivot(index="date", columns="instrument", values="future_ret")
        labels = labels.reindex(index=values.index, columns=values.columns)
        return labels.astype("float64")

    def _h5_labels(self, values: pd.DataFrame) -> pd.DataFrame:
        """Next-open execution labels from the H5 store (evaluation-only)."""
        start = values.index[0]
        tail_end = values.index[-1] + pd.Timedelta(days=1 + self.rebalance_period)
        opens = self._manager.dp.get_single_data("open", start=start, end=tail_end)
        opens = opens.reindex(columns=values.columns)
        return make_labels(opens, self.rebalance_period).reindex(values.index)

    def _ic_block(self, result: dict, sample: str) -> dict:
        values = result["factor_value"]
        labels = self._external_labels(values)
        if labels is None:
            return self._performance_block(result)["samples"][sample]["ic"]
        values = self._slice_signal_sample(values, result, sample)
        labels = self._slice_signal_sample(labels, result, sample)
        daily = _daily_ic(values, labels)
        return _ic_summary(daily, periods_per_year=_PERIODS_PER_YEAR)

    @staticmethod
    def _daily_frame(ic_block: dict) -> pd.DataFrame:
        daily = pd.DataFrame(ic_block["daily"], columns=["date", "ic", "rank_ic"])
        if daily.empty:
            daily = pd.DataFrame(columns=["date", "ic", "rank_ic"])
        else:
            daily["date"] = pd.to_datetime(daily["date"])
        daily["acc_ic"] = daily["ic"].cumsum()
        return daily

    def _monotonicity(self, result: dict, sample: str):
        group_returns = result.get("group_returns")
        if not isinstance(group_returns, pd.DataFrame) or group_returns.empty:
            return None
        sliced = self._slice_sample(group_returns, result, sample)
        means = sliced.mean()
        means = means.dropna()
        if len(means) < 2:
            return None
        corr, _ = stats.spearmanr(np.arange(1, len(means) + 1), means.to_numpy())
        return float(corr) if np.isfinite(corr) else None

    # ------------------------------------------------------------------
    # IC family (legacy tuple order preserved)
    # ------------------------------------------------------------------

    def _ic_tuple(self, sample: str) -> tuple:
        result = self._evaluation()
        ic_block = self._ic_block(result, sample)
        daily = self._daily_frame(ic_block)
        acc_ic = daily.loc[:, ["date", "acc_ic"]]
        gross = self._performance_block(result)["scenarios"]["gross"].get(sample)
        ir = gross.get("sharpe") if isinstance(gross, dict) else None
        return (
            ic_block["ic_mean"],
            acc_ic,
            ir,
            ic_block["icir"],
            ic_block["t_stat"],
            ic_block["p_value"],
            self._monotonicity(result, sample),
            ic_block["rank_ic_mean"],
            daily,
        )

    def IC(self):
        """Legacy 9-tuple: (ic, acc_ic, ir, ic_ir, t_stat, p_value, spearman_corr, rank_ic, daily_ic)."""
        return self._ic_tuple("full")

    def IC_with_sample_split(self, data=None, sample_type="全样本"):
        _reject_custom_data(data)
        return self._ic_tuple(self._sample_key(sample_type))

    # ------------------------------------------------------------------
    # performance tables (legacy column layout preserved)
    # ------------------------------------------------------------------

    def _group_label(self, group_num: int) -> str:
        if self.factor_direction == 1:
            if group_num == self.n_groups - 1:
                return "long"
            if group_num == 0:
                return "short"
        else:
            if group_num == 0:
                return "long"
            if group_num == self.n_groups - 1:
                return "short"
        return str(group_num)

    def _group_ic(self, result: dict, sample: str, group_num: int, values, groups):
        labels = self._external_labels(values)
        if labels is None:
            labels = self._h5_labels(values)
        member = groups == (group_num + 1)
        stacked_values = values.where(member).stack()
        stacked_labels = labels.where(member).stack()
        pairs = pd.concat([stacked_values, stacked_labels], axis=1).dropna()
        if len(pairs) < 2 or pairs.iloc[:, 0].nunique() < 2 or pairs.iloc[:, 1].nunique() < 2:
            return None
        return float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))

    def _performance_table(self, group_num: int, sample: str) -> pd.DataFrame:
        if not 0 <= group_num < self.n_groups:
            raise ValueError(f"group_num must be in [0, {self.n_groups - 1}]")
        result = self._evaluation()
        group_returns = result["group_returns"]
        series = self._slice_sample(
            group_returns[f"group_{group_num + 1}"], result, sample
        ).dropna()
        summary = summarize_returns(series, periods_per_year=_PERIODS_PER_YEAR)

        values = self._slice_sample(result["factor_value"], result, sample)
        groups, _ = assign_groups(values, self.n_groups)
        ic_value = self._group_ic(result, sample, group_num, values, groups)

        row = {
            "group": self._group_label(group_num),
            "return": _round4(summary["total_return"]),
            # Per-group membership-churn turnover is not retained; the common
            # turnover is portfolio-level traded notional.
            "turnover": None,
            "annual_return": _round4(summary["annual_return"]),
            "sharp": _round4(summary["sharpe"]),
            "IC": _round4(ic_value),
            "volatility": _round4(summary["volatility"]),
            "annual_volatility": _round4(summary["annual_volatility"]),
            # New convention: fraction, not the legacy rounded percentage.
            "max_drawback": _round4(summary["max_drawdown"]),
            "md_period_days": "-",
            "recovery_period_days": "-",
            "win_percent": _round4(summary["win_rate"]),
            "profit-loss ratio": _round4(summary["profit_loss_ratio"]),
        }
        return pd.DataFrame([row], columns=_PERFORMANCE_COLUMNS)

    def performance(self, group_num):
        return self._performance_table(group_num, "full")

    def performance_with_sample_split(self, group_num, data=None, sample_type="全样本"):
        _reject_custom_data(data)
        return self._performance_table(group_num, self._sample_key(sample_type))

    # ------------------------------------------------------------------
    # hedged returns with fees (fee_rate keyword preserved)
    # ------------------------------------------------------------------

    @staticmethod
    def _ledger_stats(ledger: pd.DataFrame) -> dict:
        if ledger is None or ledger.empty:
            return {}
        summary = summarize_returns(ledger["return"], periods_per_year=_PERIODS_PER_YEAR)
        return {
            "annualized_return": summary["annual_return"],
            "sharpe_ratio": summary["sharpe"],
            "max_drawdown": summary["max_drawdown"],
            "win_rate": summary["win_rate"],
        }

    def _hedged_returns(self, fee_rate: float, sample: str):
        result = self._evaluation(fee_rate=fee_rate)
        scenarios = result["factor_result"]["scenarios"]
        gross = scenarios["gross"].get("ledger")
        net = scenarios["trading_net"].get("ledger")
        if gross is None or gross.empty or net is None or net.empty:
            return pd.DataFrame({"date": [], "hedged_return": []}), {}, {}

        # Legacy column layout preserved; legs and per-day constant turnover
        # have no common-ledger source, so they degrade to explicit None
        # placeholders (matching the performance() convention).
        frame = pd.DataFrame(index=gross.index)
        frame["long_return"] = None
        frame["short_return"] = None
        frame["adjusted_hedged_return_no_fee"] = gross["return"]
        frame["adjusted_turnover"] = None
        frame["adjusted_hedged_return_with_fee"] = net["return"].reindex(gross.index)
        frame["adjusted_fee"] = (
            frame["adjusted_hedged_return_no_fee"] - frame["adjusted_hedged_return_with_fee"]
        )
        frame["cum_return_no_fee"] = gross["equity"] - 1.0
        frame["cum_return_with_fee"] = net["equity"].reindex(gross.index) - 1.0
        frame = frame.loc[:, [
            "long_return",
            "short_return",
            "adjusted_hedged_return_no_fee",
            "adjusted_turnover",
            "adjusted_fee",
            "adjusted_hedged_return_with_fee",
            "cum_return_no_fee",
            "cum_return_with_fee",
        ]]

        frame = self._slice_sample(frame, result, sample)
        stats_no_fee = self._ledger_stats(frame.rename(
            columns={"adjusted_hedged_return_no_fee": "return"}
        ))
        stats_with_fee = self._ledger_stats(frame.rename(
            columns={"adjusted_hedged_return_with_fee": "return"}
        ))
        return frame, stats_no_fee, stats_with_fee

    def calculate_hedged_returns_with_fees(self, fee_rate=0.0003):
        return self._hedged_returns(float(fee_rate), "full")

    def calculate_hedged_returns_with_fees_sample_split(
        self, data=None, fee_rate=0.0003, sample_type="全样本"
    ):
        _reject_custom_data(data)
        return self._hedged_returns(float(fee_rate), self._sample_key(sample_type))

    # ------------------------------------------------------------------
    # RankIC decay / autocorrelation / half-life
    # ------------------------------------------------------------------

    def _decay_rows(self, result: dict, sample: str) -> list:
        values = result["factor_value"]
        labels = self._external_labels(values)
        if labels is None:
            rows = self._performance_block(result)["samples"][sample]["ic_decay"]
            return [{"lag": row["horizon"], "rank_ic": row["rank_ic"]} for row in rows]
        values = self._slice_signal_sample(values, result, sample)
        labels = self._slice_signal_sample(labels, result, sample)
        rows = _ic_decay(values, labels)
        return [{"lag": row["horizon"], "rank_ic": row["rank_ic"]} for row in rows]

    def calculate_rank_ic_decay(self, max_lag=10):
        rows = self._decay_rows(self._evaluation(), "full")
        return pd.DataFrame([row for row in rows if row["lag"] <= max_lag],
                            columns=["lag", "rank_ic"])

    def calculate_rank_ic_decay_for_sample(self, data=None, max_lag=10, sample_type="全样本"):
        _reject_custom_data(data)
        rows = self._decay_rows(self._evaluation(), self._sample_key(sample_type))
        return pd.DataFrame([row for row in rows if row["lag"] <= max_lag],
                            columns=["lag", "rank_ic"])

    def _autocorr_rows(self, result: dict, sample: str) -> list:
        values = result["factor_value"]
        labels = self._external_labels(values)
        if labels is None:
            return list(
                self._performance_block(result)["samples"][sample]["rank_ic_autocorr"]
            )
        values = self._slice_signal_sample(values, result, sample)
        labels = self._slice_signal_sample(labels, result, sample)
        daily = _daily_ic(values, labels)
        return _rank_ic_autocorr(daily, values.index)

    def calculate_rank_ic_autocorr(self, max_lag=20):
        rows = self._autocorr_rows(self._evaluation(), "full")
        return pd.DataFrame([row for row in rows if row["lag"] <= max_lag],
                            columns=["lag", "autocorr"])

    def calculate_rank_ic_autocorr_for_sample(self, data=None, max_lag=20, sample_type="全样本"):
        _reject_custom_data(data)
        rows = self._autocorr_rows(self._evaluation(), self._sample_key(sample_type))
        return pd.DataFrame([row for row in rows if row["lag"] <= max_lag],
                            columns=["lag", "autocorr"])

    def calc_rankic_halflife(self, df_or_decay_curve, threshold=0.5):
        """First horizon whose |RankIC| <= half the |horizon-1| magnitude.

        ``threshold`` is retained for signature compatibility; the common
        half-life rule is fixed at one half (see the compat map).
        """
        if isinstance(df_or_decay_curve, pd.DataFrame):
            pairs = zip(df_or_decay_curve["lag"], df_or_decay_curve["rank_ic"])
        else:
            values = np.asarray(df_or_decay_curve, dtype="float64")
            pairs = enumerate(values, start=1)
        decay = []
        for lag, rank_ic in pairs:
            value = None if rank_ic is None or not np.isfinite(rank_ic) else float(rank_ic)
            decay.append({"horizon": int(lag), "rank_ic": value})
        decay.sort(key=lambda row: row["horizon"])
        return _half_life(decay)

    # ------------------------------------------------------------------
    # latest groups
    # ------------------------------------------------------------------

    def get_latest_group_symbols(self, include_incomplete_data=True):
        result = self._evaluation()
        values = result["factor_value"]
        groups, _ = assign_groups(values, self.n_groups)
        valid_dates = values.index[values.notna().any(axis=1)]
        if len(valid_dates) == 0:
            raise ValueError("no date carries a valid factor value")
        latest_date = valid_dates[-1]
        row_values = values.loc[latest_date]
        row_groups = groups.loc[latest_date]
        group_data = {}
        for group_id in range(self.n_groups):
            members = row_values[row_groups == (group_id + 1)].dropna()
            members = members.sort_values(ascending=False)
            group_data[group_id] = [
                [instrument, round(float(value), 4)]
                for instrument, value in members.items()
            ]
        return latest_date, group_data

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def render(self):
        """Render the common HTML report; returns the renderer info dict."""
        result = self._evaluation()
        if self.render_path:
            output_path = Path(self.render_path)
        else:
            output_path = (
                self._manager.reports_dir
                / f"{self._factor_id()}_{self.rebalance_period}d_{result['run_id']}.html"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        info = self._manager.plot_result(result, output_path=output_path)
        print(f"✅ 因子分析报告已保存至: {info['output_path']}")
        return info

    def show_plot(self):
        raise NotImplementedError(
            "show_plot() returning pyecharts chart objects is no longer "
            "supported; use render() (or FactorManager.plot_result) to produce "
            "the common HTML report"
        )

    def show_plot_for_sample(self, data=None, sample_type="全样本"):
        raise NotImplementedError(
            "show_plot_for_sample() returning pyecharts chart objects is no "
            "longer supported; use render() to produce the common HTML report "
            "with full/in-sample/out-of-sample sections"
        )

    # ------------------------------------------------------------------
    # legacy attribute surface (lazily derived from the common result)
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Long table with date/instrument/factor plus 0-based group column."""
        if self._data_cache is None:
            result = self._evaluation()
            values = result["factor_value"]
            groups, _ = assign_groups(values, self.n_groups)
            stacked = values.stack().rename("factor").reset_index()
            stacked.columns = ["date", "instrument", "factor"]
            stacked_groups = groups.stack().rename("group").reset_index()
            stacked_groups.columns = ["date", "instrument", "group"]
            merged = stacked.merge(
                stacked_groups, on=["date", "instrument"], how="left"
            )
            merged["group"] = merged["group"] - 1
            if "future_ret" in self.factor_data.columns:
                merged = merged.merge(
                    self.factor_data[["date", "instrument", "future_ret"]],
                    on=["date", "instrument"],
                    how="left",
                )
            self._data_cache = merged
        return self._data_cache

    @property
    def split_date(self) -> pd.Timestamp:
        return self._split_date(self._evaluation())

    @property
    def in_sample_dates(self) -> int:
        split = self._performance_block(self._evaluation())["split"]
        return int(split["in_sample_signal_dates"])

    @property
    def out_sample_dates(self) -> int:
        split = self._performance_block(self._evaluation())["split"]
        return int(split["out_of_sample_signal_dates"])

    @property
    def data_in_sample(self) -> pd.DataFrame:
        return self.data[self.data["date"] <= self.split_date].reset_index(drop=True)

    @property
    def data_out_sample(self) -> pd.DataFrame:
        return self.data[self.data["date"] > self.split_date].reset_index(drop=True)


__all__ = ["LegacyFactorMiner"]
