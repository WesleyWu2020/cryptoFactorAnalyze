"""Transaction cost computation for portfolio rebalancing."""
from __future__ import annotations


def compute_fee(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    fee_rate: float = 0.001,
) -> float:
    """Compute round-trip transaction cost for a rebalance.

    fee = sum(|w_new[i] - w_old[i]|) * fee_rate / 2

    Args:
        old_weights: {instrument: weight} before rebalance
        new_weights: {instrument: weight} after rebalance
        fee_rate: one-way fee rate (e.g. 0.001 = 10bps)

    Returns:
        Total fee as fraction of portfolio value.
    """
    instruments = set(old_weights) | set(new_weights)
    turnover = sum(
        abs(new_weights.get(ins, 0.0) - old_weights.get(ins, 0.0))
        for ins in instruments
    )
    return turnover * fee_rate / 2


def compute_perp_fee(
    old_hedge: float,
    new_hedge: float,
    fee_rate: float = 0.0005,
) -> float:
    """Compute one-sided perpetual futures trading fee.

    fee = |new - old| * fee_rate

    (Single-leg turnover, not round-trip like compute_fee.)
    """
    return abs(float(new_hedge) - float(old_hedge)) * float(fee_rate)
