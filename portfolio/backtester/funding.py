"""Perpetual funding rate cost model."""
from __future__ import annotations


def daily_funding_cost(
    hedge_ratio: float,
    funding_rate_annual: float = 0.1095,
) -> float:
    """Daily funding cost for perpetual short (paid by short when funding>0).

    cost = hedge_ratio * (funding_rate_annual / 365)

    Args:
        hedge_ratio: Short notional as fraction of NAV; must be >= 0
        funding_rate_annual: Annualized funding (e.g. 0.1095 = 10.95%/year)

    Returns:
        Daily cost as fraction of NAV.
    """
    hedge = float(hedge_ratio)
    if hedge < 0:
        raise ValueError(f"hedge_ratio must be >= 0, got {hedge}")
    return hedge * float(funding_rate_annual) / 365.0
