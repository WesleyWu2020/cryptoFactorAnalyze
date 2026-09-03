def test_funding_daily_cost():
    from portfolio.backtester.funding import daily_funding_cost
    # hedge=0.6, 10.95%/365 = 0.03%/day
    cost = daily_funding_cost(hedge_ratio=0.6, funding_rate_annual=0.1095)
    assert abs(cost - 0.6 * 0.1095 / 365) < 1e-12


def test_funding_zero_hedge():
    from portfolio.backtester.funding import daily_funding_cost
    assert daily_funding_cost(0.0, 0.1095) == 0.0


def test_funding_negative_hedge_raises():
    from portfolio.backtester.funding import daily_funding_cost
    import pytest
    with pytest.raises(ValueError):
        daily_funding_cost(-0.1, 0.1095)
