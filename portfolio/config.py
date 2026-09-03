"""Portfolio pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioConfig:
    """Baseline configuration for the portfolio pipeline.

    All fields have sensible defaults; instantiate with no args for MVP run.
    """
    top_n: int = 10
    ic_window: int = 20
    label_period: int = 1
    fee_rate: float = 0.001
    winsorize_pct: tuple[float, float] = field(default_factory=lambda: (0.01, 0.99))
    orthogonalize: bool = True
    rebalance_period: int = 1  # rebalance every N trading days (1=daily, 5=weekly)
    cluster_threshold: float = 0.0  # merge factors with |corr| > threshold (0=off, 0.6=recommended)
    universe_csv: str = ""
    kline_csv: str = ""

    # IS/OOS factor selection
    is_end_date: str = "2023-12-31"
    min_ic_ir: float = 0.05

    # Layer B: BTC Trend
    ema_short: int = 50
    ema_long: int = 200
    zscore_window: int = 120
    ema_ratio_saturation: float = 0.15
    zscore_saturation: float = 2.0
    t_smoothing_span: int = 5

    # Layer C: Exposure
    alt_max_exposure: float = 1.0
    alt_min_exposure: float = 0.5
    hedge_cap_multiplier: float = 1.2
    beta_window: int = 60
    beta_prior: float = 1.3
    alt_exposure_ema_alpha: float = 0.05

    # Layer D: Costs
    alt_fee_rate: float = 0.001
    perp_fee_rate: float = 0.0005
    funding_rate_annual: float = 0.1095

    # Path B: BTC Core + Alt Short
    btc_base_min: float = 0.4
    btc_base_max: float = 0.7
    alt_short_exposure_max: float = -0.3
    universe_rank_min: int = 30
    universe_rank_max: int = 100
    funding_filter_percentile: float = 0.8
    squeeze_stop_return: float = 0.25
    squeeze_stop_window: int = 2
    short_top_n: int = 5

    output_dir: str = "portfolio/output"

    def __post_init__(self) -> None:
        if self.ema_short >= self.ema_long:
            raise ValueError(f"ema_short ({self.ema_short}) must be < ema_long ({self.ema_long})")
        if self.alt_min_exposure > self.alt_max_exposure:
            raise ValueError(
                f"alt_min_exposure ({self.alt_min_exposure}) must be <= alt_max_exposure ({self.alt_max_exposure})"
            )
        if self.ema_ratio_saturation <= 0:
            raise ValueError(f"ema_ratio_saturation must be > 0, got {self.ema_ratio_saturation}")
        if self.zscore_saturation <= 0:
            raise ValueError(f"zscore_saturation must be > 0, got {self.zscore_saturation}")
        if not (0 < self.alt_exposure_ema_alpha <= 1):
            raise ValueError(f"alt_exposure_ema_alpha must be in (0, 1], got {self.alt_exposure_ema_alpha}")
        if self.beta_window <= 0:
            raise ValueError(f"beta_window must be > 0, got {self.beta_window}")
        if self.zscore_window <= 0:
            raise ValueError(f"zscore_window must be > 0, got {self.zscore_window}")
        
        # Path B validation
        if not (0.0 <= self.btc_base_min <= self.btc_base_max <= 1.0):
            raise ValueError("btc_base_min/max out of range")
        if self.universe_rank_min >= self.universe_rank_max:
            raise ValueError("universe_rank_min must be < universe_rank_max")
        if self.alt_short_exposure_max >= 0:
            raise ValueError("alt_short_exposure_max must be negative")
        if not (0.0 < self.funding_filter_percentile < 1.0):
            raise ValueError("funding_filter_percentile must be in (0,1)")

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
