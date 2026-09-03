"""Portfolio-layer constraint constants."""
from __future__ import annotations

STABLECOIN_BLACKLIST: frozenset[str] = frozenset({
    "USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "TUSDUSDT",
    "DAIUSDT", "USDPUSDT", "USDTUSDT",
})

WRAPPED_BLACKLIST: frozenset[str] = frozenset({
    "WBTCUSDT", "WETHUSDT", "STETHUSDT",
})

DEFAULT_BLACKLIST: frozenset[str] = STABLECOIN_BLACKLIST | WRAPPED_BLACKLIST
