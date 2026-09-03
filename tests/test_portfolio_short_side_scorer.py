"""Test short-side composite scorer."""
import pandas as pd
from portfolio.combiner.short_side_scorer import select_short_bottom_n


def test_picks_bottom_n_by_composite_rank():
    """Test basic bottom-N selection by composite rank."""
    # 2 factors, 5 instruments, 1 date
    panel = pd.DataFrame({
        "date": ["2024-01-01"] * 5,
        "instrument": ["A", "B", "C", "D", "E"],
        "f1": [5, 4, 3, 2, 1],
        "f2": [1, 2, 3, 4, 5],
    })
    directions = {"f1": +1, "f2": -1}  # +1 means higher=better; -1 means lower=better
    out = select_short_bottom_n(panel, factors=["f1", "f2"], directions=directions, n=2)
    # For +1 factors: short the lowest (bad); for -1 factors: short the highest (bad)
    # E has lowest f1 AND highest f2 -> worst composite -> shorted
    assert "E" in out["2024-01-01"]


def test_respects_allowed_mask():
    """Test that allowed_col mask is respected."""
    panel = pd.DataFrame({
        "date": ["2024-01-01"] * 3,
        "instrument": ["A", "B", "C"],
        "f1": [1, 2, 3],
        "allowed": [False, True, True],
    })
    out = select_short_bottom_n(
        panel, factors=["f1"], directions={"f1": +1}, n=2, allowed_col="allowed"
    )
    # A has worst f1 but not allowed, so picks from B,C
    assert "A" not in out["2024-01-01"]
