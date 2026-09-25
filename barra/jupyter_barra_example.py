"""Run cells from the repository root or factor_analyse directory."""
from pathlib import Path
import sys

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "factor_common").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_common import FactorManager
from barra.crypto_barra_exposure import BarraConfig, analyze_and_write


def run_example(alpha_path=None, *, start="2024-02-01", end="2024-03-15", h5_path=None):
    """Evaluate original and residual on identical samples and execution settings."""
    alpha_path = Path(alpha_path or ROOT / "factor_analyse/factor_mining/example_momentum.py")
    params = {"start": start, "end": end, "n_groups": 5, "rebalance_days": 1}
    fm = FactorManager(h5_path=h5_path)
    result = fm.evaluate(str(alpha_path), params=params)
    name = result["metadata"]["factor_name"]
    params["factor_direction"] = result["metadata"]["factor_direction"]
    exposures = analyze_and_write(
        result, cfg=BarraConfig(), label=name,
        h5_path=h5_path or ROOT / "data/crypto_quant.h5",
    )
    residual = exposures["alpha_barra_residual"]
    if not residual.notna().any().any():
        raise ValueError("No usable residual; inspect regression status and coverage")
    matched = result["factor_value"].reindex_like(residual).where(residual.notna())
    original_eval = fm.evaluate(matched, factor_name=f"{name}_matched", params=params)
    residual_eval = fm.evaluate(residual, factor_name=f"{name}_barra_residual", params=params)
    print(exposures["style_summary"].to_string(index=False))
    print("Residual coverage:", int(residual.notna().sum().sum()), "observations")
    return original_eval, exposures, residual_eval


# Run explicitly in a notebook; importing this file has no evaluation side effects:
# original, barra_result, residual = run_example()
# barra_result["daily_barra_regression"].tail()
# residual["paths"]["report_path"]
