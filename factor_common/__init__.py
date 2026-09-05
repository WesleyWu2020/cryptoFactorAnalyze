"""Public API for the factor_common research framework."""

from .definitions import FactorSpec
from .manager import FactorManager
from .profiles import BacktestProfile

__all__ = ["BacktestProfile", "FactorManager", "FactorSpec"]
