"""旧版 factor_miner 入口 —— common 框架适配器导出。

原 2467 行的 ``factor_miner`` 实现已由
``factor_common.compat.LegacyFactorMiner`` 取代：构造参数名称/顺序与公开
返回结构保持不变（IC() 九元组、performance() 表格列等），全部分析委托给
common 框架的 FactorManager 评估结果。新旧方法/返回值映射见
``.superpowers/sdd/2026-09-05-factor-common-h5-implementation/task-13-compat-map.md``。

注意：构造器不再就地修改传入的 DataFrame；显式提供的 ``future_ret`` 仅作
外部标签参与 IC 族指标，不再用于任何收益/资金费口径。Binance 客户端依赖
已从评估路径移除。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factor_common.compat import LegacyFactorMiner  # noqa: E402

factor_miner = LegacyFactorMiner

__all__ = ["LegacyFactorMiner", "factor_miner"]
