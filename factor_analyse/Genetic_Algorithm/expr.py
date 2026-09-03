from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

Series = pd.Series


def safe_div(a: Series, b: Series, eps: float = 1e-8) -> Series:
    return a / (b + eps)


def safe_log(x: Series, eps: float = 1e-8) -> Series:
    return np.log(np.abs(x) + eps)


def safe_sqrt(x: Series, eps: float = 1e-12) -> Series:
    return np.sqrt(np.abs(x) + eps)


class Node:
    def eval(self, df: pd.DataFrame) -> Series:
        raise NotImplementedError

    def children(self) -> Sequence["Node"]:
        return ()

    def with_child(self, idx: int, child: "Node") -> "Node":
        raise NotImplementedError

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children())

    def to_str(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Terminal(Node):
    name: str

    def eval(self, df: pd.DataFrame) -> Series:
        if self.name not in df.columns:
            raise KeyError(f"Unknown terminal: {self.name}")
        return df[self.name]

    def with_child(self, idx: int, child: Node) -> Node:
        raise IndexError("Terminal has no children")

    def to_str(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Node):
    value: float

    def eval(self, df: pd.DataFrame) -> Series:
        return pd.Series(self.value, index=df.index)

    def with_child(self, idx: int, child: Node) -> Node:
        raise IndexError("Const has no children")

    def to_str(self) -> str:
        # keep it stable for hashing
        return f"c{self.value:.6g}"


@dataclass(frozen=True)
class Unary(Node):
    op_name: str
    op: Callable[[Series], Series]
    child: Node

    def eval(self, df: pd.DataFrame) -> Series:
        return self.op(self.child.eval(df))

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return Unary(self.op_name, self.op, child)

    def to_str(self) -> str:
        return f"{self.op_name}({self.child.to_str()})"


@dataclass(frozen=True)
class Binary(Node):
    op_name: str
    op: Callable[[Series, Series], Series]
    left: Node
    right: Node

    def eval(self, df: pd.DataFrame) -> Series:
        return self.op(self.left.eval(df), self.right.eval(df))

    def children(self) -> Sequence[Node]:
        return (self.left, self.right)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx == 0:
            return Binary(self.op_name, self.op, child, self.right)
        if idx == 1:
            return Binary(self.op_name, self.op, self.left, child)
        raise IndexError

    def to_str(self) -> str:
        return f"{self.op_name}({self.left.to_str()},{self.right.to_str()})"


@dataclass(frozen=True)
class RollingUnary(Node):
    op_name: str
    window: int
    child: Node

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)
        by_sym = x.groupby(df["symbol"], sort=False)

        if self.op_name in {"mean", "std", "sum", "min", "max"}:
            r = getattr(by_sym.rolling(self.window), self.op_name)()
            return r.reset_index(level=0, drop=True)

        if self.op_name == "product":
            r = by_sym.rolling(self.window).apply(lambda a: float(np.prod(a)), raw=True)
            return r.reset_index(level=0, drop=True)

        if self.op_name == "zscore":
            m = by_sym.rolling(self.window).mean().reset_index(level=0, drop=True)
            s = by_sym.rolling(self.window).std().reset_index(level=0, drop=True)
            return safe_div(x - m, s)

        if self.op_name == "ts_rank":
            r = by_sym.rolling(self.window).apply(
                lambda a: float(pd.Series(a).rank(pct=True).iloc[-1]),
                raw=False,
            )
            return r.reset_index(level=0, drop=True)

        if self.op_name == "ts_argmax":
            r = by_sym.rolling(self.window).apply(lambda a: float(np.argmax(a) + 1), raw=True)
            return r.reset_index(level=0, drop=True)

        if self.op_name == "ts_argmin":
            r = by_sym.rolling(self.window).apply(lambda a: float(np.argmin(a) + 1), raw=True)
            return r.reset_index(level=0, drop=True)

        if self.op_name == "decay_linear":
            w = np.arange(1, self.window + 1, dtype=float)
            w = w / w.sum()
            r = by_sym.rolling(self.window).apply(lambda a: float(np.dot(a, w)), raw=True)
            return r.reset_index(level=0, drop=True)

        raise ValueError(f"Unknown rolling op: {self.op_name}")

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return RollingUnary(self.op_name, self.window, child)

    def to_str(self) -> str:
        return f"{self.op_name}[{self.window}]({self.child.to_str()})"
@dataclass(frozen=True)
class RollingBinary(Node):
    op_name: str
    window: int
    left: Node
    right: Node

    def eval(self, df: pd.DataFrame) -> Series:
        a = self.left.eval(df)
        b = self.right.eval(df)
        ga = a.groupby(df["symbol"], sort=False)

        if self.op_name == "correlation":
            r = ga.rolling(self.window).corr(b)
            return r.reset_index(level=0, drop=True)

        if self.op_name == "covariance":
            r = ga.rolling(self.window).cov(b)
            return r.reset_index(level=0, drop=True)

        raise ValueError(f"Unknown rolling binary op: {self.op_name}")

    def children(self) -> Sequence[Node]:
        return (self.left, self.right)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx == 0:
            return RollingBinary(self.op_name, self.window, child, self.right)
        if idx == 1:
            return RollingBinary(self.op_name, self.window, self.left, child)
        raise IndexError

    def to_str(self) -> str:
        return f"{self.op_name}[{self.window}]({self.left.to_str()},{self.right.to_str()})"


@dataclass(frozen=True)
class CSRank(Node):
    child: Node

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)
        return x.groupby(df["date"], sort=False).rank(pct=True)

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return CSRank(child)

    def to_str(self) -> str:
        return f"rank({self.child.to_str()})"


@dataclass(frozen=True)
class Scale(Node):
    child: Node
    a: float = 1.0

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)

        def _scale(g: pd.Series) -> pd.Series:
            denom = float(g.abs().sum()) + 1e-12
            return g * (self.a / denom)

        return x.groupby(df["date"], sort=False).transform(_scale)

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return Scale(child, self.a)

    def to_str(self) -> str:
        return f"scale({self.child.to_str()},{self.a:g})"


@dataclass(frozen=True)
class IndNeutralize(Node):
    child: Node
    group_col: str

    def eval(self, df: pd.DataFrame) -> Series:
        if self.group_col not in df.columns:
            raise KeyError(f"Unknown group column: {self.group_col}")
        x = self.child.eval(df)
        mean = x.groupby([df["date"], df[self.group_col]], sort=False).transform("mean")
        return x - mean

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return IndNeutralize(child, self.group_col)

    def to_str(self) -> str:
        return f"indneutralize({self.child.to_str()},{self.group_col})"


@dataclass(frozen=True)
class SignedPower(Node):
    child: Node
    a: float

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)
        return np.sign(x) * (np.abs(x) ** self.a)

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return SignedPower(child, self.a)

    def to_str(self) -> str:
        return f"signedpower({self.child.to_str()},{self.a:g})"


@dataclass(frozen=True)
class Delay(Node):
    lag: int
    child: Node

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)
        return x.groupby(df["symbol"], sort=False).shift(self.lag)

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return Delay(self.lag, child)

    def to_str(self) -> str:
        return f"delay[{self.lag}]({self.child.to_str()})"


@dataclass(frozen=True)
class Delta(Node):
    lag: int
    child: Node

    def eval(self, df: pd.DataFrame) -> Series:
        x = self.child.eval(df)
        return x - x.groupby(df["symbol"], sort=False).shift(self.lag)

    def children(self) -> Sequence[Node]:
        return (self.child,)

    def with_child(self, idx: int, child: Node) -> Node:
        if idx != 0:
            raise IndexError
        return Delta(self.lag, child)

    def to_str(self) -> str:
        return f"delta[{self.lag}]({self.child.to_str()})"
