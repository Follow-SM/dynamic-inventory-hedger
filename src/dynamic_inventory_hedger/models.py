"""Unified schemas shared by the signal, inventory, execution and risk layers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ToxicityLevel(StrEnum):
    NORMAL = "NORMAL"
    PRE_HEDGING_ALERT = "PRE_HEDGING_ALERT"
    EMERGENCY_HEDGE_EXECUTION = "EMERGENCY_HEDGE_EXECUTION"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {
    ToxicityLevel.NORMAL: 0,
    ToxicityLevel.PRE_HEDGING_ALERT: 1,
    ToxicityLevel.EMERGENCY_HEDGE_EXECUTION: 2,
}


class ToxicityMetric(BaseModel):
    """FollowSM microstructure for one Binance symbol, flattened for the hedger."""

    symbol: str
    timestamp_ms: int
    price: float
    vpin: float
    vpin_percentile: float | None
    ob_imbalance_l1: float
    depth_imbalance: dict[str, float] = Field(default_factory=dict)  # "0.5%" / "1.0%" / "2.0%" bid share
    ob_toxicity_1pct: float
    volume_z_score: float
    natr_15m: float
    price_delta_15m_pct: float
    whale_sweeps_1h_usdt: float = 0.0
    # Polymarket YES/Up token id -> bid / (bid + ask) notional on that token's book
    order_flow_by_token: dict[str, float] = Field(default_factory=dict)


class PositionState(BaseModel):
    """Current exposure on one venue, expressed as an equivalent perp notional in USD.

    `net_delta_usd` is the USD P&L per +100% log move of the underlying (i.e. the perp
    notional with the same first-order exposure). Positive = long the underlying.
    """

    exchange: Literal["polymarket", "binance_usdm"]
    symbol: str  # Binance underlying, e.g. "BTCUSDT"
    instrument: str  # Polymarket market slug + outcome, or the perp symbol
    size: float  # shares (Polymarket) or signed contracts (perp)
    entry_price: float
    net_delta_usd: float
    token_id: str | None = None
    mark_price: float | None = None


class HedgeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    POST_ONLY_LIMIT = "POST_ONLY_LIMIT"  # passive: rest on the book, earn the spread / maker fee
    IOC_LIMIT = "IOC_LIMIT"  # aggressive: take liquidity now, capped by a protection price


class HedgeSignal(BaseModel):
    """One order the execution engine should place on the perp venue."""

    action: HedgeAction
    symbol: str  # Binance symbol, e.g. "BTCUSDT"
    notional_usd: float = Field(gt=0)
    order_type: OrderType
    level: ToxicityLevel
    reduce_only: bool
    target_hedge_usd: float
    current_hedge_usd: float
    reason: str
