"""Environment -> HedgerConfig. Every threshold is overridable; defaults follow the README."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class ManualMarket(BaseModel):
    """Override for a Polymarket market the delta model can't price (macro, non-crypto, ...)."""

    underlying: str  # Binance symbol to hedge with, e.g. "BTCUSDT"
    beta_pp_per_pct: float  # YES probability points gained per +1% move of the underlying


class HedgerConfig(BaseModel):
    # ── FollowSM ──
    followsm_api_key: str | None = None
    signal_source: Literal["stream", "poll"] = "stream"  # stream = Enterprise WebSocket
    poll_interval_secs: float = Field(default=5.0, gt=0)

    # ── Polymarket inventory ──
    polymarket_wallet: str | None = None  # proxy wallet that holds the conditional tokens
    positions_refresh_secs: float = Field(default=30.0, gt=0)
    manual_markets: dict[str, ManualMarket] = Field(default_factory=dict)  # keyed by market slug

    # ── Toxicity bands (vpin_percentile; raw VPIN only while the percentile is warming up) ──
    pre_hedge_percentile: float = 0.85
    emergency_percentile: float = 0.95
    rebalance_percentile: float = 0.60
    pre_hedge_raw_vpin: float = 0.80
    emergency_raw_vpin: float = 0.90
    rebalance_raw_vpin: float = 0.60
    ob_imbalance_percentile_high: float = 0.99  # toxic 1% book = either tail of the symbol's own history
    ob_imbalance_percentile_low: float = 0.01
    ob_toxicity_threshold: float = 2.0  # ask/bid notional ratio, only while the percentile is warming up
    sweep_volume_z: float = 6.0  # liquidity sweep = volume burst ...
    sweep_move_natr: float = 2.0  # ... AND a 15m move of this many 15m-NATRs
    whale_sweeps_escalate_usd: float = 250_000.0  # smart-money notional that escalates PRE -> EMERGENCY ...
    adverse_flow_escalate: float = 0.65  # ... when this share of Polymarket book flow is against us

    # ── Hedge policy ──
    baseline_hedge_ratio: float = Field(default=0.0, ge=0, le=1)
    pre_hedge_ratio: float = Field(default=0.5, ge=0, le=1)
    emergency_hedge_ratio: float = Field(default=1.0, ge=0, le=1)
    max_inventory_usd: float = Field(default=25_000.0, gt=0)  # |unhedged delta| hard limit per underlying
    min_rebalance_usd: float = Field(default=150.0, gt=0)  # ignore smaller target changes
    passive_slice_usd: float = Field(default=2_000.0, gt=0)  # adaptive limit slicing child size
    min_cycle_secs: float = Field(default=2.0, ge=0)  # per-symbol decision throttle

    # ── Delta model ──
    max_delta_per_position_usd: float = Field(default=50_000.0, gt=0)
    no_hedge_final_secs: float = Field(default=300.0, ge=0)  # digital gamma explodes into expiry
    min_sigma_15m: float = Field(default=0.0005, gt=0)

    # ── Execution / risk ──
    live_trading: bool = False
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    binance_demo: bool = True  # Binance futures demo-trading endpoint
    max_slippage_bps: float = Field(default=15.0, gt=0)  # IOC protection vs mid
    max_spread_bps: float = Field(default=25.0, gt=0)  # refuse to trade a wider perp book
    max_consecutive_failures: int = Field(default=3, ge=1)
    journal_path: str = "hedger_journal.jsonl"

    @model_validator(mode="after")
    def _check(self) -> HedgerConfig:
        if not self.rebalance_percentile < self.pre_hedge_percentile < self.emergency_percentile:
            raise ValueError("need rebalance_percentile < pre_hedge_percentile < emergency_percentile")
        if not self.rebalance_raw_vpin <= self.pre_hedge_raw_vpin < self.emergency_raw_vpin:
            raise ValueError("need rebalance_raw_vpin <= pre_hedge_raw_vpin < emergency_raw_vpin")
        if not 0 <= self.ob_imbalance_percentile_low < self.ob_imbalance_percentile_high <= 1:
            raise ValueError("need 0 <= ob_imbalance_percentile_low < ob_imbalance_percentile_high <= 1")
        if not self.baseline_hedge_ratio <= self.pre_hedge_ratio <= self.emergency_hedge_ratio:
            raise ValueError("need baseline <= pre <= emergency hedge ratio")
        if self.live_trading and not (self.binance_api_key and self.binance_api_secret):
            raise ValueError("LIVE_TRADING=true requires BINANCE_API_KEY and BINANCE_API_SECRET")
        return self


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes")


def load_config() -> HedgerConfig:
    overrides: dict[str, object] = {}
    for field in HedgerConfig.model_fields:
        raw = _env(field.upper())
        if raw is not None and field not in ("live_trading", "binance_demo", "manual_markets"):
            overrides[field] = raw
    manual_path = _env("MANUAL_MARKETS_FILE")
    if manual_path:
        overrides["manual_markets"] = json.loads(Path(manual_path).read_text())
    overrides["live_trading"] = _flag("LIVE_TRADING", False)
    overrides["binance_demo"] = _flag("BINANCE_DEMO", True)
    return HedgerConfig.model_validate(overrides)
