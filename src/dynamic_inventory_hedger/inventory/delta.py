"""Binary-market -> perp-delta mapping (pure functions, no I/O).

Every exposure is expressed as `delta_usd`: the USD P&L per +1.0 log-move of the underlying,
i.e. the perp notional with the same first-order exposure. Hedge = -delta_usd in the perp.

Crypto price markets are priced as zero-drift digital options on the Binance spot close:

    above K:  P = N(d),  d = (ln(S/K) - σ²/2) / σ          dP/dlnS =  φ(d) / σ
    below K:  P = N(-d)                                    dP/dlnS = -φ(d) / σ
    up/down:  "above" with K = the window's opening price  (zero delta before the window opens)
    reach K:  one-touch up,   P ≈ 2·N(-ln(K/S)/σ)          dP/dlnS =  2·φ(ln(K/S)/σ) / σ
    dip to K: one-touch down, P ≈ 2·N(-ln(S/K)/σ)          dP/dlnS = -2·φ(ln(S/K)/σ) / σ

where σ is the volatility over the remaining time. It is derived from FollowSM's `natr_15m`
(average 15-minute true range / price): for Brownian motion E[range] ≈ 1.596·σ, so
σ_15m ≈ natr_15m / 1.596 and σ(τ) = σ_15m · sqrt(τ / 900s).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

MarketKind = Literal["above", "below", "updown", "touch_up", "touch_down", "manual"]

RANGE_TO_SIGMA = math.sqrt(8 / math.pi)  # E[range of BM over T] / (σ√T) ≈ 1.596
SECONDS_PER_15M = 900.0

UNDERLYINGS: dict[str, tuple[str, ...]] = {
    "BTCUSDT": ("bitcoin", "btc"),
    "ETHUSDT": ("ethereum", "eth", "ether"),
    "SOLUSDT": ("solana", "sol"),
    "XRPUSDT": ("xrp", "ripple"),
    "DOGEUSDT": ("dogecoin", "doge"),
}

_PRICE = r"\$?\s?(\d[\d,]*(?:\.\d+)?)(?![\d,]|\.\d|\s?%)\s?([kKmM]?)\b"
_PATTERNS: tuple[tuple[MarketKind, re.Pattern[str]], ...] = (
    ("above", re.compile(r"\babove\s+" + _PRICE)),
    ("below", re.compile(r"\bbelow\s+" + _PRICE)),
    ("touch_down", re.compile(r"\b(?:dip|drop|fall)\s+to\s+" + _PRICE)),
    ("touch_up", re.compile(r"\b(?:reach|hit)\s+" + _PRICE)),
)


@dataclass(frozen=True)
class MarketSpec:
    slug: str
    kind: MarketKind
    underlying: str
    expiry_ts: float
    strike: float | None = None  # None for up/down until the window opens, and for manual
    window_start_ts: float | None = None  # up/down only
    beta_pp_per_pct: float | None = None  # manual only


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _parse_price(number: str, suffix: str) -> float:
    value = float(number.replace(",", ""))
    return value * {"k": 1e3, "m": 1e6}.get(suffix.lower(), 1.0)


def detect_underlying(question: str) -> str | None:
    q = question.lower()
    for symbol, names in UNDERLYINGS.items():
        if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names):
            return symbol
    return None


def classify_question(question: str) -> tuple[MarketKind, float | None] | None:
    """(kind, strike) for a crypto price question, or None if the model can't price it."""
    if re.search(r"\bup or down\b", question, re.IGNORECASE):
        return "updown", None
    for kind, pattern in _PATTERNS:
        match = pattern.search(question)
        if match:
            return kind, _parse_price(match.group(1), match.group(2))
    return None


def sigma_15m_from_natr(natr_15m: float, floor: float) -> float:
    return max(natr_15m / RANGE_TO_SIGMA, floor)


def dprob_dlogspot(spec: MarketSpec, spot: float, sigma_15m: float, now_ts: float) -> float:
    """Sensitivity of the YES / Up price to ln(spot). 0 when expired or not yet priceable."""
    if spec.kind == "manual":
        return spec.beta_pp_per_pct or 0.0  # pp per 1% == probability per 1.0 log-move
    tau = spec.expiry_ts - now_ts
    if tau <= 0 or spot <= 0:
        return 0.0
    if spec.kind == "updown":
        if spec.window_start_ts is not None and now_ts < spec.window_start_ts:
            return 0.0  # before the window opens, spot moves don't change P(Up)
        if spec.strike is None:
            return 0.0
    if spec.strike is None or spec.strike <= 0:
        return 0.0
    sigma = sigma_15m * math.sqrt(tau / SECONDS_PER_15M)
    x = math.log(spot / spec.strike)
    if spec.kind in ("above", "updown"):
        return _norm_pdf((x - sigma * sigma / 2) / sigma) / sigma
    if spec.kind == "below":
        return -_norm_pdf((x - sigma * sigma / 2) / sigma) / sigma
    if spec.kind == "touch_up":
        return 0.0 if x >= 0 else 2 * _norm_pdf(x / sigma) / sigma
    if spec.kind == "touch_down":
        return 0.0 if x <= 0 else -2 * _norm_pdf(x / sigma) / sigma
    return 0.0


def position_delta_usd(
    spec: MarketSpec,
    holds_yes: bool,
    shares: float,
    spot: float,
    sigma_15m: float,
    now_ts: float,
    *,
    no_hedge_final_secs: float,
    max_abs_delta_usd: float,
) -> float:
    """Equivalent perp notional (USD) for `shares` of the YES (or NO) outcome.

    Zeroed in the final `no_hedge_final_secs` (digital gamma explodes into expiry; a perp
    hedge there churns far more than it protects) and capped at `max_abs_delta_usd`.
    """
    if spec.kind != "manual" and spec.expiry_ts - now_ts < no_hedge_final_secs:
        return 0.0
    delta = shares * dprob_dlogspot(spec, spot, sigma_15m, now_ts)
    if not holds_yes:
        delta = -delta
    return max(-max_abs_delta_usd, min(max_abs_delta_usd, delta))
