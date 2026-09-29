"""Polymarket inventory: public positions -> MarketSpec -> per-underlying perp-equivalent delta."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.inventory.delta import (
    MarketSpec,
    classify_question,
    detect_underlying,
    position_delta_usd,
    sigma_15m_from_natr,
)
from dynamic_inventory_hedger.models import PositionState, ToxicityMetric

log = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
BINANCE_SPOT_API = "https://data-api.binance.vision"


@dataclass
class PolymarketPosition:
    slug: str
    title: str
    token_id: str
    yes_token_id: str  # the YES / Up token of the same market (order-flow key)
    holds_yes: bool
    shares: float
    avg_price: float
    current_price: float


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


class PolymarketInventory:
    def __init__(self, config: HedgerConfig, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=15.0)
        self._specs: dict[str, MarketSpec | None] = {}
        self.positions: list[PolymarketPosition] = []

    async def aclose(self) -> None:
        await self._http.aclose()

    async def refresh(self) -> None:
        if not self.config.polymarket_wallet:
            self.positions = []
            return
        resp = await self._http.get(
            f"{DATA_API}/v2/positions",
            params={"user": self.config.polymarket_wallet, "limit": 500, "sizeThreshold": 0.1},
        )
        resp.raise_for_status()
        body = resp.json()
        rows = body.get("data", []) if isinstance(body, dict) else body
        positions = []
        for row in rows:
            if row.get("status", "OPEN") != "OPEN" or row.get("redeemable"):
                continue
            holds_yes = int(row.get("outcome_index", 0)) == 0
            positions.append(
                PolymarketPosition(
                    slug=row["slug"],
                    title=row.get("title", ""),
                    token_id=str(row["token_id"]),
                    yes_token_id=str(row["token_id"] if holds_yes else row.get("opposite_token_id", "")),
                    holds_yes=holds_yes,
                    shares=float(row.get("current_size", 0.0)),
                    avg_price=float(row.get("avg_price", 0.0)),
                    current_price=float(row.get("current_price", 0.0)),
                )
            )
        self.positions = positions
        for p in positions:
            if p.slug not in self._specs:
                self._specs[p.slug] = await self._resolve_spec(p.slug, p.title)

    async def _resolve_spec(self, slug: str, title: str) -> MarketSpec | None:
        manual = self.config.manual_markets.get(slug)
        resp = await self._http.get(f"{GAMMA_API}/markets", params={"slug": slug})
        resp.raise_for_status()
        rows = resp.json()
        market = rows[0] if rows else {}
        question = market.get("question") or title
        expiry = _ts(market.get("endDate")) or 0.0
        if manual:
            return MarketSpec(
                slug, "manual", manual.underlying, expiry, beta_pp_per_pct=manual.beta_pp_per_pct
            )
        underlying = detect_underlying(question)
        classified = classify_question(question)
        if underlying is None or classified is None or not expiry:
            log.warning("No delta model for %r; add it to MANUAL_MARKETS_FILE to hedge it", slug)
            return None
        kind, strike = classified
        if kind != "updown":
            return MarketSpec(slug, kind, underlying, expiry, strike=strike)
        start = _ts(market.get("eventStartTime"))
        if start is None:
            log.warning("Up/Down market %r has no eventStartTime; skipping", slug)
            return None
        return MarketSpec(slug, "updown", underlying, expiry, window_start_ts=start)

    async def _updown_strike(self, spec: MarketSpec, now_ts: float) -> MarketSpec:
        """Up/Down resolves against the Binance candle opened at window start: that open is the strike."""
        if spec.strike is not None or spec.window_start_ts is None or now_ts < spec.window_start_ts:
            return spec
        resp = await self._http.get(
            f"{BINANCE_SPOT_API}/api/v3/klines",
            params={
                "symbol": spec.underlying,
                "interval": "1m",
                "startTime": int(spec.window_start_ts * 1000),
                "limit": 1,
            },
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return spec
        resolved = MarketSpec(
            spec.slug,
            "updown",
            spec.underlying,
            spec.expiry_ts,
            strike=float(rows[0][1]),
            window_start_ts=spec.window_start_ts,
        )
        self._specs[spec.slug] = resolved
        return resolved

    def underlyings(self) -> set[str]:
        return {s.underlying for p in self.positions if (s := self._specs.get(p.slug)) is not None}

    async def position_states(self, metric: ToxicityMetric, now_ts: float) -> list[PositionState]:
        """Polymarket positions on `metric.symbol`, each with its perp-equivalent delta (USD)."""
        c = self.config
        sigma = sigma_15m_from_natr(metric.natr_15m, c.min_sigma_15m)
        states = []
        for p in self.positions:
            spec = self._specs.get(p.slug)
            if spec is None or spec.underlying != metric.symbol:
                continue
            if spec.kind == "updown":
                spec = await self._updown_strike(spec, now_ts)
            delta = position_delta_usd(
                spec,
                p.holds_yes,
                p.shares,
                metric.price,
                sigma,
                now_ts,
                no_hedge_final_secs=c.no_hedge_final_secs,
                max_abs_delta_usd=c.max_delta_per_position_usd,
            )
            states.append(
                PositionState(
                    exchange="polymarket",
                    symbol=metric.symbol,
                    instrument=f"{p.slug}:{'YES' if p.holds_yes else 'NO'}",
                    size=p.shares,
                    entry_price=p.avg_price,
                    net_delta_usd=delta,
                    token_id=p.token_id,
                    mark_price=p.current_price,
                )
            )
        return states

    def adverse_flow(self, metric: ToxicityMetric) -> float | None:
        """|delta|-weighted share of Polymarket book flow betting against our positions (0-1)."""
        num = den = 0.0
        for p in self.positions:
            spec = self._specs.get(p.slug)
            share = metric.order_flow_by_token.get(p.yes_token_id)
            if spec is None or spec.underlying != metric.symbol or share is None:
                continue
            weight = p.shares * max(p.current_price, 0.01)
            num += weight * (1.0 - share if p.holds_yes else share)
            den += weight
        return num / den if den else None


def describe(states: list[PositionState]) -> str:
    return json.dumps([{"instrument": s.instrument, "delta_usd": round(s.net_delta_usd, 2)} for s in states])
