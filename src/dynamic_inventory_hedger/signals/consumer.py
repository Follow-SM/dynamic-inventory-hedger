"""FollowSM signal consumer: ConfluenceSnapshot frames -> ToxicityMetric, streamed or polled."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import requests
from followsm_sdk import ConfluenceSnapshot, FollowSMClient, RateLimitExceededException
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from dynamic_inventory_hedger.models import ToxicityMetric

log = logging.getLogger(__name__)

PRICING_URL = "https://follow-sm.com/pricing"


class EnterpriseRequiredError(RuntimeError):
    """The confluence WebSocket rejected the key (missing, invalid or not on ENTERPRISE)."""


def to_metric(s: ConfluenceSnapshot) -> ToxicityMetric:
    m = s.binance_microstructure
    events = s.polymarket_confluence.active_events
    return ToxicityMetric(
        symbol=s.symbol,
        timestamp_ms=s.timestamp_ms,
        price=m.price,
        vpin=m.vpin,
        vpin_percentile=m.vpin_percentile,
        ob_imbalance_l1=m.ob_imbalance_l1,
        depth_imbalance={band: d.imbalance_ratio for band, d in m.depth_bands.items()},
        ob_toxicity_1pct=m.ob_toxicity_1pct,
        ob_imbalance_percentile=m.ob_imbalance_percentile,
        volume_z_score=m.volume_z_score,
        natr_15m=m.natr_15m,
        price_delta_15m_pct=m.price_delta_15m_pct,
        whale_sweeps_1h_usdt=sum(e.smart_money_whale_sweeps_1h_usdt for e in events),
        order_flow_by_token={e.yes_token_id: e.clob_order_flow_imbalance for e in events},
    )


def _is_enterprise_rejection(exc: BaseException) -> bool:
    if isinstance(exc, InvalidStatus):
        return exc.response.status_code in (401, 403)
    if isinstance(exc, ConnectionClosed):
        return exc.rcvd is not None and exc.rcvd.code == 4003
    return False


class SignalConsumer:
    """Yields ToxicityMetric for the watched symbols, or None when the feed is down (fail closed)."""

    def __init__(self, client: FollowSMClient, source: str, poll_interval_secs: float) -> None:
        self.client = client
        self.source = source
        self.poll_interval_secs = poll_interval_secs

    async def metrics(self) -> AsyncIterator[ToxicityMetric | None]:
        if self.source == "stream":
            async for item in self._stream():
                yield item
        else:
            async for item in self._poll():
                yield item

    async def _stream(self) -> AsyncIterator[ToxicityMetric | None]:
        backoff = 1.0
        while True:
            try:
                async for snapshot in self.client.stream_confluence():
                    backoff = 1.0
                    yield to_metric(snapshot)
                log.warning("Confluence stream closed by server")
            except (TimeoutError, WebSocketException, OSError) as exc:
                if _is_enterprise_rejection(exc):
                    raise EnterpriseRequiredError(
                        f"/ws/v1/confluence requires an ENTERPRISE API key ({PRICING_URL}); "
                        "set SIGNAL_SOURCE=poll to use REST instead"
                    ) from exc
                log.warning("Confluence stream error: %s", exc)
            yield None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _poll(self) -> AsyncIterator[ToxicityMetric | None]:
        while True:
            delay = self.poll_interval_secs
            try:
                snapshots = await asyncio.to_thread(self.client.get_confluence_snapshots)
            except RateLimitExceededException as exc:
                log.warning("%s\nUpgrade for higher limits: %s", exc, PRICING_URL)
                yield None
                delay = 60.0
            except requests.RequestException as exc:
                log.warning("FollowSM REST error: %s", exc)
                yield None
            else:
                for snapshot in snapshots:
                    yield to_metric(snapshot)
            await asyncio.sleep(delay)
