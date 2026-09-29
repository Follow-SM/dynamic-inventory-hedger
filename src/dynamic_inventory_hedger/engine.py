"""Engine: FollowSM frame -> toxicity level -> Polymarket delta -> hedge decision -> risk -> perp order."""

from __future__ import annotations

import asyncio
import logging
import time

import ccxt.async_support as ccxt
import httpx

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.execution.drivers import ExecutionDriver
from dynamic_inventory_hedger.execution.policy import decide, inventory_imbalance
from dynamic_inventory_hedger.inventory.polymarket import PolymarketInventory
from dynamic_inventory_hedger.models import HedgeSignal, ToxicityMetric
from dynamic_inventory_hedger.risk.controls import FailureBreaker, Journal, check_spread
from dynamic_inventory_hedger.signals.consumer import SignalConsumer
from dynamic_inventory_hedger.signals.evaluator import ToxicityEvaluator

log = logging.getLogger(__name__)

RETRYABLE = (ccxt.BaseError, httpx.HTTPError, OSError)


class Hedger:
    def __init__(
        self,
        config: HedgerConfig,
        consumer: SignalConsumer,
        inventory: PolymarketInventory,
        driver: ExecutionDriver,
        journal: Journal,
    ) -> None:
        self.config = config
        self.consumer = consumer
        self.inventory = inventory
        self.driver = driver
        self.journal = journal
        self.evaluator = ToxicityEvaluator(config)
        self.breaker = FailureBreaker(config)
        self._last_cycle: dict[str, float] = {}

    async def refresh_inventory_forever(self) -> None:
        while True:
            try:
                await self.inventory.refresh()
            except httpx.HTTPError as exc:
                log.warning("Polymarket positions refresh failed (keeping last snapshot): %s", exc)
            await asyncio.sleep(self.config.positions_refresh_secs)

    async def run(self) -> None:
        await self.inventory.refresh()
        refresher = asyncio.create_task(self.refresh_inventory_forever())
        try:
            async for metric in self.consumer.metrics():
                if metric is None:
                    log.warning("FollowSM feed unavailable: holding current hedges, no new orders")
                    continue
                if metric.symbol not in self.inventory.underlyings():
                    continue
                now = time.time()
                if now - self._last_cycle.get(metric.symbol, 0.0) < self.config.min_cycle_secs:
                    continue
                self._last_cycle[metric.symbol] = now
                await self.on_metric(metric, now)
        finally:
            refresher.cancel()

    async def on_metric(self, metric: ToxicityMetric, now: float) -> HedgeSignal | None:
        symbol = metric.symbol
        states = await self.inventory.position_states(metric, now)
        poly_delta = sum(s.net_delta_usd for s in states)
        evaluation = self.evaluator.evaluate(metric, self.inventory.adverse_flow(metric))
        try:
            quote = await self.driver.quote(symbol)
            hedge = await self.driver.hedge_notional_usd(symbol, quote.mid)
        except RETRYABLE as exc:
            self.breaker.failure(exc)
            return None

        signal = decide(
            symbol, evaluation.level, poly_delta, hedge, self.config, "; ".join(evaluation.reasons)
        )
        self.journal.write(
            "decision",
            symbol=symbol,
            level=evaluation.level.value,
            reasons=evaluation.reasons,
            vpin=metric.vpin,
            vpin_percentile=metric.vpin_percentile,
            poly_delta_usd=poly_delta,
            hedge_usd=hedge,
            imbalance=inventory_imbalance(poly_delta, hedge, self.config),
            positions=[s.model_dump() for s in states],
            mid=quote.mid,
            signal=signal.model_dump() if signal else None,
        )
        if signal is None:
            return None

        refusal = check_spread(quote, self.config)
        if refusal:
            log.warning("%s: not hedging, %s", symbol, refusal)
            self.journal.write("refused", symbol=symbol, reason=refusal)
            return None

        log.info(
            "%s %s | poly Δ=%+.0f hedge=%+.0f -> target=%+.0f | %s %s $%.0f | %s",
            symbol,
            evaluation.level.value,
            poly_delta,
            hedge,
            signal.target_hedge_usd,
            signal.order_type.value,
            signal.action.value,
            signal.notional_usd,
            signal.reason,
        )
        try:
            await self.driver.cancel_open_orders(symbol)
            filled = await self.driver.execute(signal, quote)
        except RETRYABLE as exc:
            self.journal.write("error", symbol=symbol, error=str(exc))
            self.breaker.failure(exc)
            return None
        self.breaker.success()
        self.journal.write("order", symbol=symbol, signal=signal.model_dump(), filled=filled, mid=quote.mid)
        return signal
