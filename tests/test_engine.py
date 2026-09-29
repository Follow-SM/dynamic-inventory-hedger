"""Engine end to end with fakes: no network, no keys."""

import asyncio

import pytest

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.engine import Hedger
from dynamic_inventory_hedger.inventory.delta import MarketSpec
from dynamic_inventory_hedger.inventory.polymarket import PolymarketInventory, PolymarketPosition
from dynamic_inventory_hedger.models import HedgeAction, OrderType, ToxicityMetric
from dynamic_inventory_hedger.risk.controls import ExecutionHalted, Journal, Quote

NOW = 1_800_000_000.0
TIGHT = Quote(84_000.0, 84_000.1)


class FakeDriver:
    def __init__(self, quote=TIGHT, fail=False):
        self._quote, self.fail, self.hedge, self.orders = quote, fail, 0.0, []

    async def quote(self, symbol):
        return self._quote

    async def hedge_notional_usd(self, symbol, mark):
        return self.hedge

    async def execute(self, signal, quote):
        if self.fail:
            raise OSError("exchange down")
        self.orders.append(signal)
        self.hedge += signal.notional_usd if signal.action is HedgeAction.BUY else -signal.notional_usd
        return signal.notional_usd / quote.mid

    async def cancel_open_orders(self, symbol):
        return None

    async def close(self):
        return None


def inventory_with_btc_above(config):
    inv = PolymarketInventory(config)
    inv.positions = [
        PolymarketPosition(
            "btc-above-84k", "Will Bitcoin be above $84,000?", "yes", "yes", True, 20_000, 0.5, 0.5
        )
    ]
    inv._specs["btc-above-84k"] = MarketSpec(
        "btc-above-84k", "above", "BTCUSDT", NOW + 6 * 3600, strike=84_000
    )
    return inv


def m(pctl, symbol="BTCUSDT"):
    return ToxicityMetric(
        symbol=symbol,
        timestamp_ms=1,
        price=84_000,
        vpin=0.4,
        vpin_percentile=pctl,
        ob_imbalance_l1=0.5,
        ob_toxicity_1pct=1.0,
        volume_z_score=0.0,
        natr_15m=0.0022,
        price_delta_15m_pct=0.0,
    )


def make(tmp_path, driver, **cfg):
    config = HedgerConfig(journal_path=str(tmp_path / "j.jsonl"), max_inventory_usd=1e9, **cfg)
    return Hedger(
        config,
        consumer=None,
        inventory=inventory_with_btc_above(config),
        driver=driver,
        journal=Journal(config.journal_path),
    )


def test_pre_hedge_is_passive_then_emergency_neutralises_with_ioc(tmp_path):
    driver = FakeDriver()
    hedger = make(tmp_path, driver)

    async def scenario():
        assert await hedger.on_metric(m(0.50), NOW) is None
        pre = await hedger.on_metric(m(0.88), NOW)
        emergency = await hedger.on_metric(m(0.97), NOW)
        return pre, emergency

    pre, emergency = asyncio.run(scenario())
    assert pre.order_type is OrderType.POST_ONLY_LIMIT and pre.action is HedgeAction.SELL
    assert emergency.order_type is OrderType.IOC_LIMIT
    assert driver.hedge == pytest.approx(emergency.target_hedge_usd)  # fully neutral after the IOC
    assert (tmp_path / "j.jsonl").read_text().count('"event": "order"') == 2


def test_wide_perp_spread_blocks_trading(tmp_path):
    driver = FakeDriver(quote=Quote(84_000, 84_600))
    hedger = make(tmp_path, driver)
    assert asyncio.run(hedger.on_metric(m(0.97), NOW)) is None
    assert driver.orders == []


def test_repeated_execution_failures_halt_the_hedger(tmp_path):
    hedger = make(tmp_path, FakeDriver(fail=True), max_consecutive_failures=2)

    async def scenario():
        await hedger.on_metric(m(0.97), NOW)
        await hedger.on_metric(m(0.97), NOW)

    with pytest.raises(ExecutionHalted):
        asyncio.run(scenario())


def test_adverse_flow_is_weighted_against_our_side():
    inv = inventory_with_btc_above(HedgerConfig())
    metric = m(0.5)
    metric.order_flow_by_token = {"yes": 0.2}  # 80% of book flow leans NO, we hold YES
    assert inv.adverse_flow(metric) == pytest.approx(0.8)
