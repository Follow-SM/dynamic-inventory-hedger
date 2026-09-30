import pytest

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.execution.policy import decide, inventory_imbalance
from dynamic_inventory_hedger.models import HedgeAction, OrderType, ToxicityLevel, ToxicityMetric
from dynamic_inventory_hedger.risk.controls import (
    ExecutionHalted,
    FailureBreaker,
    Quote,
    check_spread,
    protection_price,
)
from dynamic_inventory_hedger.signals.evaluator import ToxicityEvaluator

L = ToxicityLevel


def metric(
    pctl=0.5,
    vpin=0.4,
    ob_tox=1.0,
    ob_pctl=None,
    vol_z=0.0,
    move=0.0,
    natr=0.002,
    whales=0.0,
    symbol="BTCUSDT",
):
    return ToxicityMetric(
        symbol=symbol,
        timestamp_ms=1,
        price=84_000,
        vpin=vpin,
        vpin_percentile=pctl,
        ob_imbalance_l1=0.5,
        ob_toxicity_1pct=ob_tox,
        ob_imbalance_percentile=ob_pctl,
        volume_z_score=vol_z,
        natr_15m=natr,
        price_delta_15m_pct=move,
        whale_sweeps_1h_usdt=whales,
    )


def test_percentile_bands_and_hysteresis():
    ev = ToxicityEvaluator(HedgerConfig())
    assert ev.evaluate(metric(pctl=0.50)).level is L.NORMAL
    assert ev.evaluate(metric(pctl=0.88)).level is L.PRE_HEDGING_ALERT
    assert ev.evaluate(metric(pctl=0.96)).level is L.EMERGENCY_HEDGE_EXECUTION
    assert ev.evaluate(metric(pctl=0.90)).level is L.PRE_HEDGING_ALERT  # emergency cleared, still elevated
    assert ev.evaluate(metric(pctl=0.70)).level is L.PRE_HEDGING_ALERT  # inside the 0.60-0.85 band: hold
    assert ev.evaluate(metric(pctl=0.55)).level is L.NORMAL


def test_raw_vpin_fallback_while_percentile_warms_up():
    ev = ToxicityEvaluator(HedgerConfig())
    assert ev.evaluate(metric(pctl=None, vpin=0.70)).level is L.NORMAL  # normal for a thin pair
    assert ev.evaluate(metric(pctl=None, vpin=0.92)).level is L.EMERGENCY_HEDGE_EXECUTION


def test_toxic_book_sweep_and_smart_money_escalation():
    cfg = HedgerConfig()
    assert ToxicityEvaluator(cfg).evaluate(metric(ob_tox=3.0)).level is L.PRE_HEDGING_ALERT
    sweep = metric(vol_z=8.0, move=-0.006, natr=0.002)  # 3 NATRs down on a volume burst
    assert ToxicityEvaluator(cfg).evaluate(sweep).level is L.EMERGENCY_HEDGE_EXECUTION
    pre = metric(pctl=0.88, whales=400_000)
    assert ToxicityEvaluator(cfg).evaluate(pre, adverse_flow=0.4).level is L.PRE_HEDGING_ALERT
    assert ToxicityEvaluator(cfg).evaluate(pre, adverse_flow=0.8).level is L.EMERGENCY_HEDGE_EXECUTION


def test_toxic_book_uses_the_symbols_own_tails_once_warm():
    cfg = HedgerConfig()
    # structurally lopsided book that is normal for this symbol: no alert once the percentile is warm
    assert ToxicityEvaluator(cfg).evaluate(metric(ob_tox=3.0, ob_pctl=0.6)).level is L.NORMAL
    assert ToxicityEvaluator(cfg).evaluate(metric(ob_tox=1.0, ob_pctl=0.995)).level is L.PRE_HEDGING_ALERT
    assert ToxicityEvaluator(cfg).evaluate(metric(ob_tox=0.3, ob_pctl=0.005)).level is L.PRE_HEDGING_ALERT
    # warming up: fixed ratio fallback
    assert ToxicityEvaluator(cfg).evaluate(metric(ob_tox=3.0, ob_pctl=None)).level is L.PRE_HEDGING_ALERT
    with pytest.raises(ValueError):
        HedgerConfig(ob_imbalance_percentile_low=0.5, ob_imbalance_percentile_high=0.4)


def test_policy_ratios_slicing_and_order_styles():
    cfg = HedgerConfig(pre_hedge_ratio=0.5, passive_slice_usd=2_000, max_inventory_usd=1e9)
    assert decide("BTCUSDT", L.NORMAL, 10_000, 0.0, cfg) is None  # baseline ratio 0: no hedge
    pre = decide("BTCUSDT", L.PRE_HEDGING_ALERT, 10_000, 0.0, cfg)
    assert (pre.action, pre.order_type, pre.notional_usd, pre.target_hedge_usd) == (
        HedgeAction.SELL,
        OrderType.POST_ONLY_LIMIT,
        2_000,
        -5_000,
    )
    emergency = decide("BTCUSDT", L.EMERGENCY_HEDGE_EXECUTION, 10_000, -2_000, cfg)
    assert (emergency.order_type, emergency.notional_usd, emergency.reduce_only) == (
        OrderType.IOC_LIMIT,
        8_000,
        False,
    )
    unwind = decide("BTCUSDT", L.NORMAL, 10_000, -8_000, cfg)
    assert (unwind.action, unwind.order_type, unwind.reduce_only) == (
        HedgeAction.BUY,
        OrderType.POST_ONLY_LIMIT,
        True,
    )
    assert decide("BTCUSDT", L.PRE_HEDGING_ALERT, 10_000, -4_900, cfg) is None  # below min_rebalance_usd


def test_inventory_limit_forces_a_hedge_even_when_calm():
    cfg = HedgerConfig(max_inventory_usd=25_000)
    signal = decide("BTCUSDT", L.NORMAL, 40_000, 0.0, cfg)
    assert signal.target_hedge_usd == pytest.approx(-15_000)
    assert signal.order_type is OrderType.IOC_LIMIT  # currently over the limit -> act now
    assert inventory_imbalance(40_000, 0.0, cfg) == pytest.approx(1.6)


def test_spread_guard_and_ioc_protection_price():
    cfg = HedgerConfig(max_spread_bps=25, max_slippage_bps=15)
    assert check_spread(Quote(84_000.0, 84_000.1), cfg) is None
    assert "spread" in check_spread(Quote(84_000, 84_500), cfg)
    q = Quote(84_000, 84_000.2)
    assert protection_price(HedgeAction.BUY, q, cfg) == pytest.approx(q.mid * 1.0015)
    assert protection_price(HedgeAction.SELL, q, cfg) == pytest.approx(q.mid * 0.9985)


def test_failure_breaker_trips_after_consecutive_failures_only():
    breaker = FailureBreaker(HedgerConfig(max_consecutive_failures=3))
    breaker.failure(RuntimeError("429"))
    breaker.success()
    breaker.failure(RuntimeError("429"))
    breaker.failure(RuntimeError("503"))
    with pytest.raises(ExecutionHalted):
        breaker.failure(RuntimeError("timeout"))


def test_config_rejects_inconsistent_bands_and_unsafe_live_mode():
    with pytest.raises(ValueError):
        HedgerConfig(pre_hedge_percentile=0.97, emergency_percentile=0.95)
    with pytest.raises(ValueError):
        HedgerConfig(live_trading=True)
