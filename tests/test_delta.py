import math

import pytest

from dynamic_inventory_hedger.inventory.delta import (
    MarketSpec,
    classify_question,
    detect_underlying,
    dprob_dlogspot,
    position_delta_usd,
    sigma_15m_from_natr,
)

NOW = 1_800_000_000.0


def _prob(spec: MarketSpec, spot: float, sigma_15m: float) -> float:
    """Reference prices the delta should be the derivative of."""
    tau = spec.expiry_ts - NOW
    sigma = sigma_15m * math.sqrt(tau / 900)
    x = math.log(spot / spec.strike)
    n = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    return {
        "above": n((x - sigma**2 / 2) / sigma),
        "below": 1 - n((x - sigma**2 / 2) / sigma),
        "touch_up": 2 * n(x / sigma),
        "touch_down": 2 * n(-x / sigma),
    }[spec.kind]


@pytest.mark.parametrize(
    "kind,strike,spot",
    [
        ("above", 84_000, 84_500),
        ("below", 84_000, 83_700),
        ("touch_up", 90_000, 84_000),
        ("touch_down", 80_000, 84_000),
    ],
)
def test_delta_matches_finite_difference_of_the_price(kind, strike, spot):
    spec = MarketSpec("m", kind, "BTCUSDT", NOW + 6 * 3600, strike=float(strike))
    sigma, h = 0.0015, 1e-5
    numeric = (_prob(spec, spot * math.exp(h), sigma) - _prob(spec, spot * math.exp(-h), sigma)) / (2 * h)
    assert dprob_dlogspot(spec, spot, sigma, NOW) == pytest.approx(numeric, rel=1e-4)


def test_updown_has_no_delta_before_window_and_uses_open_as_strike_after():
    spec = MarketSpec("u", "updown", "BTCUSDT", NOW + 3600, window_start_ts=NOW + 600)
    assert dprob_dlogspot(spec, 84_000, 0.002, NOW) == 0.0
    opened = MarketSpec("u", "updown", "BTCUSDT", NOW + 3600, strike=84_000, window_start_ts=NOW - 60)
    above = MarketSpec("a", "above", "BTCUSDT", NOW + 3600, strike=84_000)
    assert dprob_dlogspot(opened, 84_100, 0.002, NOW) == dprob_dlogspot(above, 84_100, 0.002, NOW) > 0


def test_position_delta_sign_cap_and_expiry_cutoff():
    spec = MarketSpec("a", "above", "BTCUSDT", NOW + 3600, strike=84_000)
    kw = {"no_hedge_final_secs": 300, "max_abs_delta_usd": 50_000}
    yes = position_delta_usd(spec, True, 1_000, 84_000, 0.002, NOW, **kw)
    no = position_delta_usd(spec, False, 1_000, 84_000, 0.002, NOW, **kw)
    assert yes > 0 and no == pytest.approx(-yes)
    assert position_delta_usd(spec, True, 10**7, 84_000, 0.002, NOW, **kw) == 50_000
    assert position_delta_usd(spec, True, 1_000, 84_000, 0.002, spec.expiry_ts - 60, **kw) == 0.0


def test_manual_market_uses_configured_beta():
    spec = MarketSpec("fed", "manual", "BTCUSDT", NOW + 86_400, beta_pp_per_pct=0.8)
    kw = {"no_hedge_final_secs": 300, "max_abs_delta_usd": 1e9}
    # 1,000 YES shares, +0.8pp per +1% -> $8 per 1% -> $800 per 1.0 log-move
    assert position_delta_usd(spec, True, 1_000, 84_000, 0.002, NOW, **kw) == pytest.approx(800)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Will the price of Bitcoin be above $76,000 on September 29?", ("above", 76_000.0)),
        ("Will Bitcoin dip to $70k in October?", ("touch_down", 70_000.0)),
        ("Will Ethereum reach $4,000 by December 31?", ("touch_up", 4_000.0)),
        ("Bitcoin Up or Down - September 29, 6AM ET", ("updown", None)),
        ("Bitcoin above 86,000 on September 29, 7AM ET?", ("above", 86_000.0)),
        ("Will BTC close above $86,000.", ("above", 86_000.0)),
        ("Will BTC dominance be above 60% on Friday?", None),
        ("Will BTC stay above 86,000 mark?", ("above", 86_000.0)),
        ("Will the Fed cut rates in October?", None),
    ],
)
def test_classify_question(question, expected):
    assert classify_question(question) == expected


def test_detect_underlying_and_sigma_floor():
    assert detect_underlying("Will the price of Bitcoin be above $76,000?") == "BTCUSDT"
    assert detect_underlying("Will ETH flip...") == "ETHUSDT"
    assert detect_underlying("Will the Fed cut rates?") is None
    assert sigma_15m_from_natr(0.00221, 0.0005) == pytest.approx(0.00221 / math.sqrt(8 / math.pi))
    assert sigma_15m_from_natr(0.0, 0.0005) == 0.0005
