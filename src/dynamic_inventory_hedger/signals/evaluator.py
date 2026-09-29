"""Toxicity Threshold Evaluator: FollowSM metrics -> NORMAL / PRE_HEDGING_ALERT / EMERGENCY_HEDGE_EXECUTION.

Bands use `vpin_percentile` (VPIN ranked against the symbol's own history); raw VPIN
thresholds apply only while the percentile is still warming up (None).

    EMERGENCY  vpin rank >= emergency band, OR a liquidity sweep (volume burst + a 15m move
               of several NATRs), OR a PRE condition confirmed by heavy smart-money sweeps
               with Polymarket book flow running against our position
    PRE        vpin rank >= pre-hedge band, OR a toxic 1% book (ask/bid notional ratio)
    NORMAL     otherwise

Hysteresis: escalate immediately; step EMERGENCY -> PRE as soon as the emergency trigger
clears; return to NORMAL only once every signal is back below the rebalance band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.models import ToxicityLevel, ToxicityMetric


@dataclass
class Evaluation:
    level: ToxicityLevel
    reasons: list[str] = field(default_factory=list)


class ToxicityEvaluator:
    def __init__(self, config: HedgerConfig) -> None:
        self.config = config
        self._levels: dict[str, ToxicityLevel] = {}

    def level(self, symbol: str) -> ToxicityLevel:
        return self._levels.get(symbol, ToxicityLevel.NORMAL)

    def _vpin_bands(self, m: ToxicityMetric) -> tuple[str, float, float, float, float]:
        c = self.config
        if m.vpin_percentile is not None:
            return (
                "vpin_percentile",
                m.vpin_percentile,
                c.pre_hedge_percentile,
                c.emergency_percentile,
                c.rebalance_percentile,
            )
        return "vpin(raw)", m.vpin, c.pre_hedge_raw_vpin, c.emergency_raw_vpin, c.rebalance_raw_vpin

    def is_sweep(self, m: ToxicityMetric) -> bool:
        c = self.config
        return (
            m.volume_z_score >= c.sweep_volume_z
            and m.natr_15m > 0
            and abs(m.price_delta_15m_pct) >= c.sweep_move_natr * m.natr_15m
        )

    def evaluate(self, m: ToxicityMetric, adverse_flow: float | None = None) -> Evaluation:
        """`adverse_flow`: share of Polymarket book flow against our net position (0-1), if known."""
        c = self.config
        label, value, pre, emergency, rebalance = self._vpin_bands(m)
        reasons: list[str] = []
        raw = ToxicityLevel.NORMAL

        if value >= emergency:
            raw = ToxicityLevel.EMERGENCY_HEDGE_EXECUTION
            reasons.append(f"{label} {value:.3f} >= {emergency:.2f}")
        elif value >= pre:
            raw = ToxicityLevel.PRE_HEDGING_ALERT
            reasons.append(f"{label} {value:.3f} >= {pre:.2f}")

        toxic_book = m.ob_toxicity_1pct > c.ob_toxicity_threshold
        if toxic_book and raw is ToxicityLevel.NORMAL:
            raw = ToxicityLevel.PRE_HEDGING_ALERT
            reasons.append(f"ob_toxicity_1pct {m.ob_toxicity_1pct:.2f} > {c.ob_toxicity_threshold:.2f}")

        sweep = self.is_sweep(m)
        if sweep:
            raw = ToxicityLevel.EMERGENCY_HEDGE_EXECUTION
            reasons.append(
                f"liquidity sweep: volume_z {m.volume_z_score:+.1f}, 15m move "
                f"{m.price_delta_15m_pct:+.2%} vs natr {m.natr_15m:.2%}"
            )

        if (
            raw is ToxicityLevel.PRE_HEDGING_ALERT
            and m.whale_sweeps_1h_usdt >= c.whale_sweeps_escalate_usd
            and adverse_flow is not None
            and adverse_flow >= c.adverse_flow_escalate
        ):
            raw = ToxicityLevel.EMERGENCY_HEDGE_EXECUTION
            reasons.append(
                f"smart money: ${m.whale_sweeps_1h_usdt:,.0f} swept in 1h, "
                f"{adverse_flow:.0%} of Polymarket book flow against us"
            )

        previous = self.level(m.symbol)
        calm = value < rebalance and not toxic_book and not sweep
        if raw.severity >= previous.severity:
            level = raw
        elif calm:
            level = ToxicityLevel.NORMAL
            reasons.append(f"normalised: {label} {value:.3f} < {rebalance:.2f}")
        else:
            level = max(raw, ToxicityLevel.PRE_HEDGING_ALERT, key=lambda lv: lv.severity)
            reasons.append(f"holding {level.value} until {label} < {rebalance:.2f}")

        self._levels[m.symbol] = level
        return Evaluation(level=level, reasons=reasons)
