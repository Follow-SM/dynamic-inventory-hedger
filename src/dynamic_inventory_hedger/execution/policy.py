"""Hedge policy (pure): toxicity level + Polymarket delta + current perp hedge -> HedgeSignal.

    target_hedge = -ratio(level) * polymarket_delta
    ratio        = baseline (NORMAL) | pre_hedge (PRE_HEDGING_ALERT) | emergency (EMERGENCY_HEDGE_EXECUTION)

The inventory limit is enforced on top: the unhedged remainder |poly_delta + hedge| may never
exceed `max_inventory_usd`, so a large enough position is partly hedged even in NORMAL.

Order style:
    EMERGENCY, or breaching the inventory limit      -> IOC limit (take liquidity now, price-protected)
    PRE_HEDGING_ALERT                                -> post-only limit, sliced (adaptive limit slicing)
    NORMAL (unwinding back toward baseline)          -> post-only, reduce-only rebalance
"""

from __future__ import annotations

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.models import HedgeAction, HedgeSignal, OrderType, ToxicityLevel


def hedge_ratio(level: ToxicityLevel, config: HedgerConfig) -> float:
    return {
        ToxicityLevel.NORMAL: config.baseline_hedge_ratio,
        ToxicityLevel.PRE_HEDGING_ALERT: config.pre_hedge_ratio,
        ToxicityLevel.EMERGENCY_HEDGE_EXECUTION: config.emergency_hedge_ratio,
    }[level]


def inventory_imbalance(poly_delta_usd: float, hedge_usd: float, config: HedgerConfig) -> float:
    """I = unhedged position / max inventory limit (signed; |I| > 1 means over the limit)."""
    return (poly_delta_usd + hedge_usd) / config.max_inventory_usd


def decide(
    symbol: str,
    level: ToxicityLevel,
    poly_delta_usd: float,
    current_hedge_usd: float,
    config: HedgerConfig,
    reason: str = "",
) -> HedgeSignal | None:
    target = -hedge_ratio(level, config) * poly_delta_usd

    limit_breach = False
    unhedged = poly_delta_usd + target
    if abs(unhedged) > config.max_inventory_usd:
        target = -poly_delta_usd + (config.max_inventory_usd if unhedged > 0 else -config.max_inventory_usd)
        limit_breach = abs(poly_delta_usd + current_hedge_usd) > config.max_inventory_usd

    diff = target - current_hedge_usd
    if abs(diff) < config.min_rebalance_usd:
        return None

    urgent = level is ToxicityLevel.EMERGENCY_HEDGE_EXECUTION or limit_breach
    order_type = OrderType.IOC_LIMIT if urgent else OrderType.POST_ONLY_LIMIT
    notional = abs(diff) if urgent else min(abs(diff), config.passive_slice_usd)
    reduces = (
        current_hedge_usd != 0 and abs(target) < abs(current_hedge_usd) and target * current_hedge_usd >= 0
    )

    why = [reason] if reason else []
    if limit_breach:
        why.append(
            f"inventory limit: |{poly_delta_usd + current_hedge_usd:,.0f}| > {config.max_inventory_usd:,.0f}"
        )
    return HedgeSignal(
        action=HedgeAction.BUY if diff > 0 else HedgeAction.SELL,
        symbol=symbol,
        notional_usd=notional,
        order_type=order_type,
        level=level,
        reduce_only=reduces,
        target_hedge_usd=target,
        current_hedge_usd=current_hedge_usd,
        reason="; ".join(why),
    )
