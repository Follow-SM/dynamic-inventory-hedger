"""Backtest: a cross-venue market maker that prices Polymarket equity Up/Down books off Binance TradFi perps.

Universe: every daily "<TICKER> Up or Down on <date>?" market in tick-microstructure-research's Polymarket
store (`tmr tradfi pm-fetch`), from the previous official close to the resolving close ("a night").
The book quotes the Up token (a Down order is the same book mirrored) and is filled only by the
historical taker orders that actually printed:

    fair value      FV = N(d), d = (ln(S/K) - σ²/2) / σ, S = perp / (1 + basis at the previous close),
                    K = the previous official close, σ² = variance remaining to the close. This is the
                    hedger's own digital model (inventory/delta.py). S is the last perp print on a 1-second
                    grid at decision time; σ is one of
                      session   per-minute RTH/pre/post/overnight/closed variances + the open jump, from
                                the previous 20 trading days (Part 3's session-aware forecast),
                      wall      natr_15m spread evenly over wall-clock time (the repo default), or
                      anchor    mean ln(C/S)² at the same minutes-to-close over the symbol's previous 20
                                nights (±15 min pooled). Added after tick-microstructure-research's
                                calibration table showed the session sum overstates RTH variance.
    quotes          centre ± HALF_SPREAD, centre = FV (or Polymarket's own mid for the naive book)
                    minus an inventory skew; FV quotes never lean past Polymarket's last mid ± 0.5¢,
                    so the book stays a maker; no quotes outside PRICE_BAND; inventory capped.
    fills           a taker order fills us only if it traded strictly through our price (its worst
                    print below our bid / above our ask), at our price, up to QUOTE_SHARES per order.
                    Orders that only touch our price never fill us (we are assumed last in the queue).
                    Quotes are recomputed for every order from information at (order time - latency).
    fees            Polymarket makers pay nothing; the finance-category taker fee (shares x 0.04 x
                    p(1-p)) applies to the early-exit flatten. The 25% maker rebate is paid pro rata
                    from a daily pool and is not credited here. Perp hedges pay 5 bps taker and walk
                    the Binance aggTrades tape (Part 3's IOC model).

Strategies (same nights, same fills logic):

    pm_mid                 quotes around Polymarket's own mid; unhedged; held to resolution
    pm_mid_shield          + session-σ delta hedge on the perp, exit 1 h before resolution, toxicity breaker
    fv_wall / fv_session   quotes around the perp fair value (wall-clock / session σ); unhedged; held
    fv_session_hedged      + session-σ delta hedge
    fv_session_hedged_exit + exit one hour before resolution
    shield_sword           + toxicity breaker: the full system
    shield_sword_unhedged  the full system without the perp hedge
    shield_sword_wall      the full system with wall-clock σ for both quoting and hedging
    shield_sword_lat_*     the full system with slower quote updates (1 s, 5 s, 60 s)
    shield_sword_exit_5c   the full system paying 5¢ (not 2¢) through the mid to exit
    shield_sword_deadband_* hedge deadband $2,000 / $5,000 (default $500)
    offhours_sword*        FV quotes only until 09:25 ET, then flat (unhedged / hedged / mid-centred)
    fv_anchor*             FV quotes with anchor σ: unhedged and held; 4¢ half-spread; + breaker and
                           early exit; + hedge ($5,000 deadband); slower quote updates
    *_hs_*                 half-spread sweep (1¢ / 3¢ / 4¢; default 2¢)
    placebo_<system>_<k>   shield_sword, offhours_sword and fv_anchor quoting off another night's perp
                           path (same symbol, aligned on time to the close, drawn from the previous 40
                           nights): perp-shaped noise with no information about this night. 20 seeds.

    offhours_sword and fv_anchor were added after looking at results (see above); the placebo seeds and
    the split-half totals in results_confluence.json are the check against that selection.

Shield: session σ (quoting and hedge ratio), early exit (the Part 2 fix for the digital expiry jump),
and an `ob_toxicity_1pct` breaker: quotes are pulled while max(ask/bid, bid/ask) depth within ±1% >
2.5 on the last completed minute, and re-armed only after 5 consecutive minutes below 2.0
(hysteresis). Fail-closed: no quote without a perp print in the last 60 s, a Polymarket mid, or a depth
reading; no hedge order without a perp bar or tape.

    pip install -e ".[tradfi]" && tmr tradfi fetch && tmr tradfi pm-fetch
    python research/backtest_confluence.py
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
from backtest_tradfi import (
    ASSETS,
    CAL,
    FREEZE_MIN,
    HERE,
    TAKER_FEE,
    TRADFI_ROOT,
    Tape,
    TapeCache,
    cvar,
    variance_models,
    walk,
)
from tick_microstructure.research import polymarket_study as ps
from tick_microstructure.research.data.fetch import EQUITY_SYMBOLS
from tick_microstructure.research.data.polymarket import PmMarket, load_markets, load_prices, load_trades
from tick_microstructure.research.data.store import TradfiStore
from tick_microstructure.research.tradfi import MIN_MS, MinuteFeatures, load_features

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.inventory.delta import RANGE_TO_SIGMA, MarketSpec, position_delta_usd

F = npt.NDArray[np.float64]
QUOTE_SHARES = 100.0
MAX_INVENTORY = 500.0
HALF_SPREAD = 0.02
SKEW = HALF_SPREAD / MAX_INVENTORY  # the centre moves one half-spread at the inventory cap
TICK = 0.001
MAKER_LEAN = 0.005
PRICE_BAND = (0.03, 0.97)
PM_TAKER_RATE = 0.04
EXIT_MIN = 60
EXIT_CUSHION = 0.02
TOX_TRIP, TOX_CLEAR, TOX_REARM_MIN = 2.5, 2.0, 5
LATENCY_MS = 250
SLOW_LATENCIES_MS = (1_000, 5_000, 60_000)
PERP_STALE_S = 60
HEDGE_DEADBAND_USD = 500.0
ANCHOR_NIGHTS = 20
ANCHOR_SMOOTH_MIN = 15  # ± minutes-to-close pooled into each anchor estimate
MIN_HISTORY = 5
SWEEP_HALF_SPREADS = (0.01, 0.03, 0.04)
SWEEP_DEADBANDS_USD = (2_000.0, 5_000.0)
PLACEBO_SEEDS = 20
DONOR_NIGHTS = 40
MIN_DONORS = 5
TRADING_DAYS = 252


@dataclass(frozen=True)
class Strategy:
    name: str
    centre: str  # "fv" | "pm"
    sigma: str = "session"  # "session" | "wall"
    hedge: bool = False
    exit: str = "none"  # "none" (hold to resolution) | "1h" (1 h before resolution) | "open" (09:25 ET)
    breaker: bool = False
    latency_ms: int = LATENCY_MS
    exit_cushion: float = EXIT_CUSHION
    half_spread: float = HALF_SPREAD
    deadband_usd: float = HEDGE_DEADBAND_USD


FULL = Strategy("shield_sword", "fv", hedge=True, exit="1h", breaker=True)
# chosen after the per-session calibration diagnostic (see the report): quote off the perp only while the
# stock is closed, flatten before the open
OFFHOURS = Strategy("offhours_sword", "fv", exit="open", breaker=True)
# chosen after tmr's per-session calibration table: σ measured directly at each minutes-to-close
FV_ANCHOR = Strategy("fv_anchor", "fv", sigma="anchor")
PLACEBO_SYSTEMS = (FULL, OFFHOURS, FV_ANCHOR)


def strategies() -> list[Strategy]:
    return [
        Strategy("pm_mid", "pm"),
        Strategy("pm_mid_shield", "pm", hedge=True, exit="1h", breaker=True),
        Strategy("fv_wall", "fv", sigma="wall"),
        Strategy("fv_session", "fv"),
        Strategy("fv_session_hedged", "fv", hedge=True),
        Strategy("fv_session_hedged_exit", "fv", hedge=True, exit="1h"),
        FULL,
        replace(FULL, name="shield_sword_unhedged", hedge=False),
        replace(FULL, name="shield_sword_wall", sigma="wall"),
        *(replace(FULL, name=f"shield_sword_lat_{ms}", latency_ms=ms) for ms in SLOW_LATENCIES_MS),
        replace(FULL, name="shield_sword_exit_5c", exit_cushion=0.05),
        OFFHOURS,
        replace(OFFHOURS, name="offhours_sword_hedged", hedge=True),
        replace(OFFHOURS, name="offhours_pm_mid", centre="pm"),
        *(replace(OFFHOURS, name=f"offhours_sword_lat_{ms}", latency_ms=ms) for ms in SLOW_LATENCIES_MS),
        *(
            replace(base, name=f"{base.name}_hs_{round(hs * 100)}c", half_spread=hs)
            for base in (Strategy("fv_session", "fv"), OFFHOURS)
            for hs in SWEEP_HALF_SPREADS
        ),
        *(replace(FULL, name=f"shield_sword_deadband_{b}", deadband_usd=b) for b in SWEEP_DEADBANDS_USD),
        FV_ANCHOR,
        replace(FV_ANCHOR, name="fv_anchor_hs_4c", half_spread=0.04),
        replace(FV_ANCHOR, name="fv_anchor_breaker_exit", breaker=True, exit="1h"),
        replace(
            FV_ANCHOR, name="fv_anchor_shield", hedge=True, breaker=True, exit="1h", deadband_usd=5_000.0
        ),
        *(replace(FV_ANCHOR, name=f"fv_anchor_lat_{ms}", latency_ms=ms) for ms in SLOW_LATENCIES_MS),
    ]


# ───────────────────────────── inputs per night ─────────────────────────────


@dataclass
class NightData:
    sym: str
    day: date
    strike: float
    basis: float
    payout: float  # Up
    t0_s: int  # epoch second of the first path point (the previous close)
    quote_from_ms: float
    exit_ms: float
    open_exit_ms: float
    close_ms: float
    # taker orders (one row per order): time, +1 = buys Up, worst price, shares
    o_t: F
    o_dir: F
    o_through: F
    o_shares: F
    hist_t: F
    hist_p: F
    # minute grid [i0, i1)
    t_min: F
    perp_close: F
    missing: npt.NDArray[np.bool_]
    rem_var: F  # session variance from the start of each minute to the close
    rem_var_anchor: F  # mean ln(C/S)² at the same minutes-to-close over the previous nights (NaN in warm-up)
    sig_wall: F  # wall-clock σ_15m from natr_15m (known at the minute's end)
    tox_ok: npt.NDArray[np.bool_]  # breaker state at the minute's end

    def minute(self, t_ms: float) -> int:
        return int((t_ms - self.t_min[0]) // MIN_MS)


def breaker_path(obt: F) -> npt.NDArray[np.bool_]:
    """True where quoting is allowed at the end of each minute (trip > 2.5, re-arm after 5 min < 2.0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.maximum(obt, 1 / obt)
    ok = np.zeros(ratio.size, dtype=bool)
    tripped, calm = False, 0
    for k, r in enumerate(ratio.tolist()):
        if not math.isfinite(r):
            continue  # no depth reading: fail closed for this minute, state unchanged
        if r > TOX_TRIP:
            tripped, calm = True, 0
        elif tripped:
            calm = calm + 1 if r < TOX_CLEAR else 0
            tripped = calm < TOX_REARM_MIN
        ok[k] = not tripped
    return ok


def perp_path(tape: Tape, t0_s: int, n_sec: int) -> F:
    """Last perp print at or before each second (NaN if older than PERP_STALE_S)."""
    t = np.concatenate([tape.buys.t, tape.sells.t])
    px = np.concatenate([tape.buys.px, tape.sells.px])
    order = np.argsort(t, kind="stable")
    t, px = t[order], px[order]
    grid = (t0_s + np.arange(n_sec, dtype=np.float64)) * 1000
    j = np.searchsorted(t, grid, side="right") - 1
    out = np.where(j >= 0, px[np.maximum(j, 0)], np.nan)
    out[(j < 0) | (grid - t[np.maximum(j, 0)] > PERP_STALE_S * 1000)] = np.nan
    return out


def _orders(store: TradfiStore, slug: str, lo: float, hi: float) -> tuple[F, F, F, F, float]:
    tr = load_trades(store, slug)
    keep = (tr.t >= lo) & (tr.t < hi)
    if not keep.any():
        e = np.empty(0)
        return e, e, e, e, 0.0
    order, t, d, p, sh = tr.order[keep], tr.t[keep], tr.toward_yes[keep], tr.price_yes[keep], tr.shares[keep]
    ids, first = np.unique(order, return_index=True)
    inv = np.searchsorted(ids, order)
    n = ids.size
    lo_p = np.full(n, np.inf)
    hi_p = np.full(n, -np.inf)
    np.minimum.at(lo_p, inv, p)
    np.maximum.at(hi_p, inv, p)
    shares = np.bincount(inv, weights=sh, minlength=n)
    t0 = np.full(n, np.inf)
    np.minimum.at(t0, inv, t)
    direction = d[first]
    through = np.where(direction > 0, hi_p, lo_p)
    o = np.argsort(t0, kind="stable")
    return t0[o], direction[o], through[o], shares[o], float(sh.sum())


def night_data(
    store: TradfiStore,
    f: MinuteFeatures,
    nt: ps.Night,
    market: PmMarket,
    rem_var_all: F,
    rem_var_anchor: F,
    sig_wall_all: F,
    tox_all: npt.NDArray[np.bool_],
) -> tuple[NightData, float]:
    close_ms = CAL.close_ms(nt.day)
    t_start = float(f.t[nt.i0])
    quote_from = max(t_start, market.start_ms if math.isfinite(market.start_ms) else t_start)
    o_t, o_dir, o_through, o_shares, taker_shares = _orders(store, market.slug, quote_from, close_ms)
    hist_t, hist_p = load_prices(store, market.slug)
    sl = slice(nt.i0, nt.i1)
    nd = NightData(
        sym=nt.sym,
        day=nt.day,
        strike=nt.close_prev,
        basis=nt.basis,
        payout=nt.payout_up,
        t0_s=int(t_start // 1000),
        quote_from_ms=quote_from,
        exit_ms=close_ms - EXIT_MIN * MIN_MS,
        open_exit_ms=CAL.open_ms(nt.day) - FREEZE_MIN * MIN_MS,
        close_ms=close_ms,
        o_t=o_t,
        o_dir=o_dir,
        o_through=o_through,
        o_shares=o_shares,
        hist_t=hist_t,
        hist_p=hist_p,
        t_min=f.t[sl],
        perp_close=f.close[sl],
        missing=f.missing[sl],
        rem_var=rem_var_all,
        rem_var_anchor=rem_var_anchor,
        sig_wall=sig_wall_all[sl],
        tox_ok=tox_all[sl],
    )
    return nd, taker_shares


# ───────────────────────────── simulation ─────────────────────────────


@dataclass
class Result:
    sym: str
    day: str
    pnl: float
    pm_pnl: float
    hedge_pnl: float  # perp P&L after fees
    exit_cost: float  # cushion + taker fee paid to flatten Polymarket early
    fills: int
    shares: float
    max_inventory: float
    capture: float  # our filled shares / all taker shares in the quoting window


def _fair(s: float, k: float, var: float) -> float:
    sd = math.sqrt(var)
    d = (math.log(s / k) - var / 2) / sd
    return 0.5 * (1 + math.erf(d / math.sqrt(2)))


def _pm_mid(nd: NightData, t: float) -> float:
    j = int(np.searchsorted(nd.hist_t, t, side="right")) - 1
    return float(nd.hist_p[j]) if j >= 0 and t - nd.hist_t[j] <= 5 * MIN_MS else float("nan")


def _var_left(nd: NightData, s: Strategy, k: int, t: float, floor: float) -> float:
    tau_15 = (nd.close_ms - t) / 900_000
    if s.sigma == "wall":
        sig = nd.sig_wall[k - 1] if k >= 1 else float("nan")
        var = sig * sig * tau_15
    elif s.sigma == "anchor":
        var = float(nd.rem_var_anchor[k]) if 0 <= k < nd.rem_var_anchor.size else float("nan")
    else:
        var = float(nd.rem_var[k]) if 0 <= k < nd.rem_var.size else float("nan")
    return max(var, floor * floor * tau_15) if math.isfinite(var) else float("nan")


def simulate(
    nd: NightData, tape: Tape, s: Strategy, quote_path: F, taker_shares: float, config: HedgerConfig
) -> Result:
    stop_ms = {"none": nd.close_ms - MIN_MS, "1h": nd.exit_ms, "open": nd.open_exit_ms}[s.exit]
    spec = MarketSpec(f"{nd.sym}-{nd.day}", "above", nd.sym, nd.close_ms / 1000, strike=nd.strike)
    inv = cash = perp_q = perp_cash = perp_fees = 0.0
    fills = 0
    shares = max_inv = 0.0
    j = 0
    n_orders = nd.o_t.size
    for k in range(nd.t_min.size):
        m_end = float(nd.t_min[k]) + MIN_MS
        while j < n_orders and nd.o_t[j] < m_end:
            t = float(nd.o_t[j])
            j += 1
            td = t - s.latency_ms
            if t >= stop_ms or td < nd.quote_from_ms:
                continue
            km = nd.minute(td)
            if km < 1 or (s.breaker and not nd.tox_ok[km - 1]):
                continue
            mid = _pm_mid(nd, td)
            if not math.isfinite(mid):
                continue
            if s.centre == "pm":
                centre = mid
            else:
                sec = int(td // 1000) - nd.t0_s
                spot = float(quote_path[sec]) / (1 + nd.basis) if 0 <= sec < quote_path.size else float("nan")
                var = _var_left(nd, s, km, td, config.min_sigma_15m)
                if not (math.isfinite(spot) and math.isfinite(var)):
                    continue
                centre = _fair(spot, nd.strike, var)
            centre -= SKEW * inv
            bid = math.floor((centre - s.half_spread) / TICK) * TICK
            ask = math.ceil((centre + s.half_spread) / TICK) * TICK
            if s.centre == "fv":
                bid, ask = min(bid, mid - MAKER_LEAN), max(ask, mid + MAKER_LEAN)
            through, size = float(nd.o_through[j - 1]), float(nd.o_shares[j - 1])
            if nd.o_dir[j - 1] < 0 and bid >= PRICE_BAND[0] and bid > through:
                q = min(size, QUOTE_SHARES, MAX_INVENTORY - inv)
                inv, cash = inv + q, cash - q * bid
            elif nd.o_dir[j - 1] > 0 and ask <= PRICE_BAND[1] and ask < through:
                q = min(size, QUOTE_SHARES, MAX_INVENTORY + inv)
                inv, cash = inv - q, cash + q * ask
            else:
                q = 0.0
            if q > 0:
                fills += 1
                shares += q
                max_inv = max(max_inv, abs(inv))
        if s.hedge and not nd.missing[k] and (inv != 0 or perp_q != 0) and m_end <= stop_ms:
            now = m_end + s.latency_ms
            spot = float(nd.perp_close[k]) / (1 + nd.basis)
            tau = (nd.close_ms - m_end) / 1000
            var = _var_left(nd, s, k + 1, m_end, config.min_sigma_15m)
            if math.isfinite(var) and tau > 0:
                sigma_15m = math.sqrt(var * 900 / tau)
                target = -position_delta_usd(
                    spec,
                    True,
                    inv,
                    spot,
                    sigma_15m,
                    m_end / 1000,
                    no_hedge_final_secs=config.no_hedge_final_secs,
                    max_abs_delta_usd=config.max_delta_per_position_usd,
                )
                diff = target - perp_q * float(nd.perp_close[k])
                if abs(diff) >= s.deadband_usd or (target == 0 and perp_q != 0):
                    notional = abs(diff) if target != 0 else abs(perp_q) * float(nd.perp_close[k])
                    price, _thin = walk(tape, now, diff > 0, notional) if notional >= 1 else (math.nan, True)
                    if math.isfinite(price) and price > 0:
                        qty = notional / price * (1 if diff > 0 else -1)
                        perp_q += qty
                        perp_cash -= qty * price
                        perp_fees += notional * TAKER_FEE
    exit_cost = 0.0
    settle = nd.payout
    if s.exit != "none" and inv != 0:
        mid = _pm_mid(nd, stop_ms)
        if math.isfinite(mid):
            px = min(max(mid - s.exit_cushion * np.sign(inv), 0.001), 0.999)
            fee = abs(inv) * PM_TAKER_RATE * px * (1 - px)
            exit_cost = abs(inv) * s.exit_cushion + fee
            cash += inv * px - fee
            inv = 0.0
    pm_pnl = cash + inv * settle
    if perp_q != 0:
        t_flat = (stop_ms if s.exit != "none" else nd.close_ms) + s.latency_ms
        ref = float(nd.perp_close[min(nd.minute(t_flat) - 1, nd.t_min.size - 1)])
        notional = abs(perp_q) * ref
        price, _thin = walk(tape, t_flat, perp_q < 0, notional) if notional >= 1 else (math.nan, True)
        if not (math.isfinite(price) and price > 0):
            price = ref
        perp_cash += perp_q * price
        perp_fees += notional * TAKER_FEE
    hedge_pnl = perp_cash - perp_fees
    return Result(
        nd.sym,
        nd.day.isoformat(),
        pm_pnl + hedge_pnl,
        pm_pnl,
        hedge_pnl,
        exit_cost,
        fills,
        shares,
        max_inv,
        shares / taker_shares if taker_shares else 0.0,
    )


# ───────────────────────────── per symbol ─────────────────────────────


def _session_var(f: MinuteFeatures, nt: ps.Night, vm: Any) -> F:
    sl = slice(nt.i0, nt.i1)
    var = np.array([vm.per_minute[int(x)] for x in f.session[sl]])
    mfo = CAL.minutes_from_open(f.t[sl])
    var[np.abs(mfo + 0.5) <= FREEZE_MIN] = vm.open_jump / (2 * FREEZE_MIN)
    return np.asarray(np.cumsum(var[::-1])[::-1], dtype=np.float64)


def _sq_by_ttc(f: MinuteFeatures, nt: ps.Night) -> F:
    """ln(C/S)² indexed by minutes to the close, S = spot at the start of that minute."""
    s = f.close[nt.i0 - 1 : nt.i1 - 1] / (1 + nt.basis)  # S at the start of minute k, k = 0 .. n-1
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.log(nt.close_day / s) ** 2
    sq[f.missing[nt.i0 - 1 : nt.i1 - 1]] = np.nan
    return np.concatenate([[np.nan], sq[::-1]])  # [ttc] for ttc = 0 .. n


def _anchor_var(history: deque[F], n: int) -> F:
    """Remaining variance at the start of each minute k (ttc = n - k), from the previous nights."""
    if len(history) < MIN_HISTORY:
        return np.full(n, np.nan)
    stack = np.full((len(history), n + 1), np.nan)
    for r, h in enumerate(history):
        m = min(h.size, n + 1)
        stack[r, :m] = h[:m]
    ok = np.isfinite(stack)
    w = np.ones(2 * ANCHOR_SMOOTH_MIN + 1)
    total = np.convolve(np.where(ok, stack, 0.0).sum(axis=0), w, mode="same")
    count = np.convolve(ok.sum(axis=0).astype(np.float64), w, mode="same")
    nights_with = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_ttc = np.where(nights_with >= MIN_HISTORY, total / count, np.nan)
    return np.asarray(by_ttc[n - np.arange(n)], dtype=np.float64)


def run_symbol(sym: str) -> tuple[dict[str, list[Result]], dict[str, int]]:
    store = TradfiStore(TRADFI_ROOT)
    f = load_features(store, CAL, sym)
    markets = load_markets(store)
    nights, skipped = ps.build_nights(store, CAL, f, sym, markets)
    listed = {m.day: m for m in markets if m.kind == "updown" and m.sym == sym}
    variances = variance_models(f)
    config = HedgerConfig()
    natr = np.nan_to_num(f.natr_15m, nan=0.0)
    sig_wall = np.maximum(natr / RANGE_TO_SIGMA, config.min_sigma_15m)
    tox = breaker_path(f.obt)
    tapes = TapeCache(sym, size=2)
    donors: deque[F] = deque(maxlen=DONOR_NIGHTS)
    sq_hist: deque[F] = deque(maxlen=ANCHOR_NIGHTS)  # ln(C/S)² by minutes-to-close, newest last
    runs: dict[str, list[Result]] = {}
    for nt in nights:
        vm = variances.get(nt.day)
        if vm is None:
            skipped["variance warm-up"] = skipped.get("variance warm-up", 0) + 1
            continue
        nd, taker_shares = night_data(
            store, f, nt, listed[nt.day], _session_var(f, nt, vm), _anchor_var(sq_hist, nt.n), sig_wall, tox
        )
        sq_hist.append(_sq_by_ttc(f, nt))
        if nd.o_t.size == 0:
            skipped["no taker orders"] = skipped.get("no taker orders", 0) + 1
            continue
        prev = CAL.previous_trading_day(nt.day)
        tape = tapes.get(prev, nt.day)
        n_sec = int(nd.close_ms // 1000) - nd.t0_s + 1
        path = perp_path(tape, nd.t0_s, n_sec)
        for s in strategies():
            runs.setdefault(s.name, []).append(simulate(nd, tape, s, path, taker_shares, config))
        pool = [d for d in donors if d.size >= n_sec]
        if len(pool) >= MIN_DONORS:
            rng = random.Random(f"{sym}-{nt.day}")
            p0 = (
                float(path[np.flatnonzero(np.isfinite(path))[0]]) if np.isfinite(path).any() else float("nan")
            )
            fakes = []
            for _ in range(PLACEBO_SEEDS):
                donor = rng.choice(pool)[-n_sec:].astype(np.float64)
                first = np.flatnonzero(np.isfinite(donor))
                fakes.append(donor / donor[first[0]] * p0 if first.size else donor)
            for system in PLACEBO_SYSTEMS:
                runs.setdefault(f"{system.name}_placebo_nights", []).append(runs[system.name][-1])
                for k, fake in enumerate(fakes):
                    runs.setdefault(f"placebo_{system.name}_{k}", []).append(
                        simulate(nd, tape, system, fake, taker_shares, config)
                    )
        donors.append(path.astype(np.float32))
    return runs, skipped


# ───────────────────────────── reporting ─────────────────────────────


def summarise(results: list[Result]) -> dict[str, float]:
    pnl = [r.pnl for r in results]
    daily: dict[str, float] = {}
    for r in results:
        daily[r.day] = daily.get(r.day, 0.0) + r.pnl
    d = list(daily.values())
    sd = statistics.pstdev(d) if len(d) > 1 else 0.0
    shares = sum(r.shares for r in results)
    return {
        "nights": len(results),
        "total_pnl": sum(pnl),
        "mean_pnl": statistics.fmean(pnl),
        "std_pnl": statistics.pstdev(pnl),
        "es5_night": cvar(pnl),
        "worst_night": min(pnl),
        "days": len(d),
        "mean_daily_pnl": statistics.fmean(d),
        "es5_daily": cvar(d),
        "sharpe_annual": statistics.fmean(d) / sd * math.sqrt(TRADING_DAYS) if sd else 0.0,
        "pm_pnl": sum(r.pm_pnl for r in results),
        "hedge_pnl": sum(r.hedge_pnl for r in results),
        "exit_cost": sum(r.exit_cost for r in results),
        "fills": sum(r.fills for r in results),
        "shares": shares,
        "pnl_per_share_cents": 100 * sum(pnl) / shares if shares else 0.0,
        "pm_pnl_per_share_cents": 100 * sum(r.pm_pnl for r in results) / shares if shares else 0.0,
        "mean_capture": statistics.fmean(r.capture for r in results),
        "mean_max_inventory": statistics.fmean(r.max_inventory for r in results),
    }


def build_report(runs: dict[str, list[Result]], skipped: dict[str, int]) -> dict[str, Any]:
    names = [n for n in runs if not n.startswith("placebo")]
    report: dict[str, Any] = {
        "markets": len(runs["pm_mid"]),
        "skipped": skipped,
        "assumptions": {
            "quotes": f"{QUOTE_SHARES:.0f} shares a side, ±{HALF_SPREAD:.2f} around the centre, "
            f"inventory cap {MAX_INVENTORY:.0f} shares, skew {SKEW:.5f}/share, band {PRICE_BAND}, FV quotes "
            f"lean at most to Polymarket mid ± {MAKER_LEAN}",
            "fills": "strictly-through taker orders only (last in queue at our price), at our price, "
            "<= quote size per order; quotes recomputed per order at order time - latency",
            "latency_ms": LATENCY_MS,
            "fees": f"Polymarket maker 0, taker shares x {PM_TAKER_RATE} x p(1-p) (exit only), maker rebate "
            f"not credited; perp {TAKER_FEE:.4f} taker on the aggTrades tape walk",
            "early_exit": f"'1h': {EXIT_MIN} min before resolution; 'open': at 09:{30 - FREEZE_MIN} ET; at "
            f"Polymarket mid ∓ {EXIT_CUSHION} + taker fee; hedges flattened at the same time",
            "breaker": f"pull quotes while max(ask/bid, bid/ask) ±1% depth > {TOX_TRIP} on the last "
            f"completed minute; re-arm after {TOX_REARM_MIN} minutes < {TOX_CLEAR}; no depth = no quotes",
            "hedge": f"hedger's position_delta_usd (digital delta, zero in the final "
            f"{HedgerConfig().no_hedge_final_secs:.0f} s), deadband ${HEDGE_DEADBAND_USD:.0f}, per minute",
            "fail_closed": f"no quote without a perp print in the last {PERP_STALE_S} s or a Polymarket mid "
            "within 5 min; no hedge without a perp bar or tape",
            "placebo": f"{PLACEBO_SEEDS} seeds per system: quote off another night's perp path "
            f"(previous {DONOR_NIGHTS} nights of the symbol, aligned on time to the close)",
            "offhours_sword": "added after a per-session calibration diagnostic showed the perp fair value "
            "beats Polymarket only while the stock is closed; judged by placebo and split-half, not by the "
            "in-sample total alone",
        },
        "strategies": {n: summarise(runs[n]) for n in names},
    }
    days = sorted({r.day for r in runs["pm_mid"]})
    cut = days[len(days) // 2]
    report["split_half_total_pnl"] = {
        "cut": cut,
        **{
            n: [sum(r.pnl for r in runs[n] if r.day < cut), sum(r.pnl for r in runs[n] if r.day >= cut)]
            for n in names
        },
    }
    report["placebo"] = {}
    keys = ("total_pnl", "sharpe_annual", "es5_night", "pnl_per_share_cents", "pm_pnl_per_share_cents")
    for system in PLACEBO_SYSTEMS:
        placebo = [summarise(runs[n]) for n in runs if n.startswith(f"placebo_{system.name}_")]
        if not placebo:
            continue
        real = report["strategies"][f"{system.name}_placebo_nights"]
        report["placebo"][system.name] = {
            "seeds": len(placebo),
            "nights": placebo[0]["nights"],
            **{k: {"min": min(p[k] for p in placebo), "max": max(p[k] for p in placebo)} for k in keys},
            "seeds_at_least_as_good": {
                "total_pnl": sum(p["total_pnl"] >= real["total_pnl"] for p in placebo),
                "sharpe_annual": sum(p["sharpe_annual"] >= real["sharpe_annual"] for p in placebo),
            },
        }
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"\n{report['markets']} market-nights; skipped {report['skipped']}")
    cols = ("total_pnl", "mean_pnl", "es5_night", "sharpe_annual", "hedge_pnl", "exit_cost", "shares")
    print(f"{'strategy':<30}" + "".join(f"{c:>14}" for c in cols) + f"{'¢/share':>10}{'capture':>9}")
    for name, s in report["strategies"].items():
        print(
            f"{name:<30}"
            + "".join(f"{s[c]:>14,.1f}" for c in cols)
            + f"{s['pnl_per_share_cents']:>10.2f}{s['mean_capture']:>9.3f}"
        )
    for name, p in report["placebo"].items():
        print(f"placebo {name}:", json.dumps(p, default=float))
    print("split half:", json.dumps(report["split_half_total_pnl"], default=float))


def charts(report: dict[str, Any], runs: dict[str, list[Result]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    lines = (
        ("pm_mid", "naive: quote Polymarket's own mid", "#9e9e9e"),
        ("shield_sword", "Shield + Sword (session σ, hedged, exit 1 h early, breaker)", "#7e57c2"),
        ("fv_session", "Sword only, session σ", "#ef6c00"),
        ("fv_anchor", "Sword only, minutes-to-close σ", "#26a69a"),
    )
    for name, label, colour in lines:
        daily: dict[str, float] = {}
        for r in runs[name]:
            daily[r.day] = daily.get(r.day, 0.0) + r.pnl
        days = sorted(daily)
        ax.plot(
            np.array(days, dtype="datetime64[D]"),
            np.cumsum([daily[d] for d in days]),
            label=label,
            color=colour,
        )
    for system, colour in ((FULL, "#7e57c2"), (FV_ANCHOR, "#26a69a")):
        placebo = [n for n in runs if n.startswith(f"placebo_{system.name}_")]
        if not placebo:
            continue
        curves = []
        days = sorted({r.day for r in runs[f"{system.name}_placebo_nights"]})
        for n in placebo:
            daily = dict.fromkeys(days, 0.0)
            for r in runs[n]:
                daily[r.day] += r.pnl
            curves.append(np.cumsum([daily[d] for d in days]))
        c = np.array(curves)
        x = np.array(days, dtype="datetime64[D]")
        ax.fill_between(
            x,
            c.min(axis=0),
            c.max(axis=0),
            color=colour,
            alpha=0.15,
            label=f"{system.name}: 20 placebo seeds",
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative P&L, USD (all symbols)")
    ax.set_title(f"Polymarket equity Up/Down market making, {report['markets']} market-nights")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "confluence_cum_pnl.png", dpi=150)
    plt.close(fig)

    s = report["strategies"]
    lat = [LATENCY_MS, *SLOW_LATENCIES_MS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for system, colour in ((FULL, "#7e57c2"), (OFFHOURS, "#1565c0"), (FV_ANCHOR, "#26a69a")):
        names = [system.name, *(f"{system.name}_lat_{ms}" for ms in SLOW_LATENCIES_MS)]
        ax.plot(
            [ms / 1000 for ms in lat],
            [s[n]["pnl_per_share_cents"] for n in names],
            "-o",
            color=colour,
            label=system.name,
        )
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("quote update latency, seconds (log)")
    ax.set_ylabel("net P&L per filled share, ¢")
    ax.set_title("How much of the edge is speed")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "confluence_latency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    symbols = tuple(s for s in os.getenv("TRADFI_SYMBOLS", ",".join(EQUITY_SYMBOLS)).split(",") if s)
    runs: dict[str, list[Result]] = {}
    skipped: dict[str, int] = {}
    workers = min(len(symbols), max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {s: pool.submit(run_symbol, s) for s in symbols}
        for s, fut in futures.items():
            r, sk = fut.result()
            for name, res in r.items():
                runs.setdefault(name, []).extend(res)
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            print(f"  {s}: {len(r.get('pm_mid', []))} nights ({time.time() - t0:.0f}s)", flush=True)
    report = build_report(runs, skipped)
    HERE.joinpath("results_confluence.json").write_text(json.dumps(report, indent=2, default=float))
    print_report(report)
    charts(report, runs)


if __name__ == "__main__":
    main()
