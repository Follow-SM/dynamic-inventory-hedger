"""Backtest: gap risk in equity-pegged Polymarket inventory hedged with Binance TradFi perps.

Universe: every daily "<TICKER> Up or Down on <date>?" Polymarket market (Up = the official close
beats the previous official close) for the tokenized-equity perps Binance lists as
`TRADIFI_PERPETUAL`. Each book buys SHARES of Up (and, mirrored, of Down) at the previous official
close and holds to resolution, so every position carries the overnight (or weekend) closure and
the US open. Strike K = the previous official close, known at entry.

    ToxicityEvaluator (bands + hysteresis) -> position_delta_usd (digital delta) -> decide()

run on this repo's own code, minute by minute, on the same books and prices:

    unhedged              never trade the perp
    continuous            100% delta hedge at every level (the Part 2 finding: delta neutrality,
                          not signal timing, is what cut the tail)
    freeze                continuous, but no orders from 5 minutes before to 5 minutes after the
                          09:30 ET open (an Open-Window Freeze); re-arms at 09:35 and catches up
    *_session_sigma       the same, with the delta model's volatility input rescaled so that its
                          wall-clock variance to expiry equals a session-aware forecast (RTH /
                          pre / post / overnight / closed minute variances + the open jump, from
                          the previous 20 trading days). The repo's own model spreads variance
                          evenly over wall-clock time, which a weekend plus an opening auction
                          violates.
    gated                 repo defaults 0/50/100% by toxicity level (session-aware signals,
                          including the per-symbol book-imbalance percentile)
    gated_naive           the same with the crypto backtest's signals (global-V VPIN, naive z)
    placebo               `gated` with its level path circularly shifted (timing test)

Execution is priced on the Binance aggTrades tape, not minute closes:
    IOC       walks same-side taker prints from decision + latency until the order's notional is
              covered (a tape-walk proxy for the book); unfilled remainder at the last print +5 bps
    post-only fills at the touch proxy (last opposite-side print) and is assumed filled within the
              minute: queue position is ignored, so passive fills are an upper bound
    latency   0-1,000 ms from decision to the order reaching the book (equivalent to shifting
              every protection window with `Intervals.delayed`)
    fees      maker 2 bps, taker 5 bps; funding from Binance's funding history
Fail-closed: minutes with no perp bar or no tape trigger no new orders; existing hedges are kept.
Hysteresis: levels escalate at once and return to NORMAL only below the rebalance band
(vpin_percentile 0.60); the freeze re-arms at open + 5 min.

    pip install -e ".[tradfi]" && tmr tradfi fetch && python research/backtest_tradfi.py
    (TMR_TRADFI_ROOT points at the Parquet root if it isn't ../tick-microstructure-research/data/tradfi)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import random
import statistics
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
from tick_microstructure.research.data.fetch import EQUITY_SYMBOLS, ticker
from tick_microstructure.research.data.store import TickTrades, TradfiStore
from tick_microstructure.research.signals import Intervals
from tick_microstructure.research.tradfi import MIN_MS, MinuteFeatures, load_features
from tick_microstructure.research.utils.session import Session, SessionCalendar

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.execution.policy import decide
from dynamic_inventory_hedger.inventory.delta import MarketSpec, position_delta_usd, sigma_15m_from_natr
from dynamic_inventory_hedger.models import HedgeAction, OrderType, ToxicityLevel, ToxicityMetric
from dynamic_inventory_hedger.signals.evaluator import ToxicityEvaluator

F = npt.NDArray[np.float64]
SHARES = 1_000.0
MAKER_FEE, TAKER_FEE = 0.0002, 0.0005
THIN_TAPE_PENALTY = 0.0005
TAPE_WINDOW_MS = 60_000
MAX_ORDERS_PER_MINUTE = 30
FREEZE_MIN = 5
OPEN_WINDOW_MIN = 15  # execution-drag accounting window either side of the open
LATENCIES_MS = (0, 100, 250, 500, 1_000)
DEADBANDS_USD = (150, 500, 1_000, 2_500, 5_000)
PLACEBO_SEEDS = 10
VAR_LOOKBACK_DAYS = 20

HERE = Path(__file__).parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
TRADFI_ROOT = Path(
    os.getenv("TMR_TRADFI_ROOT", HERE.parent.parent / "tick-microstructure-research" / "data" / "tradfi")
)
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
START, END = date(2026, 1, 28), date(2026, 9, 28)
CAL = SessionCalendar()


# ───────────────────────────── Polymarket (cached under research/data/) ─────────────────────────────


def cached(name: str, fetch: Callable[[], Any]) -> Any:
    path = DATA / f"{name}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    value = fetch()
    DATA.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(value, fh)
    return value


async def _get_json(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, params: dict[str, Any]
) -> Any:
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError:
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code in (429,) or resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            return resp.raise_for_status().json()
    raise RuntimeError(f"giving up on {url} {params}")


def _iso_ms(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


async def _fetch_events(symbols: Sequence[str]) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(8)
    days = CAL.trading_days(START, END)

    async def one(sym: str, d: date) -> dict[str, Any] | None:
        slug = f"{ticker(sym).lower()}-up-or-down-on-{d:%B}-{d.day}-{d.year}".lower()
        events = await _get_json(client, sem, GAMMA_EVENTS, {"slug": slug})
        if not events or not events[0].get("markets"):
            return None
        m = events[0]["markets"][0]
        prices = json.loads(m.get("outcomePrices") or "[]")
        outcomes = json.loads(m.get("outcomes") or "[]")
        tokens = json.loads(m.get("clobTokenIds") or "[]")
        if outcomes[:2] != ["Up", "Down"] or len(tokens) < 2 or len(prices) < 2:
            return None
        return {
            "sym": sym,
            "day": d.isoformat(),
            "slug": slug,
            "up_token": tokens[0],
            "payout_up": float(prices[0]),
            "start_ms": _iso_ms(m.get("startDate") or events[0].get("startDate")),
            "created_ms": _iso_ms(m.get("createdAt") or events[0].get("createdAt")),
        }

    async with httpx.AsyncClient(timeout=30) as client:
        found = await asyncio.gather(*(one(s, d) for s in symbols for d in days))
    return [e for e in found if e]


async def _fetch_histories(events: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    sem = asyncio.Semaphore(8)

    async def one(e: dict[str, Any]) -> tuple[str, list[list[float]]]:
        d = date.fromisoformat(e["day"])
        start = CAL.close_ms(CAL.previous_trading_day(d)) // 1000 - 3_600
        params = {"market": e["up_token"], "startTs": start, "endTs": CAL.close_ms(d) // 1000, "fidelity": 1}
        body = await _get_json(client, sem, CLOB_HISTORY, params)
        return e["slug"], [[int(p["t"]), float(p["p"])] for p in body.get("history", [])]

    async with httpx.AsyncClient(timeout=30) as client:
        return dict(await asyncio.gather(*(one(e) for e in events)))


# ───────────────────────────── signals and volatility ─────────────────────────────


def _nan_to(x: float, default: float) -> float:
    return default if not math.isfinite(x) else x


def level_path(f: MinuteFeatures, sym: str, naive: bool) -> list[ToxicityLevel]:
    evaluator = ToxicityEvaluator(HedgerConfig())
    pctl = f.pctl_global if naive else f.pctl_session
    vpin = f.vpin_global if naive else f.vpin_session
    vol_z = f.vol_z if naive else f.vol_z_session
    obt = f.obt
    levels = []
    for i in range(f.t.size):
        p, ob = float(pctl[i]), float(f.ob_pctl[i])
        metric = ToxicityMetric.model_construct(
            symbol=sym,
            timestamp_ms=int(f.t[i]),
            price=float(f.close[i]),
            vpin=_nan_to(float(vpin[i]), 0.0),
            vpin_percentile=p if math.isfinite(p) else None,
            ob_imbalance_l1=0.5,
            depth_imbalance={},
            ob_toxicity_1pct=_nan_to(float(obt[i]), 0.0),
            ob_imbalance_percentile=None if naive or not math.isfinite(ob) else ob,
            volume_z_score=_nan_to(float(vol_z[i]), 0.0),
            natr_15m=_nan_to(float(f.natr_15m[i]), 0.0),
            price_delta_15m_pct=_nan_to(float(f.move15[i]), 0.0),
            whale_sweeps_1h_usdt=0.0,
            order_flow_by_token={},
        )
        levels.append(evaluator.evaluate(metric).level)
    return levels


@dataclass
class VarianceModel:
    per_minute: dict[int, float]  # Session -> variance of a 1m log return, open window excluded
    open_jump: float  # variance of the 09:25 -> 09:35 log return


def variance_models(f: MinuteFeatures) -> dict[date, VarianceModel]:
    """Session variance forecast per trading day from the previous VAR_LOOKBACK_DAYS trading days."""
    r = np.concatenate([[np.nan], np.diff(np.log(f.close))])
    r[f.missing] = np.nan
    mfo = CAL.minutes_from_open(f.t)
    in_open = np.abs(mfo + 0.5) <= FREEZE_MIN  # minutes 09:25 .. 09:34
    first, last = CAL.local_dates(f.t[:1])[0], CAL.local_dates(f.t[-1:])[0]
    days = CAL.trading_days(first, last)
    local_day = np.array([d.toordinal() for d in CAL.local_dates(f.t)])
    jumps: list[float] = []
    per_day: list[dict[int, tuple[float, int]]] = []
    for d in days:
        o = CAL.open_ms(d)
        # closes of the 09:24 and 09:34 bars: the 09:25 -> 09:35 return
        a, b = f.at(
            np.array([o - (FREEZE_MIN + 1) * MIN_MS, o + (FREEZE_MIN - 1) * MIN_MS], dtype=np.float64)
        )
        jumps.append(float(np.log(f.close[b] / f.close[a]) ** 2) if a >= 0 and b >= 0 else np.nan)
        m = (local_day == d.toordinal()) & ~in_open & np.isfinite(r)
        per_day.append(
            {
                int(s): (float(np.sum(r[m & (f.session == s)] ** 2)), int(np.sum(m & (f.session == s))))
                for s in Session
            }
        )
    out: dict[date, VarianceModel] = {}
    for k in range(VAR_LOOKBACK_DAYS, len(days)):
        window = per_day[k - VAR_LOOKBACK_DAYS : k]
        pm = {}
        for s in Session:
            ss = sum(w[int(s)][0] for w in window)
            n = sum(w[int(s)][1] for w in window)
            pm[int(s)] = ss / n if n else 0.0
        j = [x for x in jumps[k - VAR_LOOKBACK_DAYS : k] if math.isfinite(x)]
        out[days[k]] = VarianceModel(pm, statistics.fmean(j) if j else 0.0)
    return out


# ───────────────────────────── books ─────────────────────────────


@dataclass
class Book:
    sym: str
    slug: str
    day: date
    holds_up: bool
    idx: npt.NDArray[np.int64]  # decision minutes (grid index; decision at the bar's end)
    t_dec: F  # decision times, ms
    perp: F
    missing: npt.NDArray[np.bool_]
    poly: F  # outcome-token mark
    delta_wall: F
    delta_session: F
    funding: F
    frozen: npt.NDArray[np.bool_]
    open_window: npt.NDArray[np.bool_]
    payout: float
    closure_h: float


def _poly_marks(history: list[list[float]], t_dec: F) -> F:
    ht = np.array([h[0] * 1000.0 for h in history])
    hp = np.array([h[1] for h in history])
    j = np.searchsorted(ht, t_dec, side="right") - 1
    return np.where(j >= 0, hp[np.maximum(j, 0)], np.nan)


def build_books(
    f: MinuteFeatures,
    sym: str,
    events: list[dict[str, Any]],
    histories: dict[str, list[list[float]]],
    funding: tuple[F, F],
    skipped: dict[str, int],
) -> list[Book]:
    store = TradfiStore(TRADFI_ROOT)
    daily = store.yahoo_daily(ticker(sym))
    close_by_day = {d: float(c) for d, c in zip(daily.days, daily.close, strict=True)}
    variances = variance_models(f)
    config = HedgerConfig()
    ft, fr = funding
    books: list[Book] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for e in events:
        d = date.fromisoformat(e["day"])
        prev = CAL.previous_trading_day(d)
        entry, expiry = CAL.close_ms(prev), CAL.close_ms(d)
        if e["payout_up"] not in (0.0, 1.0):
            skip("unresolved or 50-50")
            continue
        if prev not in close_by_day or d not in close_by_day:
            skip("no official close")
            continue
        if (close_by_day[d] > close_by_day[prev]) != (e["payout_up"] == 1.0):
            skip("resolution disagrees with Yahoo closes")
            continue
        listed = e.get("start_ms") or e.get("created_ms")
        if listed is None or listed > entry:
            skip("listed after the previous close")
            continue
        if d not in variances:
            skip("variance warm-up")
            continue
        i0, i1 = f.at(np.array([entry, expiry - MIN_MS], dtype=np.float64))
        if i0 < 0 or i1 < 0 or not math.isfinite(f.pctl_session[i0]):
            skip("perp data / signal warm-up")
            continue
        idx = np.arange(i0, i1 + 1)
        t_dec = f.t[idx] + MIN_MS
        history = histories.get(e["slug"]) or []
        if len(history) < 60:
            skip("thin Polymarket history")
            continue
        up = _poly_marks(history, t_dec)
        if not math.isfinite(up[0]) or not 0.05 <= up[0] <= 0.95:
            skip("no Polymarket mark at entry")
            continue
        up = np.where(np.isfinite(up), up, up[0])
        strike = close_by_day[prev]
        basis = float(f.close[i0]) / strike - 1  # perp vs official close at entry, known then
        spot = f.close[idx] / (1 + basis)
        spec = MarketSpec(e["slug"], "above", sym, expiry / 1000, strike=strike)
        vm = variances[d]
        var = np.array([vm.per_minute[int(s)] for s in f.session[idx]])
        mfo = CAL.minutes_from_open(f.t[idx])
        in_open = np.abs(mfo + 0.5) <= FREEZE_MIN
        var[in_open] = vm.open_jump / (2 * FREEZE_MIN)
        remaining = np.concatenate([np.cumsum(var[::-1])[::-1][1:], [0.0]])  # variance after each decision
        tau = (expiry - t_dec) / 1000
        dw, ds = np.zeros(idx.size), np.zeros(idx.size)
        for k in range(idx.size):
            sig_wall = sigma_15m_from_natr(_nan_to(float(f.natr_15m[idx[k]]), 0.0), config.min_sigma_15m)
            sig_sess = (
                math.sqrt(remaining[k] * 900 / tau[k])
                if tau[k] > 0 and remaining[k] > 0
                else config.min_sigma_15m
            )
            for arr, sig in ((dw, sig_wall), (ds, max(sig_sess, config.min_sigma_15m))):
                arr[k] = position_delta_usd(
                    spec,
                    True,
                    SHARES,
                    float(spot[k]),
                    sig,
                    t_dec[k] / 1000,
                    no_hedge_final_secs=config.no_hedge_final_secs,
                    max_abs_delta_usd=config.max_delta_per_position_usd,
                )
        j = np.searchsorted(ft, t_dec - MIN_MS, side="left")
        j2 = np.searchsorted(ft, t_dec, side="left")
        fund = np.array([fr[a:b].sum() for a, b in zip(j.tolist(), j2.tolist(), strict=True)])
        opens = np.array([CAL.open_ms(d)], dtype=np.float64)
        freeze = Intervals(opens - FREEZE_MIN * MIN_MS, opens + FREEZE_MIN * MIN_MS - 1)  # 09:25 <= t < 09:35
        drag_window = Intervals(opens - OPEN_WINDOW_MIN * MIN_MS, opens + OPEN_WINDOW_MIN * MIN_MS - 1)
        for holds_up in (True, False):
            books.append(
                Book(
                    sym=sym,
                    slug=e["slug"],
                    day=d,
                    holds_up=holds_up,
                    idx=idx,
                    t_dec=t_dec,
                    perp=f.close[idx],
                    missing=f.missing[idx],
                    poly=up if holds_up else 1 - up,
                    delta_wall=dw if holds_up else -dw,
                    delta_session=ds if holds_up else -ds,
                    funding=fund,
                    frozen=freeze.contains(t_dec),
                    open_window=drag_window.contains(t_dec),
                    payout=e["payout_up"] if holds_up else 1 - e["payout_up"],
                    closure_h=CAL.closure_hours_before(d),
                )
            )
    return books


# ───────────────────────────── execution on the tape ─────────────────────────────


@dataclass
class SideTape:
    t: F
    px: F
    cum_notional: F  # inclusive cumulative notional
    cum_px_notional: F


@dataclass
class Tape:
    buys: SideTape  # taker buys: the ask side we would lift
    sells: SideTape  # taker sells: the bid side we would hit

    @classmethod
    def from_trades(cls, tr: TickTrades) -> Tape:
        def side(mask: npt.NDArray[np.bool_]) -> SideTape:
            n = tr.notional[mask]
            return SideTape(tr.t[mask], tr.price[mask], np.cumsum(n), np.cumsum(n * tr.price[mask]))

        return cls(side(tr.side > 0), side(tr.side < 0))


class TapeCache:
    def __init__(self, sym: str, size: int = 5) -> None:
        self.store = TradfiStore(TRADFI_ROOT)
        self.sym = sym
        self.size = size
        self._cache: OrderedDict[tuple[date, date], Tape] = OrderedDict()

    def get(self, start: date, end: date) -> Tape:
        key = (start, end)
        if key not in self._cache:
            self._cache[key] = Tape.from_trades(self.store.trades(self.sym, start, end))
            if len(self._cache) > self.size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]


def touch(tape: Tape, t: float, buy: bool) -> float:
    """Passive price proxy: the last print on the side we would rest on (bid for a buy)."""
    s = tape.sells if buy else tape.buys
    j = int(np.searchsorted(s.t, t, side="right")) - 1
    return float(s.px[j]) if j >= 0 and t - s.t[j] <= TAPE_WINDOW_MS else float("nan")


def walk(tape: Tape, t: float, buy: bool, notional: float) -> tuple[float, bool]:
    """(average fill price, thin) for an IOC of `notional` arriving at `t`."""
    s = tape.buys if buy else tape.sells
    k0 = int(np.searchsorted(s.t, t, side="left"))
    k_end = int(np.searchsorted(s.t, t + TAPE_WINDOW_MS, side="right"))
    if k0 >= k_end:
        return float("nan"), True
    base_n = s.cum_notional[k0 - 1] if k0 > 0 else 0.0
    base_pn = s.cum_px_notional[k0 - 1] if k0 > 0 else 0.0
    k1 = int(np.searchsorted(s.cum_notional, base_n + notional, side="left"))
    if k1 < k_end:  # prints k0 .. k1 cover the order; k1 only partly
        prev_n = s.cum_notional[k1 - 1] if k1 > 0 else 0.0
        prev_pn = s.cum_px_notional[k1 - 1] if k1 > 0 else 0.0
        pn = prev_pn - base_pn + (notional - (prev_n - base_n)) * s.px[k1]
        return float(pn / notional), False
    got = s.cum_notional[k_end - 1] - base_n
    last = float(s.px[k_end - 1]) * (1 + (THIN_TAPE_PENALTY if buy else -THIN_TAPE_PENALTY))
    return float((s.cum_px_notional[k_end - 1] - base_pn + (notional - got) * last) / notional), True


# ───────────────────────────── simulation ─────────────────────────────


@dataclass(frozen=True)
class Strategy:
    name: str
    levels: str | None  # None = unhedged; "normal" | "gated" | "gated_naive" | "placebo<k>"
    config_key: str = "always"  # "always" | "default"
    sigma: str = "wall"  # "wall" | "session"
    freeze: bool = False
    latency_ms: int = 0
    deadband_usd: float | None = None


@dataclass
class Result:
    pnl: float
    max_dd: float
    fees: float
    funding: float
    slippage: float  # signed cost vs the decision-time touch mid (positive = paid)
    open_window_cost: float  # fees + slippage on orders within +-15 min of the open
    traded_usd: float
    thin_fills: int
    closure_h: float
    day: str
    sym: str


def simulate(
    book: Book, tape: Tape, levels: Sequence[ToxicityLevel] | None, config: HedgerConfig, s: Strategy
) -> Result:
    entry = float(book.poly[0])
    delta = book.delta_session if s.sigma == "session" else book.delta_wall
    contracts = cash = fees = funding = slip = open_cost = traded = 0.0
    thin = 0
    peak = max_dd = 0.0
    for k in range(book.idx.size):
        mark = float(book.perp[k])
        can_trade = levels is not None and not book.missing[k] and not (s.freeze and book.frozen[k])
        if can_trade:
            assert levels is not None
            t_ref = float(book.t_dec[k])
            t = t_ref + s.latency_ms
            bid_ref, ask_ref = touch(tape, t_ref, True), touch(tape, t_ref, False)
            mid = (bid_ref + ask_ref) / 2 if math.isfinite(bid_ref + ask_ref) else mark
            for _ in range(MAX_ORDERS_PER_MINUTE):
                signal = decide(book.sym, levels[k], float(delta[k]), contracts * mark, config)
                if signal is None:
                    break
                buy = signal.action is HedgeAction.BUY
                taker = signal.order_type is OrderType.IOC_LIMIT
                if taker:
                    price, was_thin = walk(tape, t, buy, signal.notional_usd)
                    thin += was_thin
                else:
                    price = touch(tape, t, buy)
                if not math.isfinite(price):
                    break  # fail closed: no tape to price the order against
                qty = signal.notional_usd / price * (1 if buy else -1)
                fee = signal.notional_usd * (TAKER_FEE if taker else MAKER_FEE)
                cost = qty * (price - mid)
                contracts += qty
                cash -= qty * price
                fees += fee
                slip += cost
                traded += signal.notional_usd
                if book.open_window[k]:
                    open_cost += fee + cost
        funding += contracts * mark * float(book.funding[k])
        mtm = SHARES * (float(book.poly[k]) - entry) + cash + contracts * mark - fees - funding
        peak = max(peak, mtm)
        max_dd = max(max_dd, peak - mtm)
    if contracts:
        t = float(book.t_dec[-1]) + s.latency_ms
        buy = contracts < 0
        notional = abs(contracts) * float(book.perp[-1])
        price, was_thin = walk(tape, t, buy, notional)
        if not math.isfinite(price):
            price = float(book.perp[-1]) * (1 + (THIN_TAPE_PENALTY if buy else -THIN_TAPE_PENALTY))
            was_thin = True
        thin += was_thin
        cash += contracts * price
        fees += notional * TAKER_FEE
        slip += -contracts * (price - float(book.perp[-1]))
        traded += notional
    pnl = SHARES * (book.payout - entry) + cash - fees - funding
    peak = max(peak, pnl)
    max_dd = max(max_dd, peak - pnl)
    return Result(
        pnl,
        max_dd,
        fees,
        funding,
        slip,
        open_cost,
        traded,
        thin,
        book.closure_h,
        book.day.isoformat(),
        book.sym,
    )


def strategies() -> list[Strategy]:
    out = [
        Strategy("unhedged", None),
        Strategy("continuous", "normal"),
        Strategy("continuous_session_sigma", "normal", sigma="session"),
        Strategy("freeze", "normal", freeze=True),
        Strategy("freeze_session_sigma", "normal", sigma="session", freeze=True),
        Strategy("gated", "gated", config_key="default"),
        Strategy("gated_naive", "gated_naive", config_key="default"),
    ]
    out += [Strategy(f"placebo{k}", f"placebo{k}", config_key="default") for k in range(PLACEBO_SEEDS)]
    out += [Strategy(f"continuous_latency_{ms}", "normal", latency_ms=ms) for ms in LATENCIES_MS[1:]]
    out += [Strategy(f"freeze_latency_{ms}", "normal", freeze=True, latency_ms=ms) for ms in LATENCIES_MS[1:]]
    out += [Strategy(f"continuous_deadband_{b}", "normal", deadband_usd=b) for b in DEADBANDS_USD[1:]]
    out += [
        Strategy(f"freeze_deadband_{b}", "normal", freeze=True, deadband_usd=b) for b in DEADBANDS_USD[1:]
    ]
    return out


def run_symbol(
    sym: str, events: list[dict[str, Any]], histories: dict[str, list[list[float]]]
) -> tuple[dict[str, list[Result]], dict[str, int], dict[str, float]]:
    store = TradfiStore(TRADFI_ROOT)
    f = load_features(store, CAL, sym)
    skipped: dict[str, int] = {}
    books = build_books(f, sym, events, histories, store.funding(sym), skipped)
    if not books:
        return {}, skipped, {}
    paths = {"normal": [ToxicityLevel.NORMAL] * f.t.size, "gated": level_path(f, sym, False)}
    paths["gated_naive"] = level_path(f, sym, True)
    warm = int(np.argmax(np.isfinite(f.pctl_session)))
    span = f.t.size - warm
    rng = random.Random(f"{sym}-placebo")
    for k in range(PLACEBO_SEEDS):
        shift = rng.randrange(7 * 1440, max(7 * 1440 + 1, span - 7 * 1440))
        g = paths["gated"]
        paths[f"placebo{k}"] = g[:warm] + [g[warm + (i - warm + shift) % span] for i in range(warm, f.t.size)]
    configs = {
        "always": HedgerConfig(baseline_hedge_ratio=1.0, pre_hedge_ratio=1.0, emergency_hedge_ratio=1.0),
        "default": HedgerConfig(),
    }
    tapes = TapeCache(sym)
    runs: dict[str, list[Result]] = {}
    for book in books:
        first = CAL.local_dates(book.t_dec[:1] - MIN_MS)[0]
        tape = tapes.get(min(first, book.day), book.day)
        for s in strategies():
            config = configs[s.config_key]
            if s.deadband_usd is not None:
                config = config.model_copy(update={"min_rebalance_usd": s.deadband_usd})
            levels = None if s.levels is None else [paths[s.levels][i] for i in book.idx]
            runs.setdefault(s.name, []).append(simulate(book, tape, levels, config, s))
    level_share = {
        lv.value: float(np.mean([paths["gated"][i] is lv for b in books[::2] for i in b.idx]))
        for lv in ToxicityLevel
    }
    return runs, skipped, level_share


# ───────────────────────────── reporting ─────────────────────────────


def cvar(values: Sequence[float], q: float = 0.05) -> float:
    worst = sorted(values)[: max(1, int(len(values) * q))]
    return statistics.fmean(worst)


def summarise(results: Sequence[Result]) -> dict[str, float]:
    pnl = [r.pnl for r in results]
    dd = sorted(r.max_dd for r in results)
    return {
        "n": len(results),
        "mean_pnl": statistics.fmean(pnl),
        "std_pnl": statistics.pstdev(pnl),
        "cvar5": cvar(pnl),
        "worst": min(pnl),
        "mean_max_dd": statistics.fmean(dd),
        "p95_max_dd": dd[int(0.95 * (len(dd) - 1))],
        "mean_costs": statistics.fmean(r.fees + r.funding + r.slippage for r in results),
        "mean_fees": statistics.fmean(r.fees for r in results),
        "mean_slippage": statistics.fmean(r.slippage for r in results),
        "mean_open_window_cost": statistics.fmean(r.open_window_cost for r in results),
        "mean_traded_usd": statistics.fmean(r.traded_usd for r in results),
        "thin_fills_per_book": statistics.fmean(r.thin_fills for r in results),
    }


def main() -> None:
    t0 = time.time()
    symbols = tuple(s for s in os.getenv("TRADFI_SYMBOLS", ",".join(EQUITY_SYMBOLS)).split(",") if s)
    tag = f"{START:%Y%m%d}_{END:%Y%m%d}"
    events = cached(f"tradfi_events_{tag}", lambda: asyncio.run(_fetch_events(symbols)))
    print(f"{len(events)} Polymarket Up/Down markets", flush=True)
    histories = cached(f"tradfi_histories_{tag}", lambda: asyncio.run(_fetch_histories(events)))
    by_sym = {s: [e for e in events if e["sym"] == s] for s in symbols}
    runs: dict[str, list[Result]] = {}
    skipped: dict[str, int] = {}
    level_share: dict[str, dict[str, float]] = {}
    workers = min(len(symbols), max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            s: pool.submit(
                run_symbol, s, by_sym[s], {e["slug"]: histories.get(e["slug"], []) for e in by_sym[s]}
            )
            for s in symbols
            if by_sym[s]
        }
        for s, fut in futures.items():
            r, sk, ls = fut.result()
            for name, res in r.items():
                runs.setdefault(name, []).extend(res)
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            if ls:
                level_share[s] = ls
            print(f"  {s}: {len(r.get('unhedged', [])) // 2} markets ({time.time() - t0:.0f}s)", flush=True)
    report = build_report(runs, skipped, level_share)
    HERE.joinpath("results_tradfi.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    charts(report, runs)


def build_report(
    runs: dict[str, list[Result]], skipped: dict[str, int], level_share: dict[str, dict[str, float]]
) -> dict[str, Any]:
    main_names = [n for n in runs if not n.startswith("placebo")]
    placebo = [summarise(runs[n]) for n in runs if n.startswith("placebo")]
    gated = summarise(runs["gated"])
    weekend = {
        n: summarise([r for r in runs[n] if r.closure_h > 30])
        for n in ("unhedged", "continuous", "freeze", "continuous_session_sigma", "freeze_session_sigma")
        if any(r.closure_h > 30 for r in runs[n])
    }
    return {
        "period": f"{START} -> {END}",
        "markets": len(runs["unhedged"]) // 2,
        "books": len(runs["unhedged"]),
        "shares": SHARES,
        "skipped": skipped,
        "assumptions": {
            "entry": "previous official close (16:00 ET), strike = that close; held to the official close",
            "fills": "IOC walks same-side aggTrades prints from decision + latency (60 s window, then "
            "+5 bps); post-only fills at the last opposite-side print within the minute (queue ignored: "
            "upper bound)",
            "fees": {"maker": MAKER_FEE, "taker": TAKER_FEE},
            "fail_closed": "no perp bar or no tape -> no new orders that minute; existing hedge kept",
            "hysteresis": "evaluator: immediate escalation, NORMAL again only below vpin_percentile 0.60; "
            f"freeze: no orders {FREEZE_MIN} min either side of 09:30 ET, re-arms at 09:35 and catches up",
            "latency": "applied to every fill (equivalent to Intervals.delayed on the protection windows)",
            "delta": "repo model (wall-clock sigma from natr_15m); *_session_sigma rescales sigma so the "
            f"remaining variance matches a {VAR_LOOKBACK_DAYS}-day session forecast incl. the open jump",
            "spot": "perp price divided by (1 + perp/official-close basis at entry)",
        },
        "strategies": {n: summarise(runs[n]) for n in main_names},
        "after_weekend_or_holiday": weekend,
        "placebo": {
            key: {
                "mean": statistics.fmean(p[key] for p in placebo),
                "min": min(p[key] for p in placebo),
                "max": max(p[key] for p in placebo),
            }
            for key in placebo[0]
        }
        if placebo
        else {},
        "placebo_at_least_as_good_as_gated": {
            "cvar5": sum(p["cvar5"] >= gated["cvar5"] for p in placebo) / len(placebo),
            "std_pnl": sum(p["std_pnl"] <= gated["std_pnl"] for p in placebo) / len(placebo),
        }
        if placebo
        else {},
        "gated_level_share": level_share,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n{report['period']}: {report['markets']} markets x 2 sides; skipped {report['skipped']}")
    cols = [
        "mean_pnl",
        "std_pnl",
        "cvar5",
        "p95_max_dd",
        "mean_costs",
        "mean_slippage",
        "mean_open_window_cost",
        "mean_traded_usd",
    ]
    print(f"{'strategy':<30}" + "".join(f"{c:>14}" for c in cols))
    for name, s in report["strategies"].items():
        print(f"{name:<30}" + "".join(f"{s[c]:>14,.1f}" for c in cols))
    if report["placebo"]:
        p = report["placebo"]
        print(f"{'placebo avg':<30}" + "".join(f"{p[c]['mean']:>14,.1f}" for c in cols))
        print("placebo seeds at least as good as gated:", report["placebo_at_least_as_good_as_gated"])
    print("after a weekend or holiday:")
    for name, s in report["after_weekend_or_holiday"].items():
        print(f"  {name:<28}" + "".join(f"{s[c]:>14,.1f}" for c in cols))


def charts(report: dict[str, Any], runs: dict[str, list[Result]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(exist_ok=True)
    s = report["strategies"]
    names = ["unhedged", "continuous", "freeze", "continuous_session_sigma", "freeze_session_sigma"]
    labels = ["unhedged", "continuous", "open-window\nfreeze", "continuous\nsession σ", "freeze\nsession σ"]
    colours = ["#9e9e9e", "#26a69a", "#7e57c2", "#80cbc4", "#b39ddb"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=(3, 2))
    left.boxplot([[r.pnl for r in runs[n]] for n in names], tick_labels=labels, whis=(5, 95), showfliers=True)
    left.axhline(0, color="black", lw=0.8)
    left.set_ylabel(f"P&L per position, USD ({SHARES:.0f} shares, prior close -> close)")
    left.set_title("Per-position P&L (whiskers 5th-95th percentile)")
    left.grid(alpha=0.3, axis="y")
    x = np.arange(len(names))
    right.bar(x - 0.2, [-s[n]["cvar5"] for n in names], 0.4, color=colours, label="expected shortfall 5%")
    right.bar(
        x + 0.2,
        [s[n]["p95_max_dd"] for n in names],
        0.4,
        color=colours,
        alpha=0.45,
        hatch="//",
        label="p95 max drawdown",
    )
    right.set_xticks(x, labels, fontsize=8)
    right.set_ylabel("USD per position (lower is better)")
    right.set_title("Tail loss and drawdown")
    right.legend(fontsize=8)
    right.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"{report['markets']} equity Up/Down markets hedged with Binance TradFi perps, {report['period']}"
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "tradfi_fig3_pnl_drawdown.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for n, lab, c in zip(names, labels, colours, strict=True):
        ax.scatter(s[n]["mean_costs"], -s[n]["cvar5"], s=110, color=c, label=lab.replace("\n", " "), zorder=3)
    for prefix, c, lab in (("continuous", "#26a69a", "continuous"), ("freeze", "#7e57c2", "freeze")):
        pts = [s[prefix]] + [s[f"{prefix}_deadband_{b}"] for b in DEADBANDS_USD[1:]]
        ax.plot(
            [p["mean_costs"] for p in pts],
            [-p["cvar5"] for p in pts],
            "-D",
            color=c,
            ms=4,
            lw=1,
            label=f"{lab}, deadband $150 -> $5,000",
        )
        lat = [s[prefix]] + [s[f"{prefix}_latency_{ms}"] for ms in LATENCIES_MS[1:]]
        ax.plot(
            [p["mean_costs"] for p in lat],
            [-p["cvar5"] for p in lat],
            ":o",
            color=c,
            ms=3,
            lw=1,
            label=f"{lab}, latency 0 -> 1,000 ms",
        )
    g = s["gated"]
    ax.scatter(
        g["mean_costs"],
        -g["cvar5"],
        s=90,
        marker="s",
        color="#ef6c00",
        label="gated (toxicity levels)",
        zorder=3,
    )
    ax.set_xlabel("mean hedging cost per position, USD (fees + funding + slippage vs touch mid)")
    ax.set_ylabel("expected shortfall 5%, USD (lower is better)")
    ax.set_title("Tail risk vs execution cost: deadband and latency sweeps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "tradfi_risk_vs_cost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
