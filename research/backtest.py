"""Backtest: does toxicity-gated hedging cut the tail risk of Polymarket BTC inventory?

Universe: every daily "Will the price of Bitcoin be above $K on <date>?" event from START to END.
At expiry - 24h the book holds SHARES of the strike nearest spot, once long YES and once long NO
(mirrored books; the number of independent samples is the number of markets). Positions are held
to resolution and replayed minute by minute with this repo's own code paths:

    ToxicityEvaluator (bands + hysteresis) -> position_delta_usd (digital-option delta) -> decide()

Strategies, same book, same prices:
    unhedged     never trade the perp
    limit_only   hedge only what exceeds MAX_INVENTORY_USD (all levels forced to NORMAL)
    gated        repo defaults: 0% / 50% / 100% by toxicity level, plus the inventory limit
    always       100% delta hedge at every level
    placebo      `gated` with its level path shifted by a random offset in time (same share of
                 time hedged, same regime lengths, wrong timing): does the timing matter?

Signals are this study's own open reimplementation from Binance public 1m spot klines, which carry
exact taker-buy notional. They are NOT FollowSM's production calibration, and three inputs have no
public history at all (1% book toxicity, smart-money sweeps, Polymarket book flow), so only the
VPIN-percentile and liquidity-sweep triggers are exercised here.
    VPIN            $1M notional buckets, 50-bucket window
    vpin_percentile rank of the current VPIN among the previous 7 days of minute samples
    natr_15m        ATR(14) of 15m candles / close
    volume_z_score  last-15m quote volume vs the previous 96 15m candles
Execution: Binance USD-M BTCUSDT perp minute close; maker 2 bps (post-only, assumed filled within
the minute: optimistic), taker 5 bps + 1 bp slippage (IOC); real funding payments. Polymarket
marks from CLOB minute history; settlement at the official resolution.

    pip install -e ".[research]" && python research/backtest.py
"""

from __future__ import annotations

import bisect
import gzip
import json
import math
import os
import random
import statistics
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.execution.policy import decide
from dynamic_inventory_hedger.inventory.delta import (
    MarketSpec,
    classify_question,
    position_delta_usd,
    sigma_15m_from_natr,
)
from dynamic_inventory_hedger.models import HedgeAction, OrderType, ToxicityLevel, ToxicityMetric
from dynamic_inventory_hedger.signals.evaluator import ToxicityEvaluator

START = datetime(2026, 1, 2, tzinfo=UTC)  # first expiry date
END = datetime(2026, 9, 28, tzinfo=UTC)  # last expiry date (inclusive)
WARMUP = timedelta(days=9)  # 7d percentile history + 24h volume baseline + slack
SHARES = 1_000.0
HOLD_SECS = 86_400
VPIN_BUCKET_USD = 1_000_000.0
VPIN_BUCKETS = 50
PCTL_WINDOW_MIN = 7 * 1440
MAKER_FEE, TAKER_FEE, SLIPPAGE = 0.0002, 0.0005, 0.0001
MAX_ORDERS_PER_MINUTE = 30  # the live engine re-decides every 2 s
PLACEBO_SEEDS = 20
BUCKET_MODE = os.getenv("VPIN_BUCKET", "fixed")  # fixed = $1M buckets; adv = trailing 7d ADV / 50
EXIT_EARLY_MIN = 60  # diagnostic: close the whole book this long before resolution
DEADBANDS_USD = (150, 500, 1_000, 2_500, 5_000)

HERE = Path(__file__).parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"
PERP_KLINES = "https://fapi.binance.com/fapi/v1/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"

MS = 1000
MIN_MS = 60_000


# ───────────────────────────── data (cached under research/data/) ─────────────────────────────


def get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> Any:
    for attempt in range(6):
        resp = client.get(url, params=params)
        if resp.status_code in (418, 429) or resp.status_code >= 500:
            time.sleep(2**attempt)
            continue
        return resp.raise_for_status().json()
    resp.raise_for_status()
    return None


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


def fetch_klines(client: httpx.Client, url: str, limit: int, start_ms: int, end_ms: int) -> list[list[float]]:
    """[open_ms, high, low, close, quote_volume, taker_buy_quote] per 1m bar."""
    rows: list[list[float]] = []
    t = start_ms
    while t < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": t,
            "endTime": end_ms - 1,
            "limit": limit,
        }
        batch = get_json(client, url, params)
        if not batch:
            break
        rows += [[b[0], float(b[2]), float(b[3]), float(b[4]), float(b[7]), float(b[10])] for b in batch]
        t = batch[-1][0] + MIN_MS
    return rows


def fetch_funding(client: httpx.Client, start_ms: int, end_ms: int) -> list[list[float]]:
    rows: list[list[float]] = []
    t = start_ms
    while t < end_ms:
        params = {"symbol": "BTCUSDT", "startTime": t, "endTime": end_ms, "limit": 1000}
        batch = get_json(client, FUNDING, params)
        if not batch:
            break
        rows += [[int(r["fundingTime"]), float(r["fundingRate"])] for r in batch]
        t = int(batch[-1]["fundingTime"]) + 1
    return rows


def fetch_event(client: httpx.Client, day: datetime) -> dict[str, Any] | None:
    """Daily 'Bitcoin above' event for `day` (slug carries the year only from mid-2026)."""
    month = day.strftime("%B").lower()
    for slug in (f"bitcoin-above-on-{month}-{day.day}-{day.year}", f"bitcoin-above-on-{month}-{day.day}"):
        events = get_json(client, GAMMA_EVENTS, {"slug": slug})
        if not events:
            continue
        event = events[0]
        end = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00"))
        if end.date() != day.date():
            continue  # a no-year slug from another year
        markets = []
        for m in event.get("markets", []):
            parsed = classify_question(m.get("question", ""))
            prices = json.loads(m.get("outcomePrices") or "[]")
            if parsed is None or parsed[0] != "above" or not prices or not m.get("clobTokenIds"):
                continue
            markets.append(
                {
                    "strike": parsed[1],
                    "yes_token": json.loads(m["clobTokenIds"])[0],
                    "payout_yes": float(prices[0]),
                    "slug": m["slug"],
                }
            )
        return {"slug": slug, "end_ts": int(end.timestamp()), "markets": markets}
    return None


def fetch_history(client: httpx.Client, token: str, start_s: int, end_s: int) -> list[list[float]]:
    params = {"market": token, "startTs": start_s, "endTs": end_s, "fidelity": 1}
    history = get_json(client, CLOB_HISTORY, params).get("history", [])
    return [[int(p["t"]), float(p["p"])] for p in history]


# ───────────────────────────── signals (own reimplementation) ─────────────────────────────


@dataclass
class Signals:
    times: list[int]  # bar open, ms
    close: list[float]
    vpin: list[float]
    pctl: list[float | None]
    natr: list[float]
    vol_z: list[float]
    move15: list[float]


def build_signals(bars: Sequence[list[float]], bucket_mode: str) -> Signals:
    n = len(bars)
    times = [int(b[0]) for b in bars]
    close = [b[3] for b in bars]

    vpin, current = [0.0] * n, 0.0
    buy = sell = 0.0
    imbalances: deque[float] = deque(maxlen=VPIN_BUCKETS)
    trailing_q: deque[float] = deque(maxlen=7 * 1440)
    trailing_sum = 0.0
    for i, b in enumerate(bars):
        q, qb = b[4], b[5]
        qs = q - qb
        if bucket_mode == "adv":  # trailing 7-day average daily volume / 50, no look-ahead
            bucket = trailing_sum / 7 / VPIN_BUCKETS if len(trailing_q) == trailing_q.maxlen else 0.0
            if len(trailing_q) == trailing_q.maxlen:
                trailing_sum -= trailing_q[0]
            trailing_q.append(q)
            trailing_sum += q
            if bucket <= 0:
                continue
        else:
            bucket = VPIN_BUCKET_USD
        while q > 1e-9:
            frac = min(bucket - (buy + sell), q) / q
            buy, sell = buy + qb * frac, sell + qs * frac
            qb, qs, q = qb * (1 - frac), qs * (1 - frac), q * (1 - frac)
            if buy + sell >= bucket - 1e-6:
                imbalances.append(abs(buy - sell) / (buy + sell))
                buy = sell = 0.0
                if len(imbalances) == VPIN_BUCKETS:
                    current = sum(imbalances) / VPIN_BUCKETS
        vpin[i] = current

    pctl: list[float | None] = [None] * n
    window: deque[float] = deque()
    ranked: list[float] = []
    for i, v in enumerate(vpin):
        if len(window) >= PCTL_WINDOW_MIN:
            pctl[i] = bisect.bisect_left(ranked, v) / len(ranked)
            ranked.pop(bisect.bisect_left(ranked, window.popleft()))
        if v > 0:
            window.append(v)
            bisect.insort(ranked, v)

    # 15m candles keyed by bucket index; minute i uses only candles completed before its own.
    candles: dict[int, list[float]] = {}  # k -> [high, low, close, quote_volume]
    for b in bars:
        k = int(b[0]) // (15 * MIN_MS)
        c = candles.get(k)
        if c is None:
            candles[k] = [b[1], b[2], b[3], b[4]]
        else:
            c[0], c[1], c[2], c[3] = max(c[0], b[1]), min(c[1], b[2]), b[3], c[3] + b[4]
    keys = sorted(candles)
    natr_by_k: dict[int, float] = {}
    vol_stats_by_k: dict[int, tuple[float, float]] = {}
    for j in range(1, len(keys)):
        k = keys[j]
        if j >= 15:
            trs = []
            for jj in range(j - 14, j):
                h, lo, _, _ = candles[keys[jj]]
                prev_close = candles[keys[jj - 1]][2]
                trs.append(max(h - lo, abs(h - prev_close), abs(lo - prev_close)))
            natr_by_k[k] = (sum(trs) / 14) / candles[keys[j - 1]][2]
        if j >= 96:
            vols = [candles[keys[jj]][3] for jj in range(j - 96, j)]
            vol_stats_by_k[k] = (statistics.fmean(vols), statistics.pstdev(vols))

    natr, vol_z, move15 = [0.0] * n, [0.0] * n, [0.0] * n
    rolling_q: deque[float] = deque(maxlen=15)
    for i, b in enumerate(bars):
        rolling_q.append(b[4])
        k = int(b[0]) // (15 * MIN_MS)
        natr[i] = natr_by_k.get(k, 0.0)
        mean, sd = vol_stats_by_k.get(k, (0.0, 0.0))
        vol_z[i] = (sum(rolling_q) - mean) / sd if sd > 0 else 0.0
        move15[i] = close[i] / close[i - 15] - 1 if i >= 15 else 0.0
    return Signals(times, close, vpin, pctl, natr, vol_z, move15)


def level_path(sig: Signals, config: HedgerConfig) -> list[ToxicityLevel]:
    evaluator = ToxicityEvaluator(config)
    levels = []
    for i in range(len(sig.times)):
        metric = ToxicityMetric.model_construct(
            symbol="BTCUSDT",
            timestamp_ms=sig.times[i],
            price=sig.close[i],
            vpin=sig.vpin[i],
            vpin_percentile=sig.pctl[i],
            ob_imbalance_l1=0.5,
            depth_imbalance={},
            ob_toxicity_1pct=0.0,
            volume_z_score=sig.vol_z[i],
            natr_15m=sig.natr[i],
            price_delta_15m_pct=sig.move15[i],
            whale_sweeps_1h_usdt=0.0,
            order_flow_by_token={},
        )
        levels.append(evaluator.evaluate(metric).level)
    return levels


# ───────────────────────────── simulation ─────────────────────────────


@dataclass
class Book:
    """One market held on one side, pre-computed along its minute path."""

    market: str
    holds_yes: bool
    idx: list[int]  # indices into the global minute arrays
    poly: list[float]  # outcome-token mark at each decision time
    delta: list[float]  # position delta (USD perp-equivalent) at each decision time
    perp: list[float]
    funding: list[float]  # funding rate settled at this minute (0 if none)
    payout: float


@dataclass
class Result:
    pnl: float
    max_dd: float
    costs: float
    traded_usd: float
    path: list[float] = field(default_factory=list)


def simulate(
    book: Book, levels: Sequence[ToxicityLevel] | None, config: HedgerConfig, keep_path: bool = False
) -> Result:
    entry = book.poly[0]
    contracts = cash = fees = funding = traded = 0.0
    peak = max_dd = 0.0
    path = []
    for k in range(len(book.idx)):
        mark = book.perp[k]
        if levels is not None:
            for _ in range(MAX_ORDERS_PER_MINUTE):
                signal = decide("BTCUSDT", levels[k], book.delta[k], contracts * mark, config)
                if signal is None:
                    break
                buy = signal.action is HedgeAction.BUY
                taker = signal.order_type is OrderType.IOC_LIMIT
                price = mark * (1 + (SLIPPAGE if buy else -SLIPPAGE)) if taker else mark
                qty = signal.notional_usd / price * (1 if buy else -1)
                contracts += qty
                cash -= qty * price
                fees += signal.notional_usd * (TAKER_FEE if taker else MAKER_FEE)
                traded += signal.notional_usd
        funding += contracts * mark * book.funding[k]
        mtm = SHARES * (book.poly[k] - entry) + cash + contracts * mark - fees - funding
        peak = max(peak, mtm)
        max_dd = max(max_dd, peak - mtm)
        if keep_path:
            path.append(mtm)
    if contracts:  # flatten whatever is left at the settlement minute
        notional = abs(contracts) * book.perp[-1]
        cash += contracts * book.perp[-1] * (1 - math.copysign(SLIPPAGE, contracts))
        fees += notional * TAKER_FEE
        traded += notional
    pnl = SHARES * (book.payout - entry) + cash - fees - funding
    peak = max(peak, pnl)
    max_dd = max(max_dd, peak - pnl)
    if keep_path:
        path.append(pnl)
    return Result(pnl, max_dd, fees + funding, traded, path)


def cvar(values: Sequence[float], q: float = 0.05) -> float:
    worst = sorted(values)[: max(1, int(len(values) * q))]
    return statistics.fmean(worst)


def summarise(results: Sequence[Result]) -> dict[str, float]:
    pnl = [r.pnl for r in results]
    dd = sorted(r.max_dd for r in results)
    return {
        "mean_pnl": statistics.fmean(pnl),
        "std_pnl": statistics.pstdev(pnl),
        "cvar5": cvar(pnl),
        "worst": min(pnl),
        "mean_max_dd": statistics.fmean(dd),
        "p95_max_dd": dd[int(0.95 * (len(dd) - 1))],
        "mean_costs": statistics.fmean(r.costs for r in results),
        "mean_traded_usd": statistics.fmean(r.traded_usd for r in results),
    }


# ───────────────────────────── main ─────────────────────────────


def main() -> None:
    start_ms = int((START - WARMUP).timestamp()) * MS
    end_ms = int((END + timedelta(days=1)).timestamp()) * MS
    tag = f"{START:%Y%m%d}_{END:%Y%m%d}"
    days = [START + timedelta(days=d) for d in range((END - START).days + 1)]

    with httpx.Client(timeout=30) as client:
        print("Binance spot 1m klines ...", flush=True)
        spot = cached(f"spot_{tag}", lambda: fetch_klines(client, SPOT_KLINES, 1000, start_ms, end_ms))
        print("Binance USD-M perp 1m klines ...", flush=True)
        perp_rows = cached(f"perp_{tag}", lambda: fetch_klines(client, PERP_KLINES, 1500, start_ms, end_ms))
        funding_rows = cached(f"funding_{tag}", lambda: fetch_funding(client, start_ms, end_ms))
        print("Polymarket events ...", flush=True)
        events = cached(f"events_{tag}", lambda: [fetch_event(client, d) for d in days])

        sig = build_signals(spot, BUCKET_MODE)
        index = {t: i for i, t in enumerate(sig.times)}
        perp = {int(r[0]): r[3] for r in perp_rows}
        funding_at = {int(t) // MIN_MS * MIN_MS: rate for t, rate in funding_rows}

        # ATM strike at entry; its minute history is fetched (and cached) per market.
        chosen = []
        for event in events:
            if not event or not event["markets"]:
                continue
            entry_ms = (event["end_ts"] - HOLD_SECS) * MS
            i0 = index.get(entry_ms)
            if i0 is None:
                continue
            spot0 = sig.close[i0 - 1]
            market = min(event["markets"], key=lambda m: abs(m["strike"] - spot0))
            if market["payout_yes"] not in (0.0, 1.0):
                continue
            chosen.append((event, market))
        print(f"Polymarket minute history for {len(chosen)} markets ...", flush=True)
        histories = cached(
            f"histories_{tag}",
            lambda: {
                m["slug"]: fetch_history(client, m["yes_token"], e["end_ts"] - HOLD_SECS - 3600, e["end_ts"])
                for e, m in chosen
            },
        )

    config = HedgerConfig()
    always_cfg = HedgerConfig(baseline_hedge_ratio=1.0, pre_hedge_ratio=1.0, emergency_hedge_ratio=1.0)
    t0 = time.time()
    levels = level_path(sig, config)
    warm = next(i for i, p in enumerate(sig.pctl) if p is not None)
    print(f"Evaluator over {len(levels):,} minutes in {time.time() - t0:.0f}s", flush=True)

    books: list[Book] = []
    skipped: dict[str, int] = {}
    for event, market in chosen:
        history = histories.get(market["slug"]) or []
        entry_ms = (event["end_ts"] - HOLD_SECS) * MS
        idx = [index[t] for t in range(entry_ms, event["end_ts"] * MS, MIN_MS) if t in index]
        if len(history) < 600 or len(idx) < 1380 or idx[0] < warm:
            skipped["thin data"] = skipped.get("thin data", 0) + 1
            continue
        if any(sig.times[i] not in perp for i in idx):
            skipped["perp gap"] = skipped.get("perp gap", 0) + 1
            continue
        h_ts = [int(p[0]) for p in history]
        yes = []
        for i in idx:
            j = bisect.bisect_right(h_ts, sig.times[i] // MS + 60) - 1
            yes.append(history[max(j, 0)][1])
        if not 0.1 <= yes[0] <= 0.9:
            skipped["not near the money at entry"] = skipped.get("not near the money at entry", 0) + 1
            continue
        spec = MarketSpec(market["slug"], "above", "BTCUSDT", float(event["end_ts"]), strike=market["strike"])
        for holds_yes in (True, False):
            delta = [
                position_delta_usd(
                    spec,
                    holds_yes,
                    SHARES,
                    sig.close[i],
                    sigma_15m_from_natr(sig.natr[i], config.min_sigma_15m),
                    sig.times[i] / MS + 60,
                    no_hedge_final_secs=config.no_hedge_final_secs,
                    max_abs_delta_usd=config.max_delta_per_position_usd,
                )
                for i in idx
            ]
            books.append(
                Book(
                    market=market["slug"],
                    holds_yes=holds_yes,
                    idx=idx,
                    poly=yes if holds_yes else [1 - p for p in yes],
                    delta=delta,
                    perp=[perp[sig.times[i]] for i in idx],
                    funding=[funding_at.get(sig.times[i] + MIN_MS, 0.0) for i in idx],
                    payout=market["payout_yes"] if holds_yes else 1 - market["payout_yes"],
                )
            )
    n_markets = len(books) // 2
    print(f"{n_markets} markets ({len(books)} books); skipped: {skipped}", flush=True)

    normal = [ToxicityLevel.NORMAL] * len(levels)
    runs: dict[str, list[Result]] = {
        "unhedged": [simulate(b, None, config) for b in books],
        "limit_only": [simulate(b, [normal[i] for i in b.idx], config) for b in books],
        "gated": [simulate(b, [levels[i] for i in b.idx], config) for b in books],
        "always": [simulate(b, [normal[i] for i in b.idx], always_cfg) for b in books],
    }
    rng = random.Random(7)
    span = len(levels) - warm
    placebo = []
    for _ in range(PLACEBO_SEEDS):
        shift = rng.randrange(7 * 1440, span - 7 * 1440)
        shifted = [
            simulate(b, [levels[warm + (i - warm + shift) % span] for i in b.idx], config) for b in books
        ]
        placebo.append(summarise(shifted))

    report: dict[str, Any] = {
        "vpin_bucket": BUCKET_MODE,
        "period": f"{START:%Y-%m-%d} -> {END:%Y-%m-%d}",
        "markets": n_markets,
        "books": len(books),
        "shares": SHARES,
        "skipped": skipped,
        "strategies": {name: summarise(res) for name, res in runs.items()},
        "placebo": {
            key: {
                "mean": statistics.fmean(p[key] for p in placebo),
                "min": min(p[key] for p in placebo),
                "max": max(p[key] for p in placebo),
            }
            for key in placebo[0]
        },
    }
    gated = report["strategies"]["gated"]

    def early(b: Book) -> Book:
        cut = len(b.idx) - EXIT_EARLY_MIN
        return Book(
            b.market,
            b.holds_yes,
            b.idx[:cut],
            b.poly[:cut],
            b.delta[:cut],
            b.perp[:cut],
            b.funding[:cut],
            b.poly[cut - 1],
        )

    early_books = [early(b) for b in books]
    report[f"exit_{EXIT_EARLY_MIN}m_before_resolution"] = {
        "unhedged": summarise([simulate(b, None, config) for b in early_books]),
        "gated": summarise([simulate(b, [levels[i] for i in b.idx], config) for b in early_books]),
        "always": summarise([simulate(b, [normal[i] for i in b.idx], always_cfg) for b in early_books]),
    }
    report["deadband_sensitivity_exit_early_always"] = {
        str(band): summarise(
            [
                simulate(
                    b, [normal[i] for i in b.idx], always_cfg.model_copy(update={"min_rebalance_usd": band})
                )
                for b in early_books
            ]
        )
        for band in DEADBANDS_USD
    }
    report["placebo_beats_gated"] = {
        "cvar5": sum(p["cvar5"] >= gated["cvar5"] for p in placebo) / len(placebo),
        "std_pnl": sum(p["std_pnl"] <= gated["std_pnl"] for p in placebo) / len(placebo),
        "p95_max_dd": sum(p["p95_max_dd"] <= gated["p95_max_dd"] for p in placebo) / len(placebo),
    }

    # Premise: is the VPIN regime followed by bigger BTC moves? (all warmed-up minutes, 15m ahead)
    bands = [
        ("< 0.60", 0.0, 0.60),
        ("0.60-0.85", 0.60, 0.85),
        ("0.85-0.95", 0.85, 0.95),
        (">= 0.95", 0.95, 1.01),
    ]
    moves = {name: [] for name, _, _ in bands}
    for i in range(warm, len(sig.close) - 15):
        p = sig.pctl[i]
        if p is None:
            continue
        m = abs(sig.close[i + 15] / sig.close[i] - 1)
        for name, lo, hi in bands:
            if lo <= p < hi:
                moves[name].append(m)
    all_moves = sorted(m for v in moves.values() for m in v)
    big = all_moves[int(0.99 * (len(all_moves) - 1))]
    base_rate = sum(m >= big for m in all_moves) / len(all_moves)
    report["vpin_regimes"] = {
        name: {
            "share_of_time": len(v) / len(all_moves),
            "mean_abs_move_15m": statistics.fmean(v),
            "p_move_ge_global_p99": sum(m >= big for m in v) / len(v),
            "lift_vs_base": (sum(m >= big for m in v) / len(v)) / base_rate,
        }
        for name, v in moves.items()
        if v
    }
    window = [levels[i] for b in books[::2] for i in b.idx]
    report["level_share_in_books"] = {lv.value: window.count(lv) / len(window) for lv in ToxicityLevel}

    HERE.joinpath(f"results_{BUCKET_MODE}.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    charts(report, runs, books, levels, config, always_cfg)


def print_report(report: dict[str, Any]) -> None:
    print(f"\n{report['period']}: {report['markets']} markets x 2 sides, {report['shares']:.0f} shares each")
    cols = [
        "mean_pnl",
        "std_pnl",
        "cvar5",
        "worst",
        "mean_max_dd",
        "p95_max_dd",
        "mean_costs",
        "mean_traded_usd",
    ]
    print(f"{'strategy':<12}" + "".join(f"{c:>16}" for c in cols))
    for name, s in report["strategies"].items():
        print(f"{name:<12}" + "".join(f"{s[c]:>16,.1f}" for c in cols))
    p = report["placebo"]
    print(f"{'placebo avg':<12}" + "".join(f"{p[c]['mean']:>16,.1f}" for c in cols))
    print("share of placebo seeds at least as good as gated:", report["placebo_beats_gated"])
    print(f"exit {EXIT_EARLY_MIN} min before resolution:")
    for name, s in report[f"exit_{EXIT_EARLY_MIN}m_before_resolution"].items():
        print(f"  {name:<10}" + "".join(f"{s[c]:>16,.1f}" for c in cols))
    print("early exit + always hedged, by rebalance deadband (USD):")
    for band, s in report["deadband_sensitivity_exit_early_always"].items():
        print(f"  {band:<10}" + "".join(f"{s[c]:>16,.1f}" for c in cols))
    print("level share inside the books:", {k: f"{v:.1%}" for k, v in report["level_share_in_books"].items()})
    print("\nVPIN percentile regime -> next-15m BTC move")
    for name, r in report["vpin_regimes"].items():
        print(
            f"  {name:<10} time {r['share_of_time']:6.1%}  mean |move| {r['mean_abs_move_15m']:.3%}  "
            f"P(>= p99 move) {r['p_move_ge_global_p99']:.2%}  lift {r['lift_vs_base']:.2f}x"
        )


def charts(
    report: dict[str, Any],
    runs: dict[str, list[Result]],
    books: list[Book],
    levels: list[ToxicityLevel],
    config: HedgerConfig,
    always_cfg: HedgerConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(exist_ok=True)
    sfx = f"_{BUCKET_MODE}"
    names = ["unhedged", "limit_only", "gated", "always"]
    colours = {"unhedged": "#9e9e9e", "limit_only": "#90caf9", "gated": "#7e57c2", "always": "#26a69a"}

    fig, ax = plt.subplots(figsize=(9, 5))
    s = report["strategies"]
    for name in names:
        ax.scatter(s[name]["mean_costs"], -s[name]["cvar5"], s=120, color=colours[name], label=name, zorder=3)
    p = report["placebo"]
    ax.errorbar(
        p["mean_costs"]["mean"],
        -p["cvar5"]["mean"],
        xerr=[
            [p["mean_costs"]["mean"] - p["mean_costs"]["min"]],
            [p["mean_costs"]["max"] - p["mean_costs"]["mean"]],
        ],
        yerr=[[p["cvar5"]["max"] - p["cvar5"]["mean"]], [p["cvar5"]["mean"] - p["cvar5"]["min"]]],
        fmt="s",
        color="#ef6c00",
        label=f"placebo (shifted timing, {PLACEBO_SEEDS} seeds)",
        zorder=2,
    )
    ax.set_xlabel("mean hedging cost per position, USD (fees + funding)")
    ax.set_ylabel("expected shortfall 5%, USD (lower is better)")
    ax.set_title(f"Tail risk vs cost: {report['markets']} daily BTC markets, {report['period']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / f"risk_vs_cost{sfx}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([[r.pnl for r in runs[n]] for n in names], tick_labels=names, showfliers=True, whis=(5, 95))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"P&L per position, USD ({SHARES:.0f} shares, held 24h to resolution)")
    ax.set_title("Per-position P&L distribution (whiskers 5th-95th percentile)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS / f"pnl_distribution{sfx}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    regimes = report["vpin_regimes"]
    ax.bar(list(regimes), [r["lift_vs_base"] for r in regimes.values()], color="#7e57c2")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("VPIN percentile (vs own trailing 7 days)")
    ax.set_ylabel("P(next 15m BTC move >= p99) / base rate")
    ax.set_title("VPIN as a regime signal: how often a tail move follows")
    for x, r in enumerate(regimes.values()):
        ax.text(
            x,
            r["lift_vs_base"],
            f"{r['lift_vs_base']:.2f}x\n{r['share_of_time']:.0%} of time",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(ASSETS / f"vpin_regimes{sfx}.png", dpi=150)
    plt.close(fig)

    worst = min(range(len(books)), key=lambda j: runs["unhedged"][j].pnl)
    book = books[worst]
    lv = [levels[i] for i in book.idx]
    paths = {
        "unhedged": simulate(book, None, config, keep_path=True).path,
        "gated": simulate(book, lv, config, keep_path=True).path,
        "always": simulate(book, [ToxicityLevel.NORMAL] * len(lv), always_cfg, keep_path=True).path,
    }
    hours = [(k - len(book.idx)) / 60 for k in range(len(book.idx) + 1)]
    fig, ax = plt.subplots(figsize=(10, 5))
    start = 0
    for k in range(1, len(lv) + 1):
        if k == len(lv) or lv[k] is not lv[start]:
            if lv[start] is not ToxicityLevel.NORMAL:
                shade = "#ffcc80" if lv[start] is ToxicityLevel.PRE_HEDGING_ALERT else "#ef9a9a"
                ax.axvspan(hours[start], hours[k], color=shade, alpha=0.6, lw=0)
            start = k
    for name, path in paths.items():
        ax.plot(hours, path, color=colours[name], label=name, lw=1.6)
    side = "YES" if book.holds_yes else "NO"
    alerts = sum(level is not ToxicityLevel.NORMAL for level in lv)
    legend = "orange = PRE, red = EMERGENCY" if alerts else "no PRE/EMERGENCY minute in 24h"
    ax.set_title(f"Worst unhedged position: {book.market} ({side}); {legend}")
    ax.set_xlabel("hours to resolution")
    ax.set_ylabel("mark-to-market P&L, USD")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / f"worst_position{sfx}.png", dpi=150)
    plt.close(fig)

    early = report[f"exit_{EXIT_EARLY_MIN}m_before_resolution"]
    points = [
        ("unhedged, to expiry", s["unhedged"], "#9e9e9e", "o"),
        ("always, to expiry", s["always"], "#26a69a", "o"),
        ("gated, to expiry", s["gated"], "#7e57c2", "o"),
        (f"unhedged, exit {EXIT_EARLY_MIN}m early", early["unhedged"], "#9e9e9e", "D"),
        (f"gated, exit {EXIT_EARLY_MIN}m early", early["gated"], "#7e57c2", "D"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, stats, colour, marker in points:
        ax.scatter(
            stats["mean_costs"], -stats["cvar5"], s=110, color=colour, marker=marker, label=label, zorder=3
        )
    bands = report["deadband_sensitivity_exit_early_always"]
    xs = [b["mean_costs"] for b in bands.values()]
    ys = [-b["cvar5"] for b in bands.values()]
    ax.plot(
        xs, ys, "-D", color="#00695c", label=f"always, exit {EXIT_EARLY_MIN}m early (by deadband)", zorder=3
    )
    for band, x, y in zip(bands, xs, ys, strict=True):
        ax.annotate(f"${int(band):,}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("mean hedging cost per position, USD (fees + funding)")
    ax.set_ylabel("expected shortfall 5%, USD (lower is better)")
    ax.set_title("Where the protection actually comes from: early exit + delta neutrality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / f"early_exit{sfx}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
