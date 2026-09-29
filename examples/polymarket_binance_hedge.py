"""End-to-end demo: hedge a Polymarket BTC position on Binance USD-M perps (paper mode).

With POLYMARKET_WALLET set, the hedger tracks that wallet's real positions. Without it, this
demo seeds a sample position in the nearest live "Bitcoin above $K" market, so you can watch
the toxicity bands, delta mapping and hedge orders against live data with no wallet or keys
beyond FollowSM.

    export FOLLOWSM_API_KEY=...        # ENTERPRISE for the WebSocket stream,
    export SIGNAL_SOURCE=poll          # ...or poll REST on any tier
    python examples/polymarket_binance_hedge.py --minutes 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime

import httpx
from followsm_sdk import FollowSMClient

from dynamic_inventory_hedger.config import load_config
from dynamic_inventory_hedger.engine import Hedger
from dynamic_inventory_hedger.execution.drivers import PaperDriver
from dynamic_inventory_hedger.inventory.delta import classify_question
from dynamic_inventory_hedger.inventory.polymarket import PolymarketInventory, PolymarketPosition
from dynamic_inventory_hedger.risk.controls import Journal
from dynamic_inventory_hedger.signals.consumer import SignalConsumer

GAMMA_API = "https://gamma-api.polymarket.com"


async def seed_sample_position(inventory: PolymarketInventory, spot: float, shares: float) -> str:
    """Hold `shares` YES in the live 'Bitcoin above' market whose strike is nearest spot."""
    events = []
    async with httpx.AsyncClient(timeout=15) as http:
        for offset in range(0, 500, 100):  # Gamma pages at 100 events
            resp = await http.get(
                f"{GAMMA_API}/events",
                params={"tag_slug": "bitcoin", "closed": "false", "limit": 100, "offset": offset},
            )
            resp.raise_for_status()
            page = resp.json()
            events += page
            if len(page) < 100:
                break
    candidates = []
    for event in events:
        if not event.get("slug", "").startswith("bitcoin-above"):
            continue
        for market in event.get("markets", []):
            parsed = classify_question(market.get("question", ""))
            end = market.get("endDate")
            if parsed is None or parsed[0] != "above" or not end or not market.get("active"):
                continue
            expiry = datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
            if market.get("closed") or expiry - time.time() < 900:
                continue
            candidates.append((abs(parsed[1] - spot), market))
    if not candidates:
        raise SystemExit("No live 'Bitcoin above' market found; set POLYMARKET_WALLET instead.")
    market = min(candidates, key=lambda c: c[0])[1]
    yes_token = json.loads(market["clobTokenIds"])[0]
    inventory.positions = [
        PolymarketPosition(market["slug"], market["question"], yes_token, yes_token, True, shares, 0.5, 0.5)
    ]
    inventory._specs[market["slug"]] = await inventory._resolve_spec(market["slug"], market["question"])
    return str(market["question"])


async def main(minutes: float, shares: float) -> None:
    config = load_config()
    client = FollowSMClient(api_key=config.followsm_api_key)
    inventory = PolymarketInventory(config)
    driver = PaperDriver(config)
    hedger = Hedger(
        config,
        SignalConsumer(client, config.signal_source, config.poll_interval_secs),
        inventory,
        driver,
        Journal(config.journal_path),
    )
    if not config.polymarket_wallet:
        spot = (await driver.quote("BTCUSDT")).mid
        question = await seed_sample_position(inventory, spot, shares)
        logging.info("Sample position: %.0f YES in %r (BTC perp mid %.1f)", shares, question, spot)
        inventory.refresh = _noop  # type: ignore[method-assign]
    try:
        await asyncio.wait_for(hedger.run(), timeout=minutes * 60)
    except TimeoutError:
        logging.info("Demo finished. Paper hedge: %s. Journal: %s", driver.positions, config.journal_path)
    finally:
        await driver.close()
        await inventory.aclose()


async def _noop() -> None:
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument(
        "--shares", type=float, default=20_000.0, help="sample YES shares when no wallet is set"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main(args.minutes, args.shares))
