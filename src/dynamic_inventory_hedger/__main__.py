"""Entry point: `dynamic-inventory-hedger` / `python -m dynamic_inventory_hedger`."""

from __future__ import annotations

import asyncio
import logging
import sys

from followsm_sdk import FollowSMClient

from dynamic_inventory_hedger.config import HedgerConfig, load_config
from dynamic_inventory_hedger.engine import Hedger
from dynamic_inventory_hedger.execution.drivers import build_driver
from dynamic_inventory_hedger.inventory.polymarket import PolymarketInventory
from dynamic_inventory_hedger.risk.controls import ExecutionHalted, Journal
from dynamic_inventory_hedger.signals.consumer import EnterpriseRequiredError, SignalConsumer

log = logging.getLogger("dynamic_inventory_hedger")


async def run(config: HedgerConfig) -> None:
    client = FollowSMClient(api_key=config.followsm_api_key)
    inventory = PolymarketInventory(config)
    driver = build_driver(config)
    hedger = Hedger(
        config,
        SignalConsumer(client, config.signal_source, config.poll_interval_secs),
        inventory,
        driver,
        Journal(config.journal_path),
    )
    mode = "LIVE" + ("-DEMO" if config.binance_demo else "") if config.live_trading else "PAPER"
    log.info(
        "Starting in %s mode (signals: %s, wallet: %s)",
        mode,
        config.signal_source,
        config.polymarket_wallet or "none",
    )
    try:
        await hedger.run()
    finally:
        await driver.close()
        await inventory.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(load_config()))
    except (EnterpriseRequiredError, ExecutionHalted) as exc:
        log.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
