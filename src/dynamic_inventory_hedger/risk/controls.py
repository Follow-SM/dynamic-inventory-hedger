"""Risk controls: spread/slippage protection, a failure circuit breaker, and a JSONL trade journal."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.models import HedgeAction

log = logging.getLogger(__name__)


class ExecutionHalted(RuntimeError):
    """Raised once hedge execution failed too many times in a row; a human must restart the hedger."""


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4 if self.mid > 0 else float("inf")


def check_spread(quote: Quote, config: HedgerConfig) -> str | None:
    """Reason to refuse trading, or None if the book is tight enough."""
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return f"invalid book bid={quote.bid} ask={quote.ask}"
    if quote.spread_bps > config.max_spread_bps:
        return f"spread {quote.spread_bps:.1f}bps > {config.max_spread_bps:.1f}bps"
    return None


def protection_price(action: HedgeAction, quote: Quote, config: HedgerConfig) -> float:
    """Worst price an IOC may fill at: mid ± max_slippage_bps."""
    offset = quote.mid * config.max_slippage_bps / 1e4
    return quote.mid + offset if action is HedgeAction.BUY else quote.mid - offset


class FailureBreaker:
    """Counts consecutive execution failures; trips permanently at the configured limit."""

    def __init__(self, config: HedgerConfig) -> None:
        self.limit = config.max_consecutive_failures
        self.consecutive = 0

    def success(self) -> None:
        self.consecutive = 0

    def failure(self, exc: BaseException) -> None:
        self.consecutive += 1
        log.error("Hedge execution failed (%d/%d): %s", self.consecutive, self.limit, exc)
        if self.consecutive >= self.limit:
            raise ExecutionHalted(
                f"{self.consecutive} consecutive hedge failures (last: {exc}); stopping the hedger"
            ) from exc


class Journal:
    """Append-only JSONL log of every decision and fill: the paper-trading / replay record."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def write(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
