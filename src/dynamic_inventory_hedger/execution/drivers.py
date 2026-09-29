"""Execution drivers for Binance USD-M perps: a paper driver and a live ccxt driver.

Both expose the same interface. Order style comes from the HedgeSignal:
  POST_ONLY_LIMIT -> post-only (Binance GTX) limit at the touch: bid for buys, ask for sells
  IOC_LIMIT       -> immediate-or-cancel limit at the slippage protection price
"""

from __future__ import annotations

import logging
from typing import Protocol

import ccxt.async_support as ccxt
import httpx

from dynamic_inventory_hedger.config import HedgerConfig
from dynamic_inventory_hedger.models import HedgeAction, HedgeSignal, OrderType
from dynamic_inventory_hedger.risk.controls import Quote, protection_price

log = logging.getLogger(__name__)

FAPI_PUBLIC = "https://fapi.binance.com"


def unified(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT (ccxt linear perp symbol)."""
    return f"{symbol.removesuffix('USDT')}/USDT:USDT"


class ExecutionDriver(Protocol):
    async def quote(self, symbol: str) -> Quote: ...
    async def hedge_notional_usd(self, symbol: str, mark: float) -> float: ...
    async def execute(self, signal: HedgeSignal, quote: Quote) -> float: ...
    async def cancel_open_orders(self, symbol: str) -> None: ...
    async def close(self) -> None: ...


class PaperDriver:
    """Paper trading on real Binance USD-M quotes. Post-only orders fill at the touch; IOC at protection."""

    def __init__(self, config: HedgerConfig) -> None:
        self.config = config
        self._http = httpx.AsyncClient(base_url=FAPI_PUBLIC, timeout=10.0)
        self.positions: dict[str, float] = {}  # symbol -> signed contracts

    async def quote(self, symbol: str) -> Quote:
        resp = await self._http.get("/fapi/v1/ticker/bookTicker", params={"symbol": symbol})
        resp.raise_for_status()
        body = resp.json()
        return Quote(bid=float(body["bidPrice"]), ask=float(body["askPrice"]))

    async def hedge_notional_usd(self, symbol: str, mark: float) -> float:
        return self.positions.get(symbol, 0.0) * mark

    async def execute(self, signal: HedgeSignal, quote: Quote) -> float:
        buy = signal.action is HedgeAction.BUY
        if signal.order_type is OrderType.POST_ONLY_LIMIT:
            price = quote.bid if buy else quote.ask
        else:
            price = (
                min(quote.ask, protection_price(signal.action, quote, self.config))
                if buy
                else max(quote.bid, protection_price(signal.action, quote, self.config))
            )
        contracts = signal.notional_usd / price
        self.positions[signal.symbol] = self.positions.get(signal.symbol, 0.0) + (
            contracts if buy else -contracts
        )
        log.info(
            "[PAPER] %s %.6f %s @ %.2f (%s)",
            signal.action.value,
            contracts,
            signal.symbol,
            price,
            signal.order_type.value,
        )
        return contracts

    async def cancel_open_orders(self, symbol: str) -> None:
        return None

    async def close(self) -> None:
        await self._http.aclose()


class BinanceUsdmDriver:
    """Live (or Binance demo) execution via ccxt."""

    def __init__(self, config: HedgerConfig) -> None:
        self.config = config
        self.exchange = ccxt.binanceusdm(
            {"apiKey": config.binance_api_key, "secret": config.binance_api_secret, "enableRateLimit": True}
        )
        if config.binance_demo:
            self.exchange.enable_demo_trading(True)
        self._loaded = False

    async def _markets(self) -> None:
        if not self._loaded:
            await self.exchange.load_markets()
            self._loaded = True

    async def quote(self, symbol: str) -> Quote:
        await self._markets()
        book = await self.exchange.fetch_order_book(unified(symbol), limit=5)
        return Quote(bid=float(book["bids"][0][0]), ask=float(book["asks"][0][0]))

    async def hedge_notional_usd(self, symbol: str, mark: float) -> float:
        await self._markets()
        positions = await self.exchange.fetch_positions([unified(symbol)])
        contracts = 0.0
        for p in positions:
            size = float(p.get("contracts") or 0.0)
            contracts += size if p.get("side") == "long" else -size
        return contracts * mark

    async def execute(self, signal: HedgeSignal, quote: Quote) -> float:
        await self._markets()
        sym = unified(signal.symbol)
        buy = signal.action is HedgeAction.BUY
        params: dict[str, object] = {"reduceOnly": signal.reduce_only}
        if signal.order_type is OrderType.POST_ONLY_LIMIT:
            price = quote.bid if buy else quote.ask
            params["postOnly"] = True
        else:
            price = protection_price(signal.action, quote, self.config)
            params["timeInForce"] = "IOC"
        amount = float(self.exchange.amount_to_precision(sym, signal.notional_usd / price))
        market = self.exchange.market(sym)
        min_amount = (market["limits"]["amount"] or {}).get("min") or 0.0
        min_cost = (market["limits"]["cost"] or {}).get("min") or 0.0
        if amount < min_amount or amount * price < min_cost:
            log.info(
                "Skipping %s %s: %.6f contracts below exchange minimum", signal.action.value, sym, amount
            )
            return 0.0
        order = await self.exchange.create_order(
            sym,
            "limit",
            "buy" if buy else "sell",
            amount,
            float(self.exchange.price_to_precision(sym, price)),
            params,
        )
        filled = float(order.get("filled") or 0.0)
        log.info(
            "[LIVE%s] %s %.6f %s @ %.2f %s -> id=%s filled=%.6f",
            "-DEMO" if self.config.binance_demo else "",
            signal.action.value,
            amount,
            sym,
            price,
            signal.order_type.value,
            order.get("id"),
            filled,
        )
        return filled

    async def cancel_open_orders(self, symbol: str) -> None:
        await self._markets()
        await self.exchange.cancel_all_orders(unified(symbol))

    async def close(self) -> None:
        await self.exchange.close()


def build_driver(config: HedgerConfig) -> ExecutionDriver:
    return BinanceUsdmDriver(config) if config.live_trading else PaperDriver(config)
