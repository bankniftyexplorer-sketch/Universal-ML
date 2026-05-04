from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

SYMBOL_PAYLOAD_RE = re.compile(r"SYMBOL:\s*(.*?)\s*,\s*OPEN:", re.DOTALL)
MARKET_PREFIX_RE = re.compile(r"^(?P<exchange>[A-Z0-9_]+):(?P<symbol>.+)$")
SOURCE_SYMBOL_CLEAN_RE = re.compile(r"[^A-Z0-9!._^:-]")
PAIR_SYMBOL_CLEAN_RE = re.compile(r"[^A-Z0-9]")
EXCHANGE_CLEAN_RE = re.compile(r"[^A-Z0-9_]")
CONTINUOUS_FUTURE_RE = re.compile(r"^(?P<root>.+?)(?P<ordinal>\d*)!$")
DATED_FUTURE_RE = re.compile(
    r"^(?P<root>.+?)(?P<expiry>(?:\d{1,2})?[FGHJKMNQUVXZ]\d{2,4})$"
)
PERPETUAL_RE = re.compile(r"^(?P<root>.+?)(?:PERPETUAL|PERP|\.P)$")

COMMON_QUOTES = tuple(
    sorted(
        {
            "USDT",
            "USDC",
            "FDUSD",
            "BUSD",
            "TUSD",
            "PYUSD",
            "DAI",
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "KRW",
            "INR",
            "AUD",
            "CAD",
            "CHF",
            "BRL",
            "TRY",
            "MXN",
            "RUB",
            "ZAR",
            "SGD",
            "HKD",
            "AED",
            "SAR",
            "BTC",
            "ETH",
        },
        key=len,
        reverse=True,
    )
)

JSON_EXCHANGE_KEYS = (
    "source_exchange",
    "exchange",
    "venue",
    "market",
)
JSON_SYMBOL_KEYS = (
    "source_symbol",
    "symbol",
    "ticker",
    "tv_symbol",
    "tickerid",
    "instrument",
)
JSON_PAIR_KEYS = ("pair_symbol", "underlying", "asset", "base_asset", "root")
JSON_CONTRACT_KEYS = ("contract_kind", "instrument_type")
JSON_QUOTE_KEYS = ("quote_asset", "quote", "settlement_asset", "currency")
JSON_EXPIRY_KEYS = ("expiry_code", "expiry", "maturity")
JSON_ASSET_CLASS_KEYS = ("asset_class", "security_type")
JSON_MARKET_TYPE_KEYS = ("syminfo_type", "market_type", "type")


@dataclass(frozen=True)
class InstrumentRegistryEntry:
    pair_symbol: str
    asset_class: str
    source_exchange: str
    source_symbol: str
    contract_kind: str
    quote_asset: str | None = None


@dataclass(frozen=True)
class ParsedSymbolIdentity:
    base_symbol: str
    pair_symbol: str
    asset_class: str
    source_exchange: str | None
    source_symbol: str
    symbol_payload_json: str
    contract_kind: str
    quote_asset: str | None
    expiry_code: str | None


STRICT_INDEX_REGISTRY = (
    InstrumentRegistryEntry("NIFTY", "SPOT", "YAHOO", "^NSEI", "SPOT", "INR"),
    InstrumentRegistryEntry(
        "NIFTY", "FUT", "NSE", "NIFTY1!", "CONTINUOUS_FUTURE", "INR"
    ),
    InstrumentRegistryEntry("BANKNIFTY", "SPOT", "YAHOO", "^NSEBANK", "SPOT", "INR"),
    InstrumentRegistryEntry(
        "BANKNIFTY", "FUT", "NSE", "BANKNIFTY1!", "CONTINUOUS_FUTURE", "INR"
    ),
    InstrumentRegistryEntry(
        "FINNIFTY", "SPOT", "YAHOO", "NIFTY_FIN_SERVICE.NS", "SPOT", "INR"
    ),
    InstrumentRegistryEntry(
        "FINNIFTY", "FUT", "NSE", "FINNIFTY1!", "CONTINUOUS_FUTURE", "INR"
    ),
    InstrumentRegistryEntry(
        "MIDCPNIFTY", "SPOT", "YAHOO", "NIFTY_MID_SELECT.NS", "SPOT", "INR"
    ),
    InstrumentRegistryEntry(
        "MIDCPNIFTY",
        "FUT",
        "NSE",
        "MIDCPNIFTY1!",
        "CONTINUOUS_FUTURE",
        "INR",
    ),
    InstrumentRegistryEntry("SENSEX", "SPOT", "YAHOO", "^BSESN", "SPOT", "INR"),
    InstrumentRegistryEntry("NIFTYNXT50", "SPOT", "YAHOO", "^NSMIDCP", "SPOT", "INR"),
    InstrumentRegistryEntry("INDIA_VIX", "SPOT", "YAHOO", "^INDIAVIX", "SPOT", "INR"),
    InstrumentRegistryEntry("SPX500", "SPOT", "YAHOO", "^GSPC", "SPOT", "USD"),
    InstrumentRegistryEntry("VIX", "SPOT", "YAHOO", "^VIX", "SPOT", "USD"),
    InstrumentRegistryEntry("VIX9D", "SPOT", "YAHOO", "^VIX9D", "SPOT", "USD"),
    InstrumentRegistryEntry("VIX3M", "SPOT", "YAHOO", "^VIX3M", "SPOT", "USD"),
    InstrumentRegistryEntry(
        "SENSEX", "FUT", "BSE_DLY", "BSX1!", "CONTINUOUS_FUTURE", "INR"
    ),
    InstrumentRegistryEntry("BANKEX", "SPOT", "BSE_DLY", "BANK", "SPOT", "INR"),
    InstrumentRegistryEntry(
        "BANKEX", "FUT", "BSE_DLY", "BKX1!", "CONTINUOUS_FUTURE", "INR"
    ),
)

STRICT_CRYPTO_REGISTRY = (
    InstrumentRegistryEntry("BTC", "SPOT", "YAHOO", "BTC-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("ETH", "SPOT", "YAHOO", "ETH-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("BNB", "SPOT", "YAHOO", "BNB-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("XRP", "SPOT", "YAHOO", "XRP-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("SOL", "SPOT", "YAHOO", "SOL-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("TRX", "SPOT", "YAHOO", "TRX-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("DOGE", "SPOT", "YAHOO", "DOGE-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("ADA", "SPOT", "YAHOO", "ADA-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("BCH", "SPOT", "YAHOO", "BCH-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("LINK", "SPOT", "YAHOO", "LINK-USD", "SPOT", "USD"),
    InstrumentRegistryEntry("BTC", "FUT", "BINANCE", "BTCUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry(
        "ETH", "FUT", "CME_DL", "ETH1!", "CONTINUOUS_FUTURE", "USD"
    ),
    InstrumentRegistryEntry("BNB", "FUT", "BINANCE", "BNBUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry("XRP", "FUT", "BINANCE", "XRPUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry("SOL", "FUT", "BINANCE", "SOLUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry("TRX", "FUT", "BINANCE", "TRXUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry(
        "DOGE", "FUT", "BINANCE", "DOGEUSDT.P", "PERPETUAL", "USDT"
    ),
    InstrumentRegistryEntry("ADA", "FUT", "BINANCE", "ADAUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry("BCH", "FUT", "BINANCE", "BCHUSDT.P", "PERPETUAL", "USDT"),
    InstrumentRegistryEntry(
        "LINK", "FUT", "BINANCE", "LINKUSDT.P", "PERPETUAL", "USDT"
    ),
)

STRICT_INSTRUMENT_REGISTRY = STRICT_INDEX_REGISTRY + STRICT_CRYPTO_REGISTRY
STRICT_PAIR_SYMBOLS = frozenset(
    entry.pair_symbol for entry in STRICT_INSTRUMENT_REGISTRY
)
REGISTRY_BY_PAIR_ASSET = {
    (entry.pair_symbol, entry.asset_class): entry
    for entry in STRICT_INSTRUMENT_REGISTRY
}
REGISTRY_BY_SOURCE = {
    (entry.source_exchange, entry.source_symbol): entry
    for entry in STRICT_INSTRUMENT_REGISTRY
}

_source_symbol_buckets: dict[str, list[InstrumentRegistryEntry]] = {}
for registry_entry in STRICT_INSTRUMENT_REGISTRY:
    _source_symbol_buckets.setdefault(registry_entry.source_symbol, []).append(
        registry_entry
    )
UNIQUE_REGISTRY_BY_SOURCE_SYMBOL = {
    source_symbol: entries[0]
    for source_symbol, entries in _source_symbol_buckets.items()
    if len(entries) == 1
}

CANONICAL_SYMBOL_ALIASES = {
    "CNXFINANCE": "FINNIFTY",
    "NIFTYFINSERVICE": "FINNIFTY",
    "NIFTYFINSERVICENS": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "NIFTYMIDSELECT": "MIDCPNIFTY",
    "NIFTYMIDSELECTNS": "MIDCPNIFTY",
    "NIFTYNXT50": "NIFTYNXT50",
    "NIFTYNEXT50": "NIFTYNXT50",
    "NIFTYJR": "NIFTYNXT50",
    "NSMIDCP": "NIFTYNXT50",
    "INDIAVIX": "INDIA_VIX",
    "SP500": "SPX500",
    "SPX500": "SPX500",
    "GSPC": "SPX500",
    "VIX": "VIX",
    "VIX9D": "VIX9D",
    "VIX3M": "VIX3M",
}


def _json_pick(payload: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _jsonish_pick(raw_payload: str, keys: tuple[str, ...]) -> str | None:
    payload_text = str(raw_payload)
    for key in keys:
        pattern = re.compile(
            rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            re.DOTALL,
        )
        match = pattern.search(payload_text)
        if match is None:
            continue
        value = match.group(1)
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            cleaned = value.replace('\\"', '"').strip()
            if cleaned:
                return cleaned
    return None


def _normalize_exchange(value: str | None) -> str | None:
    if value is None:
        return None
    token = EXCHANGE_CLEAN_RE.sub("", str(value).strip().upper())
    return token or None


def _normalize_source_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    token = SOURCE_SYMBOL_CLEAN_RE.sub("", str(value).strip().upper())
    return token or None


def _normalize_pair_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    token = PAIR_SYMBOL_CLEAN_RE.sub("", str(value).strip().upper())
    return token or None


def _split_exchange_prefix(value: str) -> tuple[str | None, str]:
    token = str(value).strip().upper()
    match = MARKET_PREFIX_RE.match(token)
    if match is None:
        return None, token
    return match.group("exchange"), match.group("symbol")


def _decode_symbol_payload(raw_payload: str) -> tuple[str, dict | None]:
    candidate = str(raw_payload).strip()
    for _ in range(2):
        if not candidate:
            return "plain", None
        if candidate[0] not in {"{", '"'}:
            return "plain", None
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            return "plain", None
        if isinstance(decoded, str):
            candidate = decoded.strip()
            continue
        if isinstance(decoded, dict):
            return "json", decoded
        return "plain", None
    return "plain", None


def _normalize_contract_kind(value: str | None) -> str | None:
    if value is None:
        return None
    token = PAIR_SYMBOL_CLEAN_RE.sub("_", str(value).strip().upper()).strip("_")
    if not token:
        return None
    if token in {"SPOT", "CASH", "INDEX"}:
        return "SPOT"
    if token in {"CONTINUOUS_FUTURE", "CONTINUOUS", "FUT_CONTINUOUS"}:
        return "CONTINUOUS_FUTURE"
    if token in {"DATED_FUTURE", "DATE_FUTURE", "EXPIRY_FUTURE"}:
        return "DATED_FUTURE"
    if token in {"PERPETUAL", "PERP", "PERPETUAL_FUTURE"}:
        return "PERPETUAL"
    if token in {"FUT", "FUTURE", "FUTURES"}:
        return "DATED_FUTURE"
    return token


def _normalize_market_type(value: str | None) -> str | None:
    if value is None:
        return None
    token = PAIR_SYMBOL_CLEAN_RE.sub("_", str(value).strip().upper()).strip("_")
    return token or None


def _asset_class_from_market_type(market_type: str | None) -> str | None:
    if market_type in {"FUTURES", "FUTURE"}:
        return "FUT"
    if market_type in {
        "CRYPTO",
        "INDEX",
        "STOCK",
        "FOREX",
        "FUND",
        "DR",
        "CFD",
        "BOND",
        "WARRANT",
        "STRUCTURED",
        "RIGHT",
    }:
        return "SPOT"
    return None


def _asset_class_from_contract_kind(contract_kind: str | None) -> str | None:
    if contract_kind is None:
        return None
    return "SPOT" if contract_kind == "SPOT" else "FUT"


def _infer_contract_signature(
    source_symbol: str,
) -> tuple[str, str | None, str]:
    if match := CONTINUOUS_FUTURE_RE.match(source_symbol):
        return "CONTINUOUS_FUTURE", None, match.group("root")
    if match := DATED_FUTURE_RE.match(source_symbol):
        return "DATED_FUTURE", match.group("expiry"), match.group("root")
    if match := PERPETUAL_RE.match(source_symbol):
        return "PERPETUAL", None, match.group("root")
    return "SPOT", None, source_symbol


def _infer_pair_and_quote(core_symbol: str) -> tuple[str, str | None]:
    for quote in COMMON_QUOTES:
        if core_symbol.endswith(quote) and len(core_symbol) > len(quote):
            pair_symbol = core_symbol[: -len(quote)]
            normalized_pair = _normalize_pair_symbol(pair_symbol)
            if normalized_pair:
                return normalized_pair, quote
    normalized_pair = _normalize_pair_symbol(core_symbol)
    return normalized_pair or "", None


def extract_symbol_payload(message: str) -> str | None:
    match = SYMBOL_PAYLOAD_RE.search(str(message))
    if match is None:
        return None
    payload = match.group(1).strip()
    return payload or None


@lru_cache(maxsize=8192)
def parse_symbol_payload(raw_payload: str) -> ParsedSymbolIdentity:
    payload_text = str(raw_payload).strip()
    if not payload_text:
        raise ValueError("Empty symbol payload.")

    payload_kind, payload_object = _decode_symbol_payload(payload_text)
    jsonish_payload = (
        payload_text
        if payload_object is None and payload_text.startswith("{")
        else None
    )

    explicit_exchange = None
    explicit_symbol = None
    explicit_pair_symbol = None
    explicit_contract_kind = None
    explicit_quote_asset = None
    explicit_expiry_code = None
    explicit_asset_class = None
    explicit_market_type = None

    if payload_object is not None:
        explicit_exchange = _json_pick(payload_object, JSON_EXCHANGE_KEYS)
        explicit_symbol = _json_pick(payload_object, JSON_SYMBOL_KEYS)
        explicit_pair_symbol = _json_pick(payload_object, JSON_PAIR_KEYS)
        explicit_contract_kind = _json_pick(payload_object, JSON_CONTRACT_KEYS)
        explicit_quote_asset = _json_pick(payload_object, JSON_QUOTE_KEYS)
        explicit_expiry_code = _json_pick(payload_object, JSON_EXPIRY_KEYS)
        explicit_asset_class = _json_pick(payload_object, JSON_ASSET_CLASS_KEYS)
        explicit_market_type = _json_pick(payload_object, JSON_MARKET_TYPE_KEYS)
    elif jsonish_payload is not None:
        payload_kind = "jsonish"
        explicit_exchange = _jsonish_pick(jsonish_payload, JSON_EXCHANGE_KEYS)
        explicit_symbol = _jsonish_pick(jsonish_payload, JSON_SYMBOL_KEYS)
        explicit_pair_symbol = _jsonish_pick(jsonish_payload, JSON_PAIR_KEYS)
        explicit_contract_kind = _jsonish_pick(jsonish_payload, JSON_CONTRACT_KEYS)
        explicit_quote_asset = _jsonish_pick(jsonish_payload, JSON_QUOTE_KEYS)
        explicit_expiry_code = _jsonish_pick(jsonish_payload, JSON_EXPIRY_KEYS)
        explicit_asset_class = _jsonish_pick(jsonish_payload, JSON_ASSET_CLASS_KEYS)
        explicit_market_type = _jsonish_pick(jsonish_payload, JSON_MARKET_TYPE_KEYS)
    else:
        explicit_symbol = payload_text

    if explicit_symbol is None:
        raise ValueError(f"Unable to locate source symbol in payload: {payload_text}")

    exchange_from_symbol, symbol_without_exchange = _split_exchange_prefix(
        explicit_symbol
    )
    source_exchange = _normalize_exchange(explicit_exchange or exchange_from_symbol)
    source_symbol = _normalize_source_symbol(symbol_without_exchange)
    if source_symbol is None:
        raise ValueError(
            f"Unable to normalize source symbol from payload: {payload_text}"
        )

    (
        derived_contract_kind,
        derived_expiry_code,
        pair_core_symbol,
    ) = _infer_contract_signature(source_symbol)
    market_type = _normalize_market_type(explicit_market_type)
    contract_kind = (
        _normalize_contract_kind(explicit_contract_kind) or derived_contract_kind
    )
    if derived_contract_kind in {"CONTINUOUS_FUTURE", "PERPETUAL"}:
        contract_kind = derived_contract_kind
    elif derived_contract_kind != "SPOT" and contract_kind == "SPOT":
        contract_kind = derived_contract_kind
    elif market_type in {"FUTURES", "FUTURE"} and contract_kind == "SPOT":
        contract_kind = "DATED_FUTURE"

    asset_class = _normalize_pair_symbol(explicit_asset_class)
    asset_class_from_contract = _asset_class_from_contract_kind(contract_kind)
    asset_class_from_market = _asset_class_from_market_type(market_type)
    if asset_class not in {"SPOT", "FUT"}:
        if contract_kind != "SPOT":
            asset_class = asset_class_from_contract
        else:
            asset_class = asset_class_from_market or asset_class_from_contract
    if asset_class is None:
        raise ValueError(f"Unable to infer asset_class from payload: {payload_text}")

    pair_symbol, inferred_quote_asset = _infer_pair_and_quote(pair_core_symbol)
    explicit_pair_symbol = _normalize_pair_symbol(explicit_pair_symbol)
    if explicit_pair_symbol is not None:
        pair_symbol = explicit_pair_symbol
    if not pair_symbol:
        raise ValueError(f"Unable to infer pair_symbol from payload: {payload_text}")

    quote_asset = _normalize_pair_symbol(explicit_quote_asset) or inferred_quote_asset
    expiry_code = _normalize_pair_symbol(explicit_expiry_code) or derived_expiry_code

    registry_entry = REGISTRY_BY_SOURCE.get((source_exchange, source_symbol))
    if registry_entry is None and source_exchange is None:
        unique_entry = UNIQUE_REGISTRY_BY_SOURCE_SYMBOL.get(source_symbol)
        if unique_entry is not None:
            source_exchange = unique_entry.source_exchange
            registry_entry = unique_entry

    expected_entry = REGISTRY_BY_PAIR_ASSET.get((pair_symbol, asset_class))
    if expected_entry is not None:
        if registry_entry is None:
            if source_symbol == expected_entry.source_symbol and (
                source_exchange is None
                or source_exchange == expected_entry.source_exchange
            ):
                source_exchange = expected_entry.source_exchange
                registry_entry = expected_entry
            else:
                received_source = (
                    f"{source_exchange}:{source_symbol}"
                    if source_exchange is not None
                    else source_symbol
                )
                raise ValueError(
                    "Registry-covered instrument requires an exact source match: "
                    f"{pair_symbol} {asset_class} expected "
                    f"{expected_entry.source_exchange}:{expected_entry.source_symbol}, "
                    f"received {received_source}."
                )

    if registry_entry is not None:
        pair_symbol = registry_entry.pair_symbol
        asset_class = registry_entry.asset_class
        source_exchange = registry_entry.source_exchange
        source_symbol = registry_entry.source_symbol
        contract_kind = registry_entry.contract_kind
        quote_asset = registry_entry.quote_asset or quote_asset
        expiry_code = None if contract_kind == "CONTINUOUS_FUTURE" else expiry_code

    payload_record = {
        "asset_class": asset_class,
        "contract_kind": contract_kind,
        "expiry_code": expiry_code,
        "market_type": market_type,
        "pair_symbol": pair_symbol,
        "payload_kind": payload_kind,
        "quote_asset": quote_asset,
        "raw_payload": payload_text,
        "registry_covered": (pair_symbol, asset_class) in REGISTRY_BY_PAIR_ASSET,
        "source_exchange": source_exchange,
        "source_symbol": source_symbol,
    }
    if payload_object is not None:
        payload_record["payload"] = payload_object

    return ParsedSymbolIdentity(
        base_symbol=source_symbol,
        pair_symbol=pair_symbol,
        asset_class=asset_class,
        source_exchange=source_exchange,
        source_symbol=source_symbol,
        symbol_payload_json=json.dumps(
            payload_record,
            sort_keys=True,
            separators=(",", ":"),
        ),
        contract_kind=contract_kind,
        quote_asset=quote_asset,
        expiry_code=expiry_code,
    )


@lru_cache(maxsize=8192)
def normalize_raw_symbol(raw_symbol: str) -> str:
    payload = str(raw_symbol).strip()
    if not payload:
        return ""
    try:
        return parse_symbol_payload(payload).base_symbol
    except ValueError:
        _, symbol = _split_exchange_prefix(payload)
        return _normalize_source_symbol(symbol) or ""


@lru_cache(maxsize=8192)
def infer_asset_class(raw_symbol: str) -> str:
    payload = str(raw_symbol).strip()
    if not payload:
        return "SPOT"
    try:
        return parse_symbol_payload(payload).asset_class
    except ValueError:
        normalized_symbol = normalize_raw_symbol(payload)
        contract_kind, _, _ = _infer_contract_signature(normalized_symbol)
        return _asset_class_from_contract_kind(contract_kind) or "SPOT"


@lru_cache(maxsize=8192)
def canonical_pair_symbol(raw_symbol: str, asset_class: str | None = None) -> str:
    payload = str(raw_symbol).strip()
    if not payload:
        return ""
    try:
        parsed = parse_symbol_payload(payload)
        if asset_class is None or parsed.asset_class == asset_class:
            parsed_pair = (
                _normalize_pair_symbol(parsed.pair_symbol) or parsed.pair_symbol
            )
            return CANONICAL_SYMBOL_ALIASES.get(parsed_pair, parsed_pair)
    except ValueError:
        pass

    normalized_symbol = normalize_raw_symbol(payload)
    normalized_alias_key = _normalize_pair_symbol(normalized_symbol)
    if normalized_alias_key in CANONICAL_SYMBOL_ALIASES:
        return CANONICAL_SYMBOL_ALIASES[normalized_alias_key]

    contract_kind, _, pair_core_symbol = _infer_contract_signature(normalized_symbol)
    pair_symbol, _ = _infer_pair_and_quote(pair_core_symbol)
    if pair_symbol:
        alias_pair = CANONICAL_SYMBOL_ALIASES.get(pair_symbol)
        return alias_pair or pair_symbol
    if asset_class == "FUT" and contract_kind != "SPOT":
        return _normalize_pair_symbol(pair_core_symbol) or normalized_symbol
    fallback_pair = _normalize_pair_symbol(normalized_symbol) or normalized_symbol
    return CANONICAL_SYMBOL_ALIASES.get(fallback_pair, fallback_pair)


def lookup_registry_entry(
    pair_symbol: str,
    asset_class: str,
) -> InstrumentRegistryEntry | None:
    return REGISTRY_BY_PAIR_ASSET.get(
        (_normalize_pair_symbol(pair_symbol), _normalize_pair_symbol(asset_class))
    )


def lookup_registry_source(
    source_exchange: str,
    source_symbol: str,
) -> InstrumentRegistryEntry | None:
    normalized_exchange = _normalize_exchange(source_exchange)
    normalized_symbol = _normalize_source_symbol(source_symbol)
    if normalized_exchange is None or normalized_symbol is None:
        return None
    return REGISTRY_BY_SOURCE.get((normalized_exchange, normalized_symbol))
