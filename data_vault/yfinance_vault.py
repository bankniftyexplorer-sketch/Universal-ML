from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from symbol_identity import canonical_pair_symbol
except ImportError:
    from data_vault.symbol_identity import canonical_pair_symbol

REALIZED_VOL_ANNUALIZATION = {"1H": 1512.0, "1D": 252.0, "1W": 52.0, "1M": 12.0, "3M": 4.0, "6M": 2.0, "12M": 1.0}
UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
MARKET_DNA_COLUMNS = (
    "base_symbol",
    "pair_symbol",
    "asset_class",
    "source_exchange",
    "source_symbol",
    "symbol_payload_json",
    "contract_kind",
    "quote_asset",
    "expiry_code",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_synthetic_vol",
    "realized_volatility",
)
MARKET_DNA_PRIMARY_KEY = (
    "source_exchange",
    "source_symbol",
    "asset_class",
    "timeframe",
    "timestamp",
)


@dataclass(frozen=True)
class YahooInstrument:
    base_symbol: str
    ticker: str
    quote_asset: str | None = None
    source_exchange: str = "YAHOO"


YAHOO_INSTRUMENTS: dict[str, YahooInstrument] = {
    "NIFTY": YahooInstrument("NIFTY", "^NSEI", "INR"),
    "BANKNIFTY": YahooInstrument("BANKNIFTY", "^NSEBANK", "INR"),
    "SENSEX": YahooInstrument("SENSEX", "^BSESN", "INR"),
    "FINNIFTY": YahooInstrument("FINNIFTY", "NIFTY_FIN_SERVICE.NS", "INR"),
    "CNXFINANCE": YahooInstrument("FINNIFTY", "NIFTY_FIN_SERVICE.NS", "INR"),
    "NIFTY_FIN_SERVICE.NS": YahooInstrument(
        "FINNIFTY", "NIFTY_FIN_SERVICE.NS", "INR"
    ),
    "MIDCPNIFTY": YahooInstrument("MIDCPNIFTY", "NIFTY_MID_SELECT.NS", "INR"),
    "NIFTY_MID_SELECT.NS": YahooInstrument(
        "MIDCPNIFTY", "NIFTY_MID_SELECT.NS", "INR"
    ),
    "NIFTYNXT50": YahooInstrument("NIFTYNXT50", "^NSMIDCP", "INR"),
    "NIFTYNEXT50": YahooInstrument("NIFTYNXT50", "^NSMIDCP", "INR"),
    "NIFTYJR": YahooInstrument("NIFTYNXT50", "^NSMIDCP", "INR"),
    "^NSMIDCP": YahooInstrument("NIFTYNXT50", "^NSMIDCP", "INR"),
    "SPX500": YahooInstrument("SPX500", "^GSPC", "USD"),
    "SP500": YahooInstrument("SPX500", "^GSPC", "USD"),
    "^GSPC": YahooInstrument("SPX500", "^GSPC", "USD"),
    "BTC": YahooInstrument("BTC", "BTC-USD", "USD"),
    "ETH": YahooInstrument("ETH", "ETH-USD", "USD"),
    "BNB": YahooInstrument("BNB", "BNB-USD", "USD"),
    "XRP": YahooInstrument("XRP", "XRP-USD", "USD"),
    "SOL": YahooInstrument("SOL", "SOL-USD", "USD"),
    "TRX": YahooInstrument("TRX", "TRX-USD", "USD"),
    "DOGE": YahooInstrument("DOGE", "DOGE-USD", "USD"),
    "ADA": YahooInstrument("ADA", "ADA-USD", "USD"),
    "BCH": YahooInstrument("BCH", "BCH-USD", "USD"),
    "LINK": YahooInstrument("LINK", "LINK-USD", "USD"),
}
CORE_WATCHLIST_SYMBOLS = (
    "NIFTY",
    "BANKNIFTY",
    "SENSEX",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "SPX500",
    "BTC",
)
FETCH_TIMEFRAMES = ("1D", "1H")
MACRO_TIMEFRAMES = ("1W", "1M", "3M", "6M", "12M")
CONTINUOUS_BASE_SYMBOLS = frozenset(
    {"BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "LINK"}
)
QUALITY_STATUS_PASS = "PASS"
QUALITY_STATUS_WARN = "WARN"
QUALITY_STATUS_FAIL = "FAIL"
QUALITY_STATUS_RANK = {
    QUALITY_STATUS_PASS: 0,
    QUALITY_STATUS_WARN: 1,
    QUALITY_STATUS_FAIL: 2,
}
QUALITY_MAX_WARN_MISSING_RATIO = 0.02
QUALITY_MAX_FAIL_MISSING_RATIO = 0.15
QUALITY_MAX_WARN_SYNTHETIC_VOL_RATIO = 0.25
QUALITY_WARN_STALENESS_HOURS = {"1D": 96.0, "1H": 48.0}
QUALITY_FAIL_STALENESS_HOURS = {"1D": 336.0, "1H": 168.0}
INCREMENTAL_LOOKBACK = {
    "1D": timedelta(days=180),
    "1H": timedelta(days=45),
}
FULL_REFRESH_CADENCE = {
    "1D": timedelta(days=30),
    "1H": timedelta(days=7),
}
CUSTOM_YAHOO_INSTRUMENTS_FILENAME = "custom_yahoo_instruments.json"
DEFAULT_AUTO_SYNC_CHECK_INTERVAL_SECONDS = 300.0
DEFAULT_AUTO_SYNC_MIN_GAP_SECONDS = 3600.0
DEFAULT_AUTO_SYNC_MAX_CPU_PERCENT = 20.0
DEFAULT_AUTO_SYNC_CPU_SAMPLE_SECONDS = 1.0
DEFAULT_AUTO_SYNC_MIN_DOWNLOAD_KBPS = 32.0
DEFAULT_AUTO_SYNC_PROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_AUTO_SYNC_PROBE_READ_LIMIT_BYTES = 131072
DEFAULT_AUTO_SYNC_PROBE_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
    "?interval=1d&range=10y"
)
MARKET_SYNC_QUALITY_COLUMNS = (
    "run_id",
    "synced_at",
    "base_symbol",
    "pair_symbol",
    "asset_class",
    "source_exchange",
    "source_symbol",
    "timeframe",
    "row_count",
    "first_timestamp",
    "last_timestamp",
    "duplicate_timestamps",
    "nonfinite_ohlc_rows",
    "nonpositive_price_rows",
    "synthetic_volume_rows",
    "zero_volume_rows",
    "expected_bar_count",
    "missing_bar_count",
    "missing_bar_ratio",
    "max_gap_bars",
    "quality_status",
    "audit_payload_json",
)


class YFinanceVault:
    def __init__(
        self,
        db_path: str = "ohlcv.db",
        custom_instruments_path: str | None = None,
    ) -> None:
        self.db_path = self._resolve_db_path(db_path)
        self.custom_instruments_path = self._resolve_custom_instruments_path(
            custom_instruments_path
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._configure_connection(self.conn)
        self._build_schema()
        self.custom_instrument_records = self._load_custom_instrument_records()
        self.custom_instruments = self._build_custom_instrument_lookup(
            self.custom_instrument_records
        )

    def _resolve_db_path(self, db_path: str) -> str:
        if os.path.isabs(db_path):
            return db_path

        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(module_dir)
        normalized = os.path.normpath(str(db_path).strip())
        if not normalized or normalized == ".":
            normalized = "ohlcv.db"

        if os.path.dirname(normalized):
            first_segment = normalized.split(os.sep, 1)[0]
            if first_segment == os.path.basename(module_dir):
                return os.path.join(repo_root, normalized)
            return os.path.abspath(normalized)

        return os.path.join(module_dir, normalized)

    def _resolve_custom_instruments_path(self, custom_instruments_path: str | None) -> str:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(module_dir)
        raw_path = custom_instruments_path or CUSTOM_YAHOO_INSTRUMENTS_FILENAME
        if os.path.isabs(raw_path):
            return raw_path

        normalized = os.path.normpath(str(raw_path).strip())
        if not normalized or normalized == ".":
            normalized = CUSTOM_YAHOO_INSTRUMENTS_FILENAME

        if os.path.dirname(normalized):
            first_segment = normalized.split(os.sep, 1)[0]
            if first_segment == os.path.basename(module_dir):
                return os.path.join(repo_root, normalized)
            return os.path.abspath(normalized)

        return os.path.join(module_dir, normalized)

    def _load_custom_instrument_records(self) -> dict[str, YahooInstrument]:
        path = self.custom_instruments_path
        if not os.path.exists(path):
            return {}

        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[!] Custom Yahoo instrument registry ignored: {exc}")
            return {}

        if not isinstance(payload, dict):
            print("[!] Custom Yahoo instrument registry ignored: root JSON must be an object.")
            return {}

        records: dict[str, YahooInstrument] = {}
        for alias, item in payload.items():
            if not isinstance(item, dict):
                print(f"[!] Skipping custom Yahoo instrument '{alias}': entry must be an object.")
                continue

            ticker = self._extract_yahoo_ticker(
                str(item.get("ticker") or item.get("source_symbol") or "").strip()
            )
            if not ticker:
                print(f"[!] Skipping custom Yahoo instrument '{alias}': missing ticker.")
                continue

            alias_key = canonical_pair_symbol(str(alias).strip(), asset_class="SPOT")
            if not alias_key:
                print(f"[!] Skipping custom Yahoo instrument '{alias}': invalid alias.")
                continue

            base_symbol = canonical_pair_symbol(
                str(item.get("base_symbol") or alias_key).strip(),
                asset_class="SPOT",
            ) or alias_key
            source_exchange = str(item.get("source_exchange") or "YAHOO").strip().upper()
            if source_exchange != "YAHOO":
                print(
                    f"[!] Skipping custom Yahoo instrument '{alias_key}': "
                    "only YAHOO source_exchange is supported."
                )
                continue

            quote_asset = item.get("quote_asset")
            if quote_asset is not None:
                quote_asset = str(quote_asset).strip().upper() or None

            records[alias_key] = YahooInstrument(
                base_symbol=base_symbol,
                ticker=ticker,
                quote_asset=quote_asset,
                source_exchange=source_exchange,
            )
        return records

    def _build_custom_instrument_lookup(
        self,
        records: dict[str, YahooInstrument],
    ) -> dict[str, YahooInstrument]:
        lookup: dict[str, YahooInstrument] = {}
        for alias_key, instrument in records.items():
            for key in (
                alias_key.upper(),
                instrument.base_symbol.upper(),
                instrument.ticker.upper(),
                f"YAHOO:{instrument.ticker.upper()}",
            ):
                lookup[key] = instrument
        return lookup

    def _persist_custom_instrument_records(self) -> None:
        directory = os.path.dirname(self.custom_instruments_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            alias_key: {
                "base_symbol": instrument.base_symbol,
                "ticker": instrument.ticker,
                "quote_asset": instrument.quote_asset,
                "source_exchange": instrument.source_exchange,
            }
            for alias_key, instrument in sorted(self.custom_instrument_records.items())
        }
        with open(self.custom_instruments_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _parse_custom_index_spec(self, spec: str) -> tuple[str, str]:
        token = str(spec).strip()
        if not token:
            raise ValueError("Custom index spec cannot be empty.")
        alias, separator, ticker = token.partition("=")
        alias_key = canonical_pair_symbol(alias.strip(), asset_class="SPOT")
        ticker_value = self._extract_yahoo_ticker(ticker.strip())
        if not separator or not alias_key or not ticker_value:
            raise ValueError(
                "Custom index specs must look like BASE_SYMBOL=YAHOO_TICKER, "
                "for example DAX=^GDAXI."
            )
        return alias_key, ticker_value

    def register_custom_index(
        self,
        spec: str,
        *,
        persist: bool = True,
    ) -> YahooInstrument:
        alias_key, ticker_value = self._parse_custom_index_spec(spec)
        instrument = YahooInstrument(
            base_symbol=alias_key,
            ticker=ticker_value,
            quote_asset=self._infer_quote_asset(ticker_value),
        )

        builtin = YAHOO_INSTRUMENTS.get(alias_key)
        if builtin is not None and (
            builtin.base_symbol != instrument.base_symbol
            or builtin.ticker.upper() != instrument.ticker.upper()
        ):
            raise ValueError(
                f"Cannot overwrite built-in Yahoo mapping for {alias_key}: "
                f"{builtin.ticker} is already registered."
            )

        self.custom_instrument_records[alias_key] = instrument
        self.custom_instruments = self._build_custom_instrument_lookup(
            self.custom_instrument_records
        )
        if persist:
            self._persist_custom_instrument_records()
        return instrument

    def list_available_symbols(self) -> list[tuple[str, str, str]]:
        rows: dict[str, tuple[str, str, str]] = {}

        def _add_row(alias: str, instrument: YahooInstrument, source: str) -> None:
            alias_key = canonical_pair_symbol(alias, asset_class="SPOT") or alias.upper()
            rows[alias_key] = (alias_key, instrument.ticker, source)

        for alias, instrument in YAHOO_INSTRUMENTS.items():
            alias_key = canonical_pair_symbol(alias, asset_class="SPOT") or alias.upper()
            if alias_key != alias.upper():
                continue
            _add_row(alias_key, instrument, "builtin")

        for alias_key, instrument in self.custom_instrument_records.items():
            _add_row(alias_key, instrument, "custom")

        return [rows[key] for key in sorted(rows)]

    def _cpu_percent_from_proc_stat(self, sample_seconds: float) -> float:
        sample_seconds = max(float(sample_seconds), 0.1)

        def _read_snapshot() -> tuple[int, int] | None:
            try:
                with open("/proc/stat", encoding="utf-8") as handle:
                    columns = handle.readline().strip().split()
            except OSError:
                return None
            if not columns or columns[0] != "cpu":
                return None
            try:
                values = [int(value) for value in columns[1:]]
            except ValueError:
                return None
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle

        first = _read_snapshot()
        if first is None:
            cpu_count = os.cpu_count() or 1
            load_avg = os.getloadavg()[0]
            return max(0.0, min((load_avg / cpu_count) * 100.0, 100.0))

        time.sleep(sample_seconds)
        second = _read_snapshot()
        if second is None:
            cpu_count = os.cpu_count() or 1
            load_avg = os.getloadavg()[0]
            return max(0.0, min((load_avg / cpu_count) * 100.0, 100.0))

        total_delta = max(second[0] - first[0], 1)
        idle_delta = max(second[1] - first[1], 0)
        busy_delta = max(total_delta - idle_delta, 0)
        return max(0.0, min((busy_delta / total_delta) * 100.0, 100.0))

    def _measure_yahoo_download_kbps(
        self,
        *,
        probe_url: str,
        timeout_seconds: float,
        read_limit_bytes: int = DEFAULT_AUTO_SYNC_PROBE_READ_LIMIT_BYTES,
    ) -> float:
        request = urllib.request.Request(
            probe_url,
            headers={
                "User-Agent": "Universal-ML/1.0 (+https://query1.finance.yahoo.com/)"
            },
        )
        start = time.perf_counter()
        bytes_read = 0
        with urllib.request.urlopen(request, timeout=max(float(timeout_seconds), 1.0)) as response:
            while bytes_read < read_limit_bytes:
                chunk = response.read(min(65536, read_limit_bytes - bytes_read))
                if not chunk:
                    break
                bytes_read += len(chunk)

        elapsed = max(time.perf_counter() - start, 1e-6)
        return (bytes_read / 1024.0) / elapsed

    def resource_gate_snapshot(
        self,
        *,
        max_cpu_percent: float,
        cpu_sample_seconds: float,
        min_download_kbps: float,
        probe_url: str,
        probe_timeout_seconds: float,
    ) -> dict[str, object]:
        snapshot = {
            "checked_at": self._sync_now_utc(),
            "cpu_percent": float("nan"),
            "download_kbps": float("nan"),
            "cpu_ok": True,
            "network_ok": True,
            "ready": False,
            "network_error": None,
            "probe_url": probe_url,
        }

        cpu_percent = self._cpu_percent_from_proc_stat(cpu_sample_seconds)
        snapshot["cpu_percent"] = cpu_percent
        snapshot["cpu_ok"] = cpu_percent <= float(max_cpu_percent)

        if min_download_kbps > 0:
            try:
                download_kbps = self._measure_yahoo_download_kbps(
                    probe_url=probe_url,
                    timeout_seconds=probe_timeout_seconds,
                )
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
                snapshot["download_kbps"] = 0.0
                snapshot["network_ok"] = False
                snapshot["network_error"] = str(exc)
            else:
                snapshot["download_kbps"] = download_kbps
                snapshot["network_ok"] = download_kbps >= float(min_download_kbps)

        snapshot["ready"] = bool(snapshot["cpu_ok"] and snapshot["network_ok"])
        return snapshot

    def print_resource_gate_snapshot(
        self,
        *,
        max_cpu_percent: float,
        cpu_sample_seconds: float,
        min_download_kbps: float,
        probe_url: str,
        probe_timeout_seconds: float,
    ) -> dict[str, object]:
        snapshot = self.resource_gate_snapshot(
            max_cpu_percent=max_cpu_percent,
            cpu_sample_seconds=cpu_sample_seconds,
            min_download_kbps=min_download_kbps,
            probe_url=probe_url,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        print("Yahoo auto-sync resource gate snapshot")
        print(
            f"  checked_at    : {snapshot['checked_at']}\n"
            f"  cpu_percent   : {snapshot['cpu_percent']:.1f}\n"
            f"  cpu_ok        : {snapshot['cpu_ok']}\n"
            f"  download_kbps : {snapshot['download_kbps']:.1f}\n"
            f"  network_ok    : {snapshot['network_ok']}\n"
            f"  ready         : {snapshot['ready']}"
        )
        if snapshot["network_error"]:
            print(f"  network_error : {snapshot['network_error']}")
        print(f"  probe_url     : {snapshot['probe_url']}")
        return snapshot

    def auto_sync(
        self,
        base_symbols: Iterable[str] | None = None,
        pause_seconds: float = 1.0,
        *,
        force_full_refresh: bool = False,
        check_interval_seconds: float = DEFAULT_AUTO_SYNC_CHECK_INTERVAL_SECONDS,
        min_sync_gap_seconds: float = DEFAULT_AUTO_SYNC_MIN_GAP_SECONDS,
        max_cpu_percent: float = DEFAULT_AUTO_SYNC_MAX_CPU_PERCENT,
        cpu_sample_seconds: float = DEFAULT_AUTO_SYNC_CPU_SAMPLE_SECONDS,
        min_download_kbps: float = DEFAULT_AUTO_SYNC_MIN_DOWNLOAD_KBPS,
        probe_url: str = DEFAULT_AUTO_SYNC_PROBE_URL,
        probe_timeout_seconds: float = DEFAULT_AUTO_SYNC_PROBE_TIMEOUT_SECONDS,
    ) -> None:
        check_interval_seconds = max(float(check_interval_seconds), 1.0)
        min_sync_gap_seconds = max(float(min_sync_gap_seconds), 0.0)
        last_attempt_started_monotonic: float | None = None

        print("Starting resource-gated Yahoo auto-sync supervisor.")
        print(
            "  thresholds: "
            f"cpu<={max_cpu_percent:.1f}% | "
            f"download>={min_download_kbps:.1f} KB/s | "
            f"min_gap={min_sync_gap_seconds:.0f}s | "
            f"check_every={check_interval_seconds:.0f}s"
        )

        while True:
            now_monotonic = time.monotonic()
            if last_attempt_started_monotonic is not None:
                seconds_since_last_attempt = now_monotonic - last_attempt_started_monotonic
                if seconds_since_last_attempt < min_sync_gap_seconds:
                    remaining = min_sync_gap_seconds - seconds_since_last_attempt
                    print(
                        "  [auto] Cooling down before next sync window: "
                        f"{remaining:.0f}s remaining."
                    )
                    time.sleep(min(check_interval_seconds, max(remaining, 1.0)))
                    continue

            snapshot = self.resource_gate_snapshot(
                max_cpu_percent=max_cpu_percent,
                cpu_sample_seconds=cpu_sample_seconds,
                min_download_kbps=min_download_kbps,
                probe_url=probe_url,
                probe_timeout_seconds=probe_timeout_seconds,
            )
            if not snapshot["ready"]:
                network_detail = ""
                if snapshot["network_error"]:
                    network_detail = f" | network_error={snapshot['network_error']}"
                print(
                    "  [auto] Waiting for healthy resource window: "
                    f"cpu={snapshot['cpu_percent']:.1f}% "
                    f"(limit {max_cpu_percent:.1f}%) | "
                    f"download={snapshot['download_kbps']:.1f} KB/s "
                    f"(floor {min_download_kbps:.1f})"
                    f"{network_detail}"
                )
                time.sleep(check_interval_seconds)
                continue

            print(
                "  [auto] Resource window open. Launching sync: "
                f"cpu={snapshot['cpu_percent']:.1f}% | "
                f"download={snapshot['download_kbps']:.1f} KB/s"
            )
            last_attempt_started_monotonic = time.monotonic()
            try:
                self.sync(
                    base_symbols=base_symbols,
                    pause_seconds=pause_seconds,
                    force_full_refresh=force_full_refresh,
                )
            except Exception as exc:
                print(f"  [auto] Sync cycle failed: {exc}")
            else:
                print(f"  [auto] Sync cycle finished at {self._sync_now_utc()}.")


    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA mmap_size=1000000000;")
        conn.execute("PRAGMA cache_size=-200000;")

    def _market_dna_exists(self, cursor: sqlite3.Cursor) -> bool:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'market_dna'"
        ).fetchone()
        return row is not None

    def _market_sync_quality_exists(self, cursor: sqlite3.Cursor) -> bool:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'market_sync_quality'"
        ).fetchone()
        return row is not None

    def _market_dna_table_info(self, cursor: sqlite3.Cursor) -> list[tuple]:
        return cursor.execute("PRAGMA table_info(market_dna)").fetchall()

    def _market_dna_columns(self, cursor: sqlite3.Cursor) -> set[str]:
        return {row[1] for row in self._market_dna_table_info(cursor)}

    def _market_dna_pk_columns(self, cursor: sqlite3.Cursor) -> tuple[str, ...]:
        pk_rows = [
            (row[5], row[1])
            for row in self._market_dna_table_info(cursor)
            if row[5] > 0
        ]
        return tuple(name for _, name in sorted(pk_rows))

    def _create_market_dna_table(
        self,
        cursor: sqlite3.Cursor,
        table_name: str = "market_dna",
    ) -> None:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                base_symbol TEXT,
                pair_symbol TEXT,
                asset_class TEXT,
                source_exchange TEXT,
                source_symbol TEXT,
                symbol_payload_json TEXT,
                contract_kind TEXT,
                quote_asset TEXT,
                expiry_code TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                is_synthetic_vol BOOLEAN,
                realized_volatility REAL,
                PRIMARY KEY (
                    source_exchange, source_symbol, asset_class, timeframe, timestamp
                )
            )
            """
        )

    def _create_market_sync_quality_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS market_sync_quality (
                run_id TEXT,
                synced_at DATETIME,
                base_symbol TEXT,
                pair_symbol TEXT,
                asset_class TEXT,
                source_exchange TEXT,
                source_symbol TEXT,
                timeframe TEXT,
                row_count INTEGER,
                first_timestamp DATETIME,
                last_timestamp DATETIME,
                duplicate_timestamps INTEGER,
                nonfinite_ohlc_rows INTEGER,
                nonpositive_price_rows INTEGER,
                synthetic_volume_rows INTEGER,
                zero_volume_rows INTEGER,
                expected_bar_count INTEGER,
                missing_bar_count INTEGER,
                missing_bar_ratio REAL,
                max_gap_bars REAL,
                quality_status TEXT,
                audit_payload_json TEXT,
                PRIMARY KEY (run_id, base_symbol, timeframe)
            )
            """
        )

    def _rebuild_market_dna_table(self, cursor: sqlite3.Cursor) -> None:
        legacy_table = "market_dna_legacy_migration"
        cursor.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        cursor.execute(f"ALTER TABLE market_dna RENAME TO {legacy_table}")
        self._create_market_dna_table(cursor, "market_dna")

        legacy_columns = {
            row[1]
            for row in cursor.execute(f"PRAGMA table_info({legacy_table})").fetchall()
        }
        copy_columns = [col for col in MARKET_DNA_COLUMNS if col in legacy_columns]
        column_csv = ", ".join(copy_columns)
        cursor.execute(
            f"""
            INSERT INTO market_dna ({column_csv})
            SELECT {column_csv}
            FROM {legacy_table}
            """
        )
        cursor.execute(f"DROP TABLE {legacy_table}")

    def _build_schema(self) -> None:
        cursor = self.conn.cursor()
        if not self._market_dna_exists(cursor):
            self._create_market_dna_table(cursor)
        else:
            existing_cols = self._market_dna_columns(cursor)
            pk_cols = self._market_dna_pk_columns(cursor)
            needs_rebuild = (
                not set(MARKET_DNA_COLUMNS).issubset(existing_cols)
                or pk_cols != MARKET_DNA_PRIMARY_KEY
            )
            if needs_rebuild:
                self._rebuild_market_dna_table(cursor)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ml_inference
            ON market_dna (base_symbol, asset_class, timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pair_inference
            ON market_dna (pair_symbol, asset_class, timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source_identity
            ON market_dna (source_exchange, source_symbol, asset_class, timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pair_source_identity
            ON market_dna (
                pair_symbol, asset_class, source_exchange, source_symbol, timestamp
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_ledger (
                timestamp DATETIME,
                base_symbol TEXT,
                direction TEXT,
                conf_score REAL,
                entry_price REAL,
                exit_price REAL,
                pnl_r REAL,
                win_loss_target INTEGER
            )
            """
        )
        if not self._market_sync_quality_exists(cursor):
            self._create_market_sync_quality_table(cursor)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_sync_quality_lookup
            ON market_sync_quality (
                pair_symbol, asset_class, source_exchange, source_symbol, timeframe, synced_at
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_sync_quality_symbol
            ON market_sync_quality (base_symbol, timeframe, synced_at)
            """
        )
        self.conn.commit()

    def log_trade_result(self, data_packet: dict) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO performance_ledger (
                timestamp, base_symbol, direction, conf_score,
                entry_price, exit_price, pnl_r, win_loss_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data_packet.get("timestamp"),
                data_packet.get("base_symbol"),
                data_packet.get("direction"),
                data_packet.get("conf_score"),
                data_packet.get("entry_price"),
                data_packet.get("exit_price"),
                data_packet.get("pnl_r"),
                data_packet.get("win_loss_target"),
            ),
        )
        self.conn.commit()

    def log_bulk_trades(self, trade_list: list[dict]) -> None:
        if not trade_list:
            return
        cursor = self.conn.cursor()
        rows = [
            (
                trade.get("timestamp"),
                trade.get("base_symbol"),
                trade.get("direction"),
                trade.get("conf_score"),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("pnl_r"),
                trade.get("win_loss_target"),
            )
            for trade in trade_list
        ]
        cursor.executemany(
            """
            INSERT INTO performance_ledger (
                timestamp, base_symbol, direction, conf_score,
                entry_price, exit_price, pnl_r, win_loss_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_shadow_train_set(self, base_symbol: str, days: int = 10000) -> pd.DataFrame:
        query = f"""
            SELECT p.*, m.open, m.high, m.low, m.close, m.volume
            FROM performance_ledger p
            LEFT JOIN market_dna m
              ON p.base_symbol = COALESCE(m.pair_symbol, m.base_symbol)
              AND datetime(p.timestamp) = datetime(m.timestamp)
              AND m.timeframe = '1H'
              AND m.asset_class = 'SPOT'
            WHERE p.base_symbol = ?
              AND p.timestamp >= datetime('now', '-{days} days')
            ORDER BY p.timestamp ASC
        """
        return pd.read_sql_query(query, self.conn, params=(base_symbol.upper(),))

    def _resolve_instrument(self, base_symbol: str) -> YahooInstrument:
        raw_symbol = str(base_symbol).strip()
        if not raw_symbol:
            raise ValueError("Yahoo Finance symbol cannot be empty.")

        key = raw_symbol.upper()
        instrument = YAHOO_INSTRUMENTS.get(key)
        if instrument is not None:
            return instrument

        custom_instrument = self.custom_instruments.get(key)
        if custom_instrument is not None:
            return custom_instrument

        yahoo_ticker = self._extract_yahoo_ticker(raw_symbol)
        custom_instrument = self.custom_instruments.get(yahoo_ticker.upper())
        if custom_instrument is not None:
            return custom_instrument

        pair_symbol = canonical_pair_symbol(raw_symbol, asset_class="SPOT")
        mapped_instrument = YAHOO_INSTRUMENTS.get(pair_symbol)
        if mapped_instrument is not None:
            return mapped_instrument
        mapped_custom_instrument = self.custom_instruments.get(pair_symbol)
        if mapped_custom_instrument is not None:
            return mapped_custom_instrument

        return YahooInstrument(
            base_symbol=pair_symbol or key,
            ticker=yahoo_ticker,
            quote_asset=self._infer_quote_asset(yahoo_ticker),
        )

    def _extract_yahoo_ticker(self, raw_symbol: str) -> str:
        token = str(raw_symbol).strip()
        upper = token.upper()
        if upper.startswith("YAHOO:"):
            return token.split(":", 1)[1].strip()
        return token

    def _infer_quote_asset(self, ticker: str) -> str | None:
        token = str(ticker).strip().upper()
        if "-" in token:
            quote = token.rsplit("-", 1)[-1].strip()
            return quote or None
        if token.endswith("=X"):
            pair = token[:-2]
            if len(pair) >= 6 and pair.isalpha():
                return pair[-3:]
        return None

    def _copy_attrs(self, src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
        dst.attrs.update(dict(getattr(src, "attrs", {})))
        return dst

    def _sync_now_utc(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _timestamp_text(self, value: pd.Timestamp | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime(UTC_TIMESTAMP_FORMAT)

    def _is_continuous_market(self, instrument: YahooInstrument) -> bool:
        return instrument.base_symbol in CONTINUOUS_BASE_SYMBOLS

    def _quality_payload_dict(self, audit_payload_json: str | None) -> dict[str, object]:
        if not audit_payload_json:
            return {}
        try:
            payload = json.loads(audit_payload_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _quality_status_floor(self, current_status: str, minimum_status: str) -> str:
        current_rank = QUALITY_STATUS_RANK.get(current_status, 0)
        minimum_rank = QUALITY_STATUS_RANK.get(minimum_status, 0)
        return current_status if current_rank >= minimum_rank else minimum_status

    def _staleness_hours(
        self,
        *,
        synced_at: str,
        last_timestamp: pd.Timestamp | None,
    ) -> float:
        if last_timestamp is None or pd.isna(last_timestamp):
            return float("inf")
        synced_ts = pd.Timestamp(synced_at)
        if synced_ts.tzinfo is None:
            synced_ts = synced_ts.tz_localize("UTC")
        else:
            synced_ts = synced_ts.tz_convert("UTC")

        last_ts = pd.Timestamp(last_timestamp)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        return max(float((synced_ts - last_ts) / pd.Timedelta(hours=1)), 0.0)

    def _quality_status_for_metrics(
        self,
        *,
        timeframe: str,
        row_count: int,
        duplicate_timestamps: int,
        nonfinite_ohlc_rows: int,
        nonpositive_price_rows: int,
        missing_bar_ratio: float,
        synthetic_volume_ratio: float,
        staleness_hours: float,
    ) -> str:
        if row_count <= 0:
            return QUALITY_STATUS_FAIL
        if duplicate_timestamps > 0 or nonfinite_ohlc_rows > 0 or nonpositive_price_rows > 0:
            return QUALITY_STATUS_FAIL
        if missing_bar_ratio >= QUALITY_MAX_FAIL_MISSING_RATIO:
            return QUALITY_STATUS_FAIL
        if staleness_hours >= QUALITY_FAIL_STALENESS_HOURS.get(timeframe, float("inf")):
            return QUALITY_STATUS_FAIL
        if (
            missing_bar_ratio >= QUALITY_MAX_WARN_MISSING_RATIO
            or synthetic_volume_ratio >= QUALITY_MAX_WARN_SYNTHETIC_VOL_RATIO
            or staleness_hours >= QUALITY_WARN_STALENESS_HOURS.get(timeframe, float("inf"))
        ):
            return QUALITY_STATUS_WARN
        return QUALITY_STATUS_PASS

    def _load_existing_timeframe_frame(
        self,
        *,
        instrument: YahooInstrument,
        timeframe: str,
    ) -> pd.DataFrame:
        query = f"""
            SELECT {", ".join(MARKET_DNA_COLUMNS)}
            FROM market_dna
            WHERE pair_symbol = ?
              AND asset_class = 'SPOT'
              AND source_exchange = ?
              AND source_symbol = ?
              AND timeframe = ?
            ORDER BY timestamp ASC
        """
        existing = pd.read_sql_query(
            query,
            self.conn,
            params=(
                instrument.base_symbol,
                instrument.source_exchange,
                instrument.ticker,
                timeframe,
            ),
        )
        if existing.empty:
            return self._empty_history_frame()

        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"],
            format=UTC_TIMESTAMP_FORMAT,
            exact=True,
            utc=True,
        )
        existing["is_synthetic_vol"] = existing["is_synthetic_vol"].fillna(False).astype(bool)
        existing["realized_volatility"] = pd.to_numeric(
            existing["realized_volatility"], errors="coerce"
        )
        existing.attrs["source_timezone"] = "UTC"

        latest_quality = self._latest_quality_snapshot(
            instrument=instrument,
            timeframe=timeframe,
        )
        if latest_quality is not None:
            payload = self._quality_payload_dict(latest_quality.get("audit_payload_json"))
            existing.attrs["source_timezone"] = str(payload.get("source_timezone", "UTC"))
        return existing.loc[:, list(MARKET_DNA_COLUMNS)]

    def _latest_quality_snapshot(
        self,
        *,
        instrument: YahooInstrument,
        timeframe: str,
    ) -> dict[str, object] | None:
        row = self.conn.execute(
            """
            SELECT run_id, synced_at, row_count, first_timestamp, last_timestamp,
                   quality_status, audit_payload_json
            FROM market_sync_quality
            WHERE pair_symbol = ?
              AND asset_class = 'SPOT'
              AND source_exchange = ?
              AND source_symbol = ?
              AND timeframe = ?
            ORDER BY synced_at DESC, run_id DESC
            LIMIT 1
            """,
            (
                instrument.base_symbol,
                instrument.source_exchange,
                instrument.ticker,
                timeframe,
            ),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "synced_at": row[1],
            "row_count": row[2],
            "first_timestamp": row[3],
            "last_timestamp": row[4],
            "quality_status": row[5],
            "audit_payload_json": row[6],
        }

    def _should_force_full_refresh(
        self,
        *,
        timeframe: str,
        existing_df: pd.DataFrame,
        latest_quality: dict[str, object] | None,
        synced_at: str,
        force_full_refresh: bool = False,
    ) -> tuple[bool, str]:
        if force_full_refresh:
            return True, "operator_forced_full_refresh"
        if existing_df.empty:
            return True, "bootstrap"
        if latest_quality is None:
            return True, "repair_missing_quality_history"
        if latest_quality.get("quality_status") == QUALITY_STATUS_FAIL:
            return True, "repair_last_quality_fail"

        synced_ts = pd.Timestamp(synced_at)
        quality_synced_at = pd.Timestamp(str(latest_quality["synced_at"]))
        if quality_synced_at.tzinfo is None:
            quality_synced_at = quality_synced_at.tz_localize("UTC")
        else:
            quality_synced_at = quality_synced_at.tz_convert("UTC")
        if synced_ts.tzinfo is None:
            synced_ts = synced_ts.tz_localize("UTC")
        else:
            synced_ts = synced_ts.tz_convert("UTC")

        if synced_ts - quality_synced_at >= FULL_REFRESH_CADENCE[timeframe]:
            return True, "scheduled_full_refresh"
        return False, "incremental_refresh"

    def _fetch_start_timestamp(
        self,
        *,
        timeframe: str,
        existing_df: pd.DataFrame,
    ) -> pd.Timestamp | None:
        if existing_df.empty:
            return None
        latest_timestamp = pd.to_datetime(existing_df["timestamp"], utc=True).max()
        if pd.isna(latest_timestamp):
            return None
        return latest_timestamp - pd.Timedelta(INCREMENTAL_LOOKBACK[timeframe])

    def _merge_market_frames(
        self,
        existing_df: pd.DataFrame,
        fetched_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if existing_df.empty:
            merged = fetched_df.copy()
        elif fetched_df.empty:
            merged = existing_df.copy()
        else:
            merged = pd.concat([existing_df, fetched_df], ignore_index=True)
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
            merged = merged.dropna(subset=["timestamp"])
            merged = merged.sort_values("timestamp")
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last").reset_index(
                drop=True
            )

        merged.attrs["source_timezone"] = str(
            fetched_df.attrs.get(
                "source_timezone",
                existing_df.attrs.get("source_timezone", "UTC"),
            )
        )
        return merged.loc[:, list(MARKET_DNA_COLUMNS)]

    def _decorate_quality_row(
        self,
        *,
        quality_row: dict[str, object],
        fetch_mode: str,
        fetch_status: str,
        existing_row_count: int,
        fetched_row_count: int,
        merged_row_count: int,
        retained_row_count: int,
        minimum_status: str | None = None,
        extra_payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = self._quality_payload_dict(str(quality_row.get("audit_payload_json", "")))
        payload.update(
            {
                "existing_row_count": int(existing_row_count),
                "fetch_mode": fetch_mode,
                "fetch_status": fetch_status,
                "fetched_row_count": int(fetched_row_count),
                "merged_row_count": int(merged_row_count),
                "retained_row_count": int(retained_row_count),
            }
        )
        if extra_payload:
            payload.update(extra_payload)
        quality_row["audit_payload_json"] = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        if minimum_status is not None:
            quality_row["quality_status"] = self._quality_status_floor(
                str(quality_row["quality_status"]),
                minimum_status,
            )
        return quality_row

    def _audit_continuous_gap_metrics(
        self,
        timestamps: pd.DatetimeIndex,
        *,
        timeframe: str,
    ) -> tuple[int, int, float]:
        if len(timestamps) <= 1:
            return len(timestamps), 0, 0.0

        if timeframe == "1D":
            expected_count = int(
                (timestamps[-1].normalize() - timestamps[0].normalize()) / pd.Timedelta(days=1)
            ) + 1
            gap_units = (
                (timestamps.to_series().diff().dropna() / pd.Timedelta(days=1))
                .round()
                .astype(int)
            )
        elif timeframe == "1H":
            expected_count = int((timestamps[-1] - timestamps[0]) / pd.Timedelta(hours=1)) + 1
            gap_units = (
                (timestamps.to_series().diff().dropna() / pd.Timedelta(hours=1))
                .round()
                .astype(int)
            )
        else:
            expected_count = len(timestamps)
            gap_units = pd.Series(dtype=int)

        missing_bar_count = int(np.maximum(gap_units - 1, 0).sum()) if not gap_units.empty else 0
        max_gap_bars = float(np.maximum(gap_units - 1, 0).max()) if not gap_units.empty else 0.0
        return expected_count, missing_bar_count, max_gap_bars

    def _audit_session_daily_gap_metrics(
        self,
        local_timestamps: pd.DatetimeIndex,
    ) -> tuple[int, int, float]:
        if len(local_timestamps) <= 1:
            return len(local_timestamps), 0, 0.0

        session_dates = pd.Index(local_timestamps.normalize().unique()).sort_values()
        if len(session_dates) <= 1:
            return len(local_timestamps), 0, 0.0

        gap_days = session_dates.to_series().diff().dropna() / pd.Timedelta(days=1)
        excess_missing = np.maximum(gap_days.to_numpy(dtype=float) - 3.0, 0.0)
        missing_bar_count = int(np.round(excess_missing.sum()))
        max_gap_bars = float(excess_missing.max()) if len(excess_missing) else 0.0
        return int(len(session_dates) + missing_bar_count), missing_bar_count, max_gap_bars

    def _audit_session_hourly_gap_metrics(
        self,
        local_hourly_timestamps: pd.DatetimeIndex,
        local_daily_timestamps: pd.DatetimeIndex | None,
    ) -> tuple[int, int, float, dict]:
        hourly_df = pd.DataFrame({"timestamp": local_hourly_timestamps})
        hourly_df["session_date"] = hourly_df["timestamp"].dt.normalize()
        hourly_df["weekday"] = hourly_df["timestamp"].dt.weekday
        hourly_df["slot"] = hourly_df["timestamp"].dt.hour * 60 + hourly_df["timestamp"].dt.minute

        if hourly_df.empty:
            return 0, 0, 0.0, {"coverage_days": 0, "session_slots": 0}

        observed_dates = hourly_df["session_date"].drop_duplicates().sort_values()
        if local_daily_timestamps is not None and len(local_daily_timestamps) > 0:
            daily_dates = pd.Index(local_daily_timestamps.normalize().unique()).sort_values()
            expected_dates = daily_dates[
                (daily_dates >= observed_dates.min()) & (daily_dates <= observed_dates.max())
            ]
        else:
            expected_dates = pd.Index(observed_dates)

        dates_per_weekday = (
            hourly_df.groupby("weekday")["session_date"].nunique().to_dict()
        )
        slot_presence = (
            hourly_df.groupby(["weekday", "slot"])["session_date"].nunique().to_dict()
        )
        global_slot_presence = hourly_df.groupby("slot")["session_date"].nunique()
        global_threshold = max(3, int(np.ceil(len(observed_dates) * 0.4)))
        global_slots = sorted(
            int(slot)
            for slot, count in global_slot_presence.items()
            if int(count) >= global_threshold
        )

        weekday_slots: dict[int, list[int]] = {}
        for weekday, weekday_count in dates_per_weekday.items():
            threshold = max(3, int(np.ceil(int(weekday_count) * 0.4)))
            slots = sorted(
                int(slot)
                for (wd, slot), count in slot_presence.items()
                if int(wd) == int(weekday) and int(count) >= threshold
            )
            weekday_slots[int(weekday)] = slots or global_slots

        actual_slots_by_date = {
            pd.Timestamp(session_date): set(group["slot"].astype(int).tolist())
            for session_date, group in hourly_df.groupby("session_date")
        }

        missing_bar_count = 0
        expected_bar_count = 0
        max_gap_bars = 0.0
        for expected_date in expected_dates:
            weekday = int(pd.Timestamp(expected_date).weekday())
            expected_slots = weekday_slots.get(weekday, global_slots)
            if not expected_slots:
                continue
            expected_bar_count += len(expected_slots)
            actual_slots = actual_slots_by_date.get(pd.Timestamp(expected_date), set())
            missing_slots = [slot for slot in expected_slots if slot not in actual_slots]
            missing_bar_count += len(missing_slots)
            if missing_slots:
                sorted_missing = sorted(missing_slots)
                longest_run = 1
                current_run = 1
                for prev_slot, next_slot in zip(sorted_missing, sorted_missing[1:]):
                    if next_slot - prev_slot == 60:
                        current_run += 1
                    else:
                        longest_run = max(longest_run, current_run)
                        current_run = 1
                longest_run = max(longest_run, current_run)
                max_gap_bars = max(max_gap_bars, float(longest_run))

        payload = {
            "coverage_days": int(len(expected_dates)),
            "session_slots": int(len(global_slots)),
        }
        return expected_bar_count, missing_bar_count, max_gap_bars, payload

    def _audit_fetched_frame(
        self,
        df: pd.DataFrame,
        *,
        instrument: YahooInstrument,
        timeframe: str,
        run_id: str,
        synced_at: str,
        daily_reference: pd.DataFrame | None = None,
        payload_extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        timestamps_utc = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True, errors="coerce"))
        if timestamps_utc.tz is None:
            timestamps_utc = timestamps_utc.tz_localize("UTC")
        timestamps_utc = timestamps_utc.sort_values()

        source_timezone = str(df.attrs.get("source_timezone", "UTC"))
        local_timestamps = timestamps_utc.tz_convert(source_timezone)
        local_daily_timestamps = None
        if daily_reference is not None and not daily_reference.empty:
            daily_utc = pd.DatetimeIndex(
                pd.to_datetime(daily_reference["timestamp"], utc=True, errors="coerce")
            )
            if daily_utc.tz is None:
                daily_utc = daily_utc.tz_localize("UTC")
            local_daily_timestamps = daily_utc.tz_convert(
                str(daily_reference.attrs.get("source_timezone", source_timezone))
            )

        duplicate_timestamps = int(pd.Series(timestamps_utc).duplicated().sum())
        price_frame = df[["open", "high", "low", "close"]].copy()
        nonfinite_ohlc_rows = int((~np.isfinite(price_frame.to_numpy(dtype=float))).any(axis=1).sum())
        nonpositive_price_rows = int((price_frame.to_numpy(dtype=float) <= 0).any(axis=1).sum())
        synthetic_volume_rows = int(df["is_synthetic_vol"].fillna(False).astype(bool).sum())
        zero_volume_rows = int(
            ((~df["is_synthetic_vol"].fillna(False).astype(bool)) & (df["volume"].fillna(0.0) <= 0.0)).sum()
        )

        payload = {
            "continuity_mode": "continuous" if self._is_continuous_market(instrument) else "session",
            "source_timezone": source_timezone,
        }
        if self._is_continuous_market(instrument):
            expected_bar_count, missing_bar_count, max_gap_bars = self._audit_continuous_gap_metrics(
                timestamps_utc, timeframe=timeframe
            )
        elif timeframe == "1H":
            (
                expected_bar_count,
                missing_bar_count,
                max_gap_bars,
                hourly_payload,
            ) = self._audit_session_hourly_gap_metrics(
                local_timestamps,
                local_daily_timestamps,
            )
            payload.update(hourly_payload)
        else:
            expected_bar_count, missing_bar_count, max_gap_bars = self._audit_session_daily_gap_metrics(
                local_timestamps
            )

        missing_bar_ratio = (
            float(missing_bar_count) / float(expected_bar_count)
            if expected_bar_count > 0
            else 0.0
        )
        synthetic_volume_ratio = (
            float(synthetic_volume_rows) / float(len(df))
            if len(df) > 0
            else 0.0
        )
        last_timestamp = timestamps_utc.max() if len(timestamps_utc) else None
        staleness_hours = self._staleness_hours(
            synced_at=synced_at,
            last_timestamp=last_timestamp,
        )
        payload["staleness_hours"] = float(staleness_hours)
        if payload_extra:
            payload.update(payload_extra)
        quality_status = self._quality_status_for_metrics(
            timeframe=timeframe,
            row_count=int(len(df)),
            duplicate_timestamps=duplicate_timestamps,
            nonfinite_ohlc_rows=nonfinite_ohlc_rows,
            nonpositive_price_rows=nonpositive_price_rows,
            missing_bar_ratio=missing_bar_ratio,
            synthetic_volume_ratio=synthetic_volume_ratio,
            staleness_hours=staleness_hours,
        )

        return {
            "run_id": run_id,
            "synced_at": synced_at,
            "base_symbol": instrument.base_symbol,
            "pair_symbol": instrument.base_symbol,
            "asset_class": "SPOT",
            "source_exchange": instrument.source_exchange,
            "source_symbol": instrument.ticker,
            "timeframe": timeframe,
            "row_count": int(len(df)),
            "first_timestamp": self._timestamp_text(timestamps_utc.min() if len(timestamps_utc) else None),
            "last_timestamp": self._timestamp_text(timestamps_utc.max() if len(timestamps_utc) else None),
            "duplicate_timestamps": duplicate_timestamps,
            "nonfinite_ohlc_rows": nonfinite_ohlc_rows,
            "nonpositive_price_rows": nonpositive_price_rows,
            "synthetic_volume_rows": synthetic_volume_rows,
            "zero_volume_rows": zero_volume_rows,
            "expected_bar_count": int(expected_bar_count),
            "missing_bar_count": int(missing_bar_count),
            "missing_bar_ratio": float(missing_bar_ratio),
            "max_gap_bars": float(max_gap_bars),
            "quality_status": quality_status,
            "audit_payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }

    def _build_missing_quality_row(
        self,
        *,
        instrument: YahooInstrument,
        timeframe: str,
        run_id: str,
        synced_at: str,
        reason: str,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "synced_at": synced_at,
            "base_symbol": instrument.base_symbol,
            "pair_symbol": instrument.base_symbol,
            "asset_class": "SPOT",
            "source_exchange": instrument.source_exchange,
            "source_symbol": instrument.ticker,
            "timeframe": timeframe,
            "row_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "duplicate_timestamps": 0,
            "nonfinite_ohlc_rows": 0,
            "nonpositive_price_rows": 0,
            "synthetic_volume_rows": 0,
            "zero_volume_rows": 0,
            "expected_bar_count": 0,
            "missing_bar_count": 0,
            "missing_bar_ratio": 1.0,
            "max_gap_bars": 0.0,
            "quality_status": QUALITY_STATUS_FAIL,
            "audit_payload_json": json.dumps({"reason": reason}, sort_keys=True, separators=(",", ":")),
        }

    def _history_period(self, timeframe: str) -> str:
        if timeframe == "1D":
            return "max"
        if timeframe == "1H":
            # Ask Yahoo for the maximum available 1H history. As of 2026-04-06
            # Yahoo still returns only the last ~730 days for 1H, but using
            # "max" keeps us future-compatible if that window expands.
            return "max"
        raise ValueError(f"Unsupported fetch timeframe: {timeframe}")

    def _history_interval(self, timeframe: str) -> str:
        if timeframe == "1D":
            return "1d"
        if timeframe == "1H":
            return "1h"
        raise ValueError(f"Unsupported fetch timeframe: {timeframe}")

    def _symbol_payload_json(self, instrument: YahooInstrument, timeframe: str) -> str:
        payload = {
            "asset_class": "SPOT",
            "contract_kind": "SPOT",
            "pair_symbol": instrument.base_symbol,
            "provider": "yfinance",
            "quote_asset": instrument.quote_asset,
            "source_exchange": instrument.source_exchange,
            "source_symbol": instrument.ticker,
            "timeframe": timeframe,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _empty_history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=MARKET_DNA_COLUMNS)

    def _fetch_history_frame(
        self,
        instrument: YahooInstrument,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        history_kwargs = {
            "interval": self._history_interval(timeframe),
            "auto_adjust": False,
            "actions": False,
        }
        if start is None:
            history_kwargs["period"] = self._history_period(timeframe)
        else:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            history_kwargs["start"] = start_ts.to_pydatetime()

        history = yf.Ticker(instrument.ticker).history(**history_kwargs)
        if history.empty:
            return self._empty_history_frame()

        price_df = history.copy()
        price_df.columns = [str(col).strip().lower() for col in price_df.columns]
        source_timezone = str(pd.DatetimeIndex(price_df.index).tz or "UTC")
        index = pd.DatetimeIndex(price_df.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")

        frame = pd.DataFrame(
            {
                "base_symbol": instrument.base_symbol,
                "pair_symbol": instrument.base_symbol,
                "asset_class": "SPOT",
                "source_exchange": instrument.source_exchange,
                "source_symbol": instrument.ticker,
                "symbol_payload_json": self._symbol_payload_json(instrument, timeframe),
                "contract_kind": "SPOT",
                "quote_asset": instrument.quote_asset,
                "expiry_code": None,
                "timeframe": timeframe,
                "timestamp": index,
                "open": price_df["open"].to_numpy(dtype=float),
                "high": price_df["high"].to_numpy(dtype=float),
                "low": price_df["low"].to_numpy(dtype=float),
                "close": price_df["close"].to_numpy(dtype=float),
                "volume": price_df.get(
                    "volume", pd.Series(np.zeros(len(price_df)), index=price_df.index)
                ).to_numpy(dtype=float),
                "is_synthetic_vol": False,
                "realized_volatility": np.nan,
            }
        )
        frame = frame.reset_index(drop=True)
        frame.attrs["source_timezone"] = source_timezone
        return frame

    def _enforce_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["high"] = result[["open", "close", "high"]].max(axis=1)
        result["low"] = result[["open", "close", "low"]].min(axis=1)
        return self._copy_attrs(df, result)

    def _interpolate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["TR"] = result["high"] - result["low"]
        zero_vol_mask = (result["volume"] == 0.0) | (result["volume"].isna())
        result["is_synthetic_vol"] = zero_vol_mask

        if not zero_vol_mask.any():
            return result.drop(columns=["TR"])

        rolling_tr = (
            result["TR"]
            .replace(0, np.nan)
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )
        rolling_vol = (
            result["volume"]
            .replace(0, np.nan)
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )
        vol_tr_ratio = rolling_vol / rolling_tr
        synthetic_vol = result["TR"] * vol_tr_ratio
        synthetic_vol = np.where(result["TR"] == 0, rolling_vol, synthetic_vol)

        result.loc[zero_vol_mask, "volume"] = synthetic_vol[zero_vol_mask]
        result["volume"] = result["volume"].ffill().bfill().fillna(0.0)
        return self._copy_attrs(df, result.drop(columns=["TR"]))

    def _compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["realized_volatility"] = np.nan

        for timeframe, ann_factor in REALIZED_VOL_ANNUALIZATION.items():
            tf_mask = result["timeframe"] == timeframe
            if not tf_mask.any():
                continue

            tf_df = result.loc[tf_mask].sort_values("timestamp").copy()
            o = tf_df["open"].to_numpy(dtype=float)
            h = tf_df["high"].to_numpy(dtype=float)
            lo = tf_df["low"].to_numpy(dtype=float)
            c = tf_df["close"].to_numpy(dtype=float)
            n_bars = len(c)
            window = 21

            rv_yz = np.full(n_bars, np.nan)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_oc = np.log(np.maximum(c, 1e-9) / np.maximum(o, 1e-9))
                log_co = np.zeros(n_bars)
                log_co[1:] = np.log(np.maximum(o[1:], 1e-9) / np.maximum(c[:-1], 1e-9))
                log_hc = np.log(np.maximum(h, 1e-9) / np.maximum(c, 1e-9))
                log_ho = np.log(np.maximum(h, 1e-9) / np.maximum(o, 1e-9))
                log_lc = np.log(np.maximum(lo, 1e-9) / np.maximum(c, 1e-9))
                log_lo = np.log(np.maximum(lo, 1e-9) / np.maximum(o, 1e-9))
                rs = log_hc * log_ho + log_lc * log_lo

            k = 0.34 / (1.34 + (window + 1) / (window - 1))

            for i in range(window, n_bars):
                s = slice(i - window + 1, i + 1)
                co_s = log_co[max(i - window + 2, 1):i + 1]
                if len(co_s) < 2:
                    continue
                mu_o = co_s.mean()
                sigma_o2 = float(np.mean((co_s - mu_o) ** 2))
                oc_s = log_oc[s]
                mu_c = oc_s.mean()
                sigma_c2 = float(np.mean((oc_s - mu_c) ** 2))
                sigma_rs2 = float(rs[s].mean())
                yz_var = sigma_o2 + k * sigma_rs2 + (1.0 - k) * sigma_c2
                if np.isfinite(yz_var) and yz_var > 0:
                    rv_yz[i] = np.sqrt(yz_var * ann_factor)

            rv_series = pd.Series(rv_yz, index=tf_df.index)
            rv_series = rv_series.bfill().fillna(0.0)
            result.loc[tf_df.index, "realized_volatility"] = rv_series.to_numpy()

        return self._copy_attrs(df, result)

    def _generate_macro_layers(self, df_1d: pd.DataFrame) -> pd.DataFrame:
        macro_dfs: list[pd.DataFrame] = []
        df = df_1d.copy().sort_values("timestamp")
        df["iso_year"] = df["timestamp"].dt.isocalendar().year
        df["iso_week"] = df["timestamp"].dt.isocalendar().week
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        df["half"] = np.where(df["month"] <= 6, 1, 2)

        signatures = {
            "1W": ["iso_year", "iso_week"],
            "1M": ["year", "month"],
            "3M": ["year", "quarter"],
            "6M": ["year", "half"],
            "12M": ["year"],
        }
        agg_logic = {
            "timestamp": "first",
            "base_symbol": "first",
            "pair_symbol": "first",
            "asset_class": "first",
            "source_exchange": "first",
            "source_symbol": "first",
            "symbol_payload_json": "first",
            "contract_kind": "first",
            "quote_asset": "first",
            "expiry_code": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "is_synthetic_vol": "max",
            "realized_volatility": "last",
        }

        for label, group_cols in signatures.items():
            df_macro = df.groupby(group_cols).agg(agg_logic).reset_index(drop=True)
            if df_macro.empty:
                continue
            df_macro["timestamp"] = pd.to_datetime(df_macro["timestamp"], utc=True)
            df_macro["timeframe"] = label
            macro_dfs.append(df_macro)

        if not macro_dfs:
            return self._empty_history_frame()
        return pd.concat(macro_dfs, ignore_index=True)

    def _prepare_market_dna_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_db = df.copy()
        ts = pd.to_datetime(df_db["timestamp"], utc=True, errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df_db["timestamp"] = ts.dt.tz_convert("UTC").dt.strftime(UTC_TIMESTAMP_FORMAT)
        return df_db.loc[:, list(MARKET_DNA_COLUMNS)]

    def _insert_quality_rows(self, cursor: sqlite3.Cursor, quality_rows: list[dict]) -> None:
        if not quality_rows:
            return
        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO market_sync_quality (
                {", ".join(MARKET_SYNC_QUALITY_COLUMNS)}
            ) VALUES ({", ".join("?" for _ in MARKET_SYNC_QUALITY_COLUMNS)})
            """,
            [
                tuple(row.get(column) for column in MARKET_SYNC_QUALITY_COLUMNS)
                for row in quality_rows
            ],
        )

    def _replace_symbol_history_atomic(
        self,
        *,
        pair_symbol: str,
        market_df: pd.DataFrame | None,
        quality_rows: list[dict],
    ) -> None:
        with self.conn:
            cursor = self.conn.cursor()
            if market_df is not None and not market_df.empty:
                market_db = self._prepare_market_dna_dataframe(market_df)
                timeframes_to_replace = sorted(
                    str(timeframe) for timeframe in market_db["timeframe"].unique()
                )
                placeholders = ", ".join("?" for _ in timeframes_to_replace)
                cursor.execute(
                    f"""
                    DELETE FROM market_dna
                    WHERE pair_symbol = ?
                      AND asset_class = 'SPOT'
                      AND timeframe IN ({placeholders})
                    """,
                    (pair_symbol, *timeframes_to_replace),
                )
                cursor.execute("DROP TABLE IF EXISTS temp_market_data")
                cursor.execute(
                    """
                    CREATE TEMP TABLE temp_market_data AS
                    SELECT * FROM market_dna WHERE 0
                    """
                )
                market_db.to_sql(
                    "temp_market_data",
                    self.conn,
                    if_exists="append",
                    index=False,
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO market_dna (
                        base_symbol, pair_symbol, asset_class, source_exchange, source_symbol,
                        symbol_payload_json, contract_kind, quote_asset, expiry_code,
                        timeframe, timestamp, open, high, low, close, volume,
                        is_synthetic_vol, realized_volatility
                    )
                    SELECT
                        base_symbol, pair_symbol, asset_class, source_exchange, source_symbol,
                        symbol_payload_json, contract_kind, quote_asset, expiry_code,
                        timeframe, timestamp, open, high, low, close, volume,
                        is_synthetic_vol, realized_volatility
                    FROM temp_market_data
                    """
                )
                cursor.execute("DROP TABLE IF EXISTS temp_market_data")
            self._insert_quality_rows(cursor, quality_rows)

    def sync_symbol(self, base_symbol: str, *, force_full_refresh: bool = False) -> None:
        instrument = self._resolve_instrument(base_symbol)
        print(f"Syncing {instrument.base_symbol} <- {instrument.ticker}")

        run_id = uuid4().hex
        synced_at = self._sync_now_utc()
        frames: list[pd.DataFrame] = []
        df_1d: pd.DataFrame | None = None
        daily_updated = False
        quality_rows: list[dict] = []
        for timeframe in FETCH_TIMEFRAMES:
            existing_frame = self._load_existing_timeframe_frame(
                instrument=instrument,
                timeframe=timeframe,
            )
            latest_quality = self._latest_quality_snapshot(
                instrument=instrument,
                timeframe=timeframe,
            )
            needs_full_refresh, refresh_reason = self._should_force_full_refresh(
                timeframe=timeframe,
                existing_df=existing_frame,
                latest_quality=latest_quality,
                synced_at=synced_at,
                force_full_refresh=force_full_refresh,
            )
            fetch_mode = "full" if needs_full_refresh else "incremental"
            fetch_start = (
                None
                if needs_full_refresh
                else self._fetch_start_timestamp(
                    timeframe=timeframe,
                    existing_df=existing_frame,
                )
            )

            try:
                fetched = self._fetch_history_frame(
                    instrument,
                    timeframe,
                    start=fetch_start,
                )
            except Exception as exc:
                print(f"  [!] {timeframe} fetch error for {instrument.base_symbol}: {exc}")
                if not existing_frame.empty:
                    retained_quality = self._audit_fetched_frame(
                        existing_frame,
                        instrument=instrument,
                        timeframe=timeframe,
                        run_id=run_id,
                        synced_at=synced_at,
                        daily_reference=df_1d if timeframe == "1H" else None,
                    )
                    quality_rows.append(
                        self._decorate_quality_row(
                            quality_row=retained_quality,
                            fetch_mode=fetch_mode,
                            fetch_status=f"provider_exception:{type(exc).__name__}",
                            existing_row_count=len(existing_frame),
                            fetched_row_count=0,
                            merged_row_count=len(existing_frame),
                            retained_row_count=len(existing_frame),
                            minimum_status=QUALITY_STATUS_WARN,
                            extra_payload={"refresh_reason": refresh_reason},
                        )
                    )
                    if timeframe == "1D":
                        df_1d = existing_frame
                else:
                    quality_rows.append(
                        self._build_missing_quality_row(
                            instrument=instrument,
                            timeframe=timeframe,
                            run_id=run_id,
                            synced_at=synced_at,
                            reason=f"provider_exception:{type(exc).__name__}",
                        )
                    )
                continue
            if fetched.empty:
                print(f"  [!] No {timeframe} data returned for {instrument.base_symbol}.")
                if not existing_frame.empty:
                    retained_quality = self._audit_fetched_frame(
                        existing_frame,
                        instrument=instrument,
                        timeframe=timeframe,
                        run_id=run_id,
                        synced_at=synced_at,
                        daily_reference=df_1d if timeframe == "1H" else None,
                    )
                    quality_rows.append(
                        self._decorate_quality_row(
                            quality_row=retained_quality,
                            fetch_mode=fetch_mode,
                            fetch_status="provider_returned_empty_frame",
                            existing_row_count=len(existing_frame),
                            fetched_row_count=0,
                            merged_row_count=len(existing_frame),
                            retained_row_count=len(existing_frame),
                            minimum_status=QUALITY_STATUS_WARN,
                            extra_payload={"refresh_reason": refresh_reason},
                        )
                    )
                    if timeframe == "1D":
                        df_1d = existing_frame
                else:
                    quality_rows.append(
                        self._build_missing_quality_row(
                            instrument=instrument,
                            timeframe=timeframe,
                            run_id=run_id,
                            synced_at=synced_at,
                            reason="provider_returned_empty_frame",
                        )
                    )
                continue

            merged = self._merge_market_frames(existing_frame, fetched)
            existing_last = (
                pd.to_datetime(existing_frame["timestamp"], utc=True).max()
                if not existing_frame.empty
                else None
            )
            merged_last = (
                pd.to_datetime(merged["timestamp"], utc=True).max()
                if not merged.empty
                else None
            )
            if (
                fetch_mode == "incremental"
                and existing_last is not None
                and (merged_last is None or merged_last < existing_last)
            ):
                fetched = self._fetch_history_frame(instrument, timeframe)
                merged = self._merge_market_frames(self._empty_history_frame(), fetched)
                fetch_mode = "full"
                refresh_reason = "incremental_regressed_latest_timestamp"

            merged = self._compute_realized_volatility(
                self._interpolate_volume(self._enforce_physics(merged))
            )
            frames.append(merged)
            if timeframe == "1D":
                df_1d = merged
                daily_updated = True
            quality_row = self._audit_fetched_frame(
                merged,
                instrument=instrument,
                timeframe=timeframe,
                run_id=run_id,
                synced_at=synced_at,
                daily_reference=df_1d if timeframe == "1H" else None,
                payload_extra={"refresh_reason": refresh_reason},
            )
            retained_row_count = max(len(merged) - len(fetched), 0)
            quality_row = self._decorate_quality_row(
                quality_row=quality_row,
                fetch_mode=fetch_mode,
                fetch_status="fetched",
                existing_row_count=len(existing_frame),
                fetched_row_count=len(fetched),
                merged_row_count=len(merged),
                retained_row_count=retained_row_count,
                extra_payload={"refresh_reason": refresh_reason},
            )
            quality_rows.append(quality_row)
            print(
                f"  [+] {timeframe} rows: {len(merged)} "
                f"({merged['timestamp'].min()} -> {merged['timestamp'].max()})"
            )
            print(
                "      "
                f"mode={fetch_mode} "
                f"quality={quality_row['quality_status']} "
                f"missing={quality_row['missing_bar_count']}/{quality_row['expected_bar_count']} "
                f"({quality_row['missing_bar_ratio']:.2%}) "
                f"synthetic_vol={quality_row['synthetic_volume_rows']}"
            )

        if daily_updated and df_1d is not None and not df_1d.empty:
            macro_df = self._generate_macro_layers(df_1d)
            if not macro_df.empty:
                macro_df = self._compute_realized_volatility(macro_df)
                frames.append(macro_df)
                print(
                    f"  [+] Macro rows forged: {len(macro_df)} "
                    f"({', '.join(MACRO_TIMEFRAMES)})"
                )

        if not frames:
            self._replace_symbol_history_atomic(
                pair_symbol=instrument.base_symbol,
                market_df=None,
                quality_rows=quality_rows,
            )
            raise ValueError(f"No Yahoo Finance data fetched for {instrument.base_symbol}.")

        master_df = pd.concat(frames, ignore_index=True)
        self._replace_symbol_history_atomic(
            pair_symbol=instrument.base_symbol,
            market_df=master_df,
            quality_rows=quality_rows,
        )
        print(f"  [+] {instrument.base_symbol} sync complete. Rows written: {len(master_df)}")

    def sync(
        self,
        base_symbols: Iterable[str] | None = None,
        pause_seconds: float = 1.0,
        *,
        force_full_refresh: bool = False,
    ) -> None:
        if base_symbols is None:
            requested_symbols = list(CORE_WATCHLIST_SYMBOLS) + [
                symbol
                for symbol in YAHOO_INSTRUMENTS
                if symbol not in CORE_WATCHLIST_SYMBOLS
            ]
            requested_symbols.extend(
                alias
                for alias in sorted(self.custom_instrument_records)
                if alias not in requested_symbols
            )
        else:
            requested_symbols = [str(symbol).strip() for symbol in base_symbols if str(symbol).strip()]

        symbols: list[str] = []
        seen_base_symbols: set[str] = set()
        for requested_symbol in requested_symbols:
            instrument = self._resolve_instrument(requested_symbol)
            if instrument.base_symbol in seen_base_symbols:
                continue
            seen_base_symbols.add(instrument.base_symbol)
            symbols.append(requested_symbol)

        success_count = 0
        failures: list[tuple[str, str]] = []
        for idx, symbol in enumerate(symbols):
            try:
                self.sync_symbol(symbol, force_full_refresh=force_full_refresh)
                success_count += 1
            except Exception as exc:
                failures.append((symbol, str(exc)))
                print(f"  [!] Sync failed for {symbol}: {exc}")
            if pause_seconds > 0 and idx < len(symbols) - 1:
                time.sleep(pause_seconds)

        if failures:
            print("Yahoo Finance sync completed with warnings:")
            for symbol, message in failures:
                print(f"  - {symbol}: {message}")
        if success_count == 0:
            raise ValueError("No Yahoo Finance symbols synced successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Yahoo Finance OHLCV vault")
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help=(
            "Base symbol to sync. Repeat for multiple symbols. Accepts curated aliases, "
            "custom registered aliases, and raw Yahoo tickers. Defaults to all mapped symbols."
        ),
    )
    parser.add_argument(
        "--import-index",
        action="append",
        default=None,
        help=(
            "Import an index without agent help. Use ALIAS=YAHOO_TICKER to persist a new "
            "local alias and sync it immediately, or pass a raw Yahoo ticker like ^GDAXI."
        ),
    )
    parser.add_argument(
        "--register-index",
        action="append",
        default=None,
        help=(
            "Persist a custom Yahoo-backed index alias without syncing yet. "
            "Format: ALIAS=YAHOO_TICKER, for example DAX=^GDAXI."
        ),
    )
    parser.add_argument(
        "--db-path",
        default="ohlcv.db",
        help="SQLite database path. Defaults to data_vault/ohlcv.db.",
    )
    parser.add_argument(
        "--custom-instruments-path",
        default=None,
        help=(
            "JSON file that stores custom Yahoo-backed aliases. "
            "Defaults to data_vault/custom_yahoo_instruments.json."
        ),
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help=(
            "List the built-in and custom Yahoo-backed aliases currently available "
            "for plain --symbol use."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Sleep between Yahoo requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore incremental windows and rebuild each requested symbol from Yahoo max history.",
    )
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help=(
            "Keep the vault running and auto-sync only when CPU load and Yahoo probe speed "
            "meet the configured thresholds."
        ),
    )
    parser.add_argument(
        "--auto-check-interval-seconds",
        type=float,
        default=DEFAULT_AUTO_SYNC_CHECK_INTERVAL_SECONDS,
        help="How often the auto-sync supervisor checks resource health.",
    )
    parser.add_argument(
        "--auto-min-sync-gap-seconds",
        type=float,
        default=DEFAULT_AUTO_SYNC_MIN_GAP_SECONDS,
        help="Minimum wall-clock gap between auto-sync attempts.",
    )
    parser.add_argument(
        "--auto-max-cpu-percent",
        type=float,
        default=DEFAULT_AUTO_SYNC_MAX_CPU_PERCENT,
        help="Maximum sampled CPU usage percent allowed before an auto-sync may start.",
    )
    parser.add_argument(
        "--auto-cpu-sample-seconds",
        type=float,
        default=DEFAULT_AUTO_SYNC_CPU_SAMPLE_SECONDS,
        help="Sampling window used to estimate CPU usage before each auto-sync decision.",
    )
    parser.add_argument(
        "--auto-min-download-kbps",
        type=float,
        default=DEFAULT_AUTO_SYNC_MIN_DOWNLOAD_KBPS,
        help="Minimum Yahoo probe download throughput required before an auto-sync may start.",
    )
    parser.add_argument(
        "--auto-probe-url",
        default=DEFAULT_AUTO_SYNC_PROBE_URL,
        help="Yahoo endpoint used to estimate download throughput for auto-sync gating.",
    )
    parser.add_argument(
        "--auto-probe-timeout-seconds",
        type=float,
        default=DEFAULT_AUTO_SYNC_PROBE_TIMEOUT_SECONDS,
        help="Timeout used by the Yahoo probe request for auto-sync gating.",
    )
    parser.add_argument(
        "--resource-gate-status",
        action="store_true",
        help=(
            "Print a one-shot CPU and Yahoo throughput snapshot using the current "
            "auto-sync thresholds, then exit unless other work was requested."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    vault = YFinanceVault(
        db_path=args.db_path,
        custom_instruments_path=args.custom_instruments_path,
    )

    requested_symbols = list(args.symbol or [])

    for spec in args.register_index or []:
        instrument = vault.register_custom_index(spec)
        print(
            f"[register-index] {instrument.base_symbol} -> {instrument.ticker} "
            f"saved to {vault.custom_instruments_path}"
        )

    for spec in args.import_index or []:
        token = str(spec).strip()
        if not token:
            continue
        if "=" in token:
            instrument = vault.register_custom_index(token)
            requested_symbols.append(instrument.base_symbol)
            print(
                f"[import-index] registered {instrument.base_symbol} -> {instrument.ticker} "
                "and queued it for sync."
            )
        else:
            requested_symbols.append(token)

    did_auxiliary_action = False

    if args.list_symbols:
        print("Available Yahoo-backed aliases")
        for alias_key, ticker, source in vault.list_available_symbols():
            print(f"  {alias_key:16s} -> {ticker:20s} [{source}]")
        print("  Raw Yahoo tickers such as ^GDAXI or EURUSD=X also work directly.")
        did_auxiliary_action = True

    if args.resource_gate_status:
        vault.print_resource_gate_snapshot(
            max_cpu_percent=args.auto_max_cpu_percent,
            cpu_sample_seconds=args.auto_cpu_sample_seconds,
            min_download_kbps=args.auto_min_download_kbps,
            probe_url=args.auto_probe_url,
            probe_timeout_seconds=args.auto_probe_timeout_seconds,
        )
        did_auxiliary_action = True

    sync_requested = bool(
        requested_symbols
        or args.import_index
        or (
            not args.register_index
            and not args.list_symbols
            and not args.resource_gate_status
        )
    )

    if args.auto_sync:
        vault.auto_sync(
            base_symbols=requested_symbols or None,
            pause_seconds=args.pause_seconds,
            force_full_refresh=args.full_refresh,
            check_interval_seconds=args.auto_check_interval_seconds,
            min_sync_gap_seconds=args.auto_min_sync_gap_seconds,
            max_cpu_percent=args.auto_max_cpu_percent,
            cpu_sample_seconds=args.auto_cpu_sample_seconds,
            min_download_kbps=args.auto_min_download_kbps,
            probe_url=args.auto_probe_url,
            probe_timeout_seconds=args.auto_probe_timeout_seconds,
        )
        return 0

    if sync_requested:
        vault.sync(
            base_symbols=requested_symbols or None,
            pause_seconds=args.pause_seconds,
            force_full_refresh=args.full_refresh,
        )
        print("Yahoo Finance sync complete. Database locked and ready.")
    else:
        if not did_auxiliary_action:
            print("Custom index registry updated. No sync requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
