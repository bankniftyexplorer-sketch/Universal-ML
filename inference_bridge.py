import json
import os
import sqlite3

import pandas as pd

from data_vault.symbol_identity import canonical_pair_symbol, lookup_registry_entry

MACRO_PARENT_TIMEFRAMES = {
    "1W": "1D",
    "1M": "1D",
    "3M": "1D",
    "6M": "1D",
    "12M": "1D",
}

VIX_COMPANION_MAP: dict[str, str] = {
    "NIFTY": "INDIA_VIX",
    "BANKNIFTY": "INDIA_VIX",
    "SENSEX": "INDIA_VIX",
    "FINNIFTY": "INDIA_VIX",
    "MIDCPNIFTY": "INDIA_VIX",
    "NIFTYNXT50": "INDIA_VIX",
    "SPX500": "VIX",
}


class DataIntegrityError(RuntimeError):
    """Fatal runtime gate for model-facing market data integrity failures."""


class InferenceBridge:
    def __init__(self, db_path="data_vault/ohlcv.db"):
        """
        Initializes the connection to the Data Sovereign.
        Ensure the path accurately points to your new ohlcv.db file.
        """
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"[!] CRITICAL: Database not found at {self.db_path}"
            )
        with sqlite3.connect(self.db_path) as conn:
            self._configure_connection(conn)

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA mmap_size=1000000000;")
        conn.execute("PRAGMA cache_size=-200000;")

    def _market_dna_columns(self, conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(market_dna)").fetchall()
        return {row[1] for row in rows}

    def _market_sync_quality_exists(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'market_sync_quality'"
        ).fetchone()
        return row is not None

    def _distinct_identity_sources(
        self,
        conn: sqlite3.Connection,
        pair_symbol: str,
        asset_class: str,
    ) -> list[tuple[str, str]]:
        columns = self._market_dna_columns(conn)
        required_cols = {"pair_symbol", "source_exchange", "source_symbol"}
        if not required_cols.issubset(columns):
            return []

        rows = conn.execute(
            """
            SELECT DISTINCT source_exchange, source_symbol
            FROM market_dna
            WHERE pair_symbol = ?
              AND asset_class = ?
              AND source_exchange IS NOT NULL
              AND source_exchange != ''
              AND source_symbol IS NOT NULL
              AND source_symbol != ''
            ORDER BY source_exchange ASC, source_symbol ASC
            """,
            (pair_symbol, asset_class),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _resolve_db_symbol(
        self,
        conn: sqlite3.Connection,
        requested_symbol: str,
        asset_class: str,
    ) -> tuple[tuple[str, str] | None, str]:
        pair_symbol = canonical_pair_symbol(requested_symbol, asset_class=asset_class)
        registry_entry = lookup_registry_entry(pair_symbol, asset_class)
        sources = self._distinct_identity_sources(conn, pair_symbol, asset_class)

        if registry_entry is not None:
            expected_source = (
                registry_entry.source_exchange,
                registry_entry.source_symbol,
            )
            unexpected_sources = [
                source for source in sources if source != expected_source
            ]
            if unexpected_sources:
                raise ValueError(
                    "Registry-covered instrument has multiple venues in market_dna: "
                    f"{pair_symbol} {asset_class} expected only {expected_source}, "
                    f"found {sources}."
                )
            if expected_source not in sources:
                return None, pair_symbol
            return expected_source, pair_symbol

        if not sources:
            return None, pair_symbol
        if len(sources) > 1:
            raise ValueError(
                f"Ambiguous {asset_class} sources for {requested_symbol}: "
                f"{pair_symbol} maps to {sources}. "
                "Keep exactly one exchange/source_symbol pair or add a registry entry."
            )
        return sources[0], pair_symbol

    def _latest_quality_by_timeframe(
        self,
        conn: sqlite3.Connection,
        *,
        pair_symbol: str,
        asset_class: str,
        source_exchange: str,
        source_symbol: str,
    ) -> dict[str, dict]:
        if not self._market_sync_quality_exists(conn):
            return {}

        rows = conn.execute(
            """
            WITH ranked_quality AS (
                SELECT timeframe, synced_at, row_count, expected_bar_count,
                       missing_bar_count, missing_bar_ratio, max_gap_bars,
                       synthetic_volume_rows, quality_status, audit_payload_json,
                       ROW_NUMBER() OVER (
                           PARTITION BY timeframe
                           ORDER BY synced_at DESC, run_id DESC
                       ) AS row_rank
                FROM market_sync_quality
                WHERE pair_symbol = ?
                  AND asset_class = ?
                  AND source_exchange = ?
                  AND source_symbol = ?
            )
            SELECT timeframe, synced_at, row_count, expected_bar_count,
                   missing_bar_count, missing_bar_ratio, max_gap_bars,
                   synthetic_volume_rows, quality_status, audit_payload_json
            FROM ranked_quality
            WHERE row_rank = 1
            """,
            (
                pair_symbol,
                asset_class,
                source_exchange,
                source_symbol,
            ),
        ).fetchall()
        quality_by_timeframe = {}
        for row in rows:
            audit_payload = {}
            if row[9]:
                try:
                    parsed_payload = json.loads(row[9])
                except (TypeError, ValueError, json.JSONDecodeError):
                    parsed_payload = {}
                if isinstance(parsed_payload, dict):
                    audit_payload = parsed_payload
            quality_by_timeframe[row[0]] = {
                "synced_at": row[1],
                "row_count": row[2],
                "expected_bar_count": row[3],
                "missing_bar_count": row[4],
                "missing_bar_ratio": row[5],
                "max_gap_bars": row[6],
                "synthetic_volume_rows": row[7],
                "status": row[8],
                "quality_status": row[8],
                "audit_payload_json": row[9],
                "audit_payload": audit_payload,
            }
        return quality_by_timeframe

    def resolve_pair_source(
        self,
        requested_symbol: str,
        asset_class: str,
    ) -> dict[str, str] | None:
        with sqlite3.connect(self.db_path) as conn:
            self._configure_connection(conn)
            resolved_source, pair_symbol = self._resolve_db_symbol(
                conn, requested_symbol, asset_class
            )
            if resolved_source is None:
                return None
            source_exchange, source_symbol = resolved_source
            return {
                "pair_symbol": pair_symbol,
                "source_exchange": source_exchange,
                "source_symbol": source_symbol,
            }

    def fetch_holographic_stack_sql(
        self,
        base_symbol: str,
        asset_class: str,
        *,
        include_realized_vol: bool = False,
    ) -> pd.DataFrame:
        select_cols = [
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_synthetic_vol",
        ]
        if include_realized_vol:
            select_cols.append("realized_volatility")

        with sqlite3.connect(self.db_path) as conn:
            self._configure_connection(conn)
            resolved_source, pair_symbol = self._resolve_db_symbol(
                conn, base_symbol, asset_class
            )
            if resolved_source is None:
                return pd.DataFrame(columns=select_cols)

            source_exchange, source_symbol = resolved_source
            quality_map = self._latest_quality_by_timeframe(
                conn,
                pair_symbol=pair_symbol,
                asset_class=asset_class,
                source_exchange=source_exchange,
                source_symbol=source_symbol,
            )
            query = f"""
                SELECT {", ".join(select_cols)}
                FROM market_dna
                WHERE pair_symbol = ?
                  AND asset_class = ?
                  AND source_exchange = ?
                  AND source_symbol = ?
                ORDER BY timestamp ASC
            """
            raw_df = pd.read_sql_query(
                query,
                conn,
                params=(pair_symbol, asset_class, source_exchange, source_symbol),
            )
            if not raw_df.empty:
                raw_df.attrs["pair_symbol"] = pair_symbol
                raw_df.attrs["resolved_source_exchange"] = source_exchange
                raw_df.attrs["resolved_source_symbol"] = source_symbol
                raw_df.attrs["quality_by_timeframe"] = quality_map
            return raw_df

    def fetch_holographic_stack_pandas(
        self, raw_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        if raw_df.empty:
            return {}

        raw_df = raw_df.copy()
        raw_df["timestamp"] = pd.to_datetime(
            raw_df["timestamp"],
            format="%Y-%m-%dT%H:%M:%S+00:00",
            exact=True,
            utc=True,
        )

        # Rename timestamp -> time to preserve the model-facing contract.
        raw_df = raw_df.rename(columns={"timestamp": "time"})
        raw_df["time"] = raw_df["time"].dt.tz_convert("Asia/Kolkata")

        holographic_stack = {}
        for tf in raw_df["timeframe"].unique():
            tf_df = raw_df[raw_df["timeframe"] == tf].copy()
            tf_df = tf_df.drop(columns=["timeframe"]).reset_index(drop=True)
            for key in (
                "pair_symbol",
                "resolved_source_exchange",
                "resolved_source_symbol",
            ):
                if key in raw_df.attrs:
                    tf_df.attrs[key] = raw_df.attrs[key]
            quality_map = raw_df.attrs.get("quality_by_timeframe", {})
            if tf in quality_map:
                tf_df.attrs["data_quality"] = quality_map[tf]
            elif tf in MACRO_PARENT_TIMEFRAMES:
                parent_tf = MACRO_PARENT_TIMEFRAMES[tf]
                parent_quality = quality_map.get(parent_tf)
                if parent_quality is not None:
                    inherited_quality = dict(parent_quality)
                    inherited_payload = dict(parent_quality.get("audit_payload", {}))
                    inherited_payload["derived_from_timeframe"] = parent_tf
                    inherited_payload["derived_timeframe"] = tf
                    inherited_payload["derived_via"] = "macro_resample"
                    inherited_quality["audit_payload"] = inherited_payload
                    inherited_quality["derived_from_timeframe"] = parent_tf
                    tf_df.attrs["data_quality"] = inherited_quality
            holographic_stack[tf] = tf_df
        return holographic_stack

    def _quality_status(self, quality: dict | None) -> str | None:
        if not isinstance(quality, dict):
            return None
        status = quality.get("status")
        if status is None:
            status = quality.get("quality_status")
        if status is None:
            return None
        return str(status).strip().upper() or None

    def _enforce_strict_quality_gate(
        self,
        *,
        base_symbol: str,
        asset_class: str,
        holographic_stack: dict[str, pd.DataFrame],
        allow_fail_timeframes: tuple[str, ...] = (),
    ) -> None:
        optional_labels = {
            str(label).strip().upper()
            for label in allow_fail_timeframes
            if str(label).strip()
        }
        for timeframe, tf_df in holographic_stack.items():
            quality = tf_df.attrs.get("data_quality")
            if self._quality_status(quality) != "FAIL":
                continue
            if str(timeframe).strip().upper() in optional_labels:
                continue

            repair_symbol = tf_df.attrs.get("pair_symbol") or base_symbol
            raise DataIntegrityError(
                "Strict runtime gating blocked inference for "
                f"{base_symbol} {asset_class}. "
                f"Timeframe={timeframe} failed data_quality firewall. "
                f"data_quality={quality!r}. "
                "Repair with: "
                f"`uv run python data_vault/yfinance_vault.py --full-refresh --symbol {repair_symbol}`"
            )

    def fetch_holographic_stack(
        self,
        base_symbol: str,
        asset_class: str,
        *,
        include_realized_vol: bool = False,
        strict_gating: bool = True,
        allow_fail_timeframes: tuple[str, ...] = (),
    ) -> dict[str, pd.DataFrame]:
        """
        Pulls the complete multi-timeframe dataset for a specific asset.
        Converts the UTC database time back to IST ('Asia/Kolkata') for ML execution.
        """
        try:
            raw_df = self.fetch_holographic_stack_sql(
                base_symbol,
                asset_class,
                include_realized_vol=include_realized_vol,
            )

            if raw_df.empty:
                print(
                    f"[!] DATABASE WARNING: No data found for {base_symbol} {asset_class}."
                )
                return {}
            holographic_stack = self.fetch_holographic_stack_pandas(raw_df)
            if strict_gating:
                self._enforce_strict_quality_gate(
                    base_symbol=base_symbol,
                    asset_class=asset_class,
                    holographic_stack=holographic_stack,
                    allow_fail_timeframes=allow_fail_timeframes,
                )
            return holographic_stack

        except DataIntegrityError:
            raise
        except (sqlite3.Error, ValueError) as e:
            print(f"[!] DATABASE ERROR: {e}")
            return {}

    def fetch_vix_series(
        self,
        market_symbol: str,
    ) -> pd.DataFrame | None:
        """Fetch the VIX companion 1D series for a given market symbol.

        Returns a DataFrame with time/open/high/low/close/volume columns,
        or None if no VIX companion exists or data is unavailable.
        VIX is optional enrichment — never raises on missing data.
        """
        vix_symbol = VIX_COMPANION_MAP.get(market_symbol.strip().upper())
        if vix_symbol is None:
            return None
        try:
            stack = self.fetch_holographic_stack(
                vix_symbol,
                "SPOT",
                include_realized_vol=False,
                strict_gating=False,
            )
            df_vix = stack.get("1D")
            if df_vix is None or df_vix.empty:
                return None
            if self._quality_status(df_vix.attrs.get("data_quality")) == "FAIL":
                return None
            required = {"time", "open", "high", "low", "close"}
            if not required.issubset(set(df_vix.columns)):
                return None
            return df_vix.reset_index(drop=True)
        except Exception:
            return None


if __name__ == "__main__":
    bridge = InferenceBridge(db_path="data_vault/ohlcv.db")

    print("Initiating Database Uplink...")
    stack = bridge.fetch_holographic_stack("NIFTY", "SPOT")

    if stack:
        print(f"Uplink Successful. Timeframes loaded: {list(stack.keys())}")
        for tf, df in stack.items():
            latest = df["time"].iloc[-1]
            source_exchange = df.attrs.get("resolved_source_exchange")
            source_symbol = df.attrs.get("resolved_source_symbol")
            print(
                f"[{tf}] Matrix Shape: {df.shape} | Latest Timestamp: {latest} | "
                f"Source: {source_exchange}:{source_symbol}"
            )
    else:
        print("Uplink Failed. Check the registry entry and source identity rows.")
