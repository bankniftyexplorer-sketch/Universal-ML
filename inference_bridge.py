import os
import sqlite3

import pandas as pd

from data_vault.symbol_identity import canonical_pair_symbol, lookup_registry_entry


class InferenceBridge:
    def __init__(self, db_path="data_vault/ohlcv.db"):
        """
        Initializes the connection to the Data Sovereign.
        Ensure the path accurately points to your new ohlcv.db file.
        """
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"[!] CRITICAL: Database not found at {self.db_path}")
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
            expected_source = (registry_entry.source_exchange, registry_entry.source_symbol)
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
            holographic_stack[tf] = tf_df
        return holographic_stack

    def fetch_holographic_stack(
        self,
        base_symbol: str,
        asset_class: str,
        *,
        include_realized_vol: bool = False,
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
                print(f"[!] DATABASE WARNING: No data found for {base_symbol} {asset_class}.")
                return {}
            return self.fetch_holographic_stack_pandas(raw_df)

        except (sqlite3.Error, ValueError) as e:
            print(f"[!] DATABASE ERROR: {e}")
            return {}


if __name__ == "__main__":
    bridge = InferenceBridge(db_path="data_vault/ohlcv.db")

    print("Initiating Database Uplink...")
    stack = bridge.fetch_holographic_stack("BANKNIFTY", "FUT")

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
