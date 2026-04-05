import glob
import os
import re
import sqlite3

import numpy as np
import pandas as pd

try:
    from symbol_identity import extract_symbol_payload, parse_symbol_payload
except ImportError:
    from data_vault.symbol_identity import (
        extract_symbol_payload,
        parse_symbol_payload,
    )

NUMERIC_TOKEN_PATTERN = r"([-+\d,\.]+|NA)"
REALIZED_VOL_ANNUALIZATION = {"1H": 1512.0, "1D": 252.0, "1W": 52.0}
UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
RAW_LOG_LINE_RE = re.compile(
    r"^\s*\[?(?P<timestamp>.+?)\]?\s*:\s*(?P<message>SYMBOL:.*)$"
)
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

class DataVault:
    def __init__(self, db_path='ohlcv.db', inbox_path='inbox'):
        """Initializes the Data Vault, ensuring paths and database schemas exist."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = (
            db_path if os.path.isabs(db_path) else os.path.join(base_dir, db_path)
        )
        self.inbox_path = (
            inbox_path
            if os.path.isabs(inbox_path)
            else os.path.join(base_dir, inbox_path)
        )
        self.conn = sqlite3.connect(self.db_path)
        
        # Ensure inbox exists
        if not os.path.exists(self.inbox_path):
            os.makedirs(self.inbox_path)
            
        self._build_schema()
        
        # Pre-compile regex for maximum CPU efficiency on i7
        self.re_open = re.compile(rf'OPEN:\s*{NUMERIC_TOKEN_PATTERN}')
        self.re_high = re.compile(rf'HIGH:\s*{NUMERIC_TOKEN_PATTERN}')
        self.re_low = re.compile(rf'LOW:\s*{NUMERIC_TOKEN_PATTERN}')
        self.re_close = re.compile(rf'CLOSE:\s*{NUMERIC_TOKEN_PATTERN}')
        self.re_vol = re.compile(rf'VOLUME:\s*{NUMERIC_TOKEN_PATTERN}')
        self.re_tf = re.compile(r'TIME FRAME:\s*([^,]+)')
        
        # The Universal Timeframe Rosetta Stone
        self.tf_map = {
            "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
            "60": "1H", "120": "2H", "240": "4H",
            "1D": "1D", "D": "1D", "1W": "1W", "W": "1W", "1M": "1M", "M": "1M"
        }

    def _market_dna_exists(self, cursor: sqlite3.Cursor) -> bool:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'market_dna'"
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
        self, cursor: sqlite3.Cursor, table_name: str = "market_dna"
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

    def _build_schema(self):
        """Constructs the normalized SQLite relational matrix."""
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
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ml_inference
            ON market_dna (base_symbol, asset_class, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pair_inference
            ON market_dna (pair_symbol, asset_class, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_identity
            ON market_dna (source_exchange, source_symbol, asset_class, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pair_source_identity
            ON market_dna (
                pair_symbol, asset_class, source_exchange, source_symbol, timestamp
            )
        ''')
        cursor.execute('''
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
        ''')
        self.conn.commit()

    def log_trade_result(self, data_packet):
        """Persists live/backtest outcomes for Shadow Brain training."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO performance_ledger (
                timestamp, base_symbol, direction, conf_score,
                entry_price, exit_price, pnl_r, win_loss_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_packet.get('timestamp'),
            data_packet.get('base_symbol'),
            data_packet.get('direction'),
            data_packet.get('conf_score'),
            data_packet.get('entry_price'),
            data_packet.get('exit_price'),
            data_packet.get('pnl_r'),
            data_packet.get('win_loss_target')
        ))
        self.conn.commit()

    def log_bulk_trades(self, trade_list):
        """Ultra-fast bulk seeding of the performance ledger."""
        if not trade_list:
            return
        cursor = self.conn.cursor()
        data_tuples = [(
            t.get('timestamp'),
            t.get('base_symbol'),
            t.get('direction'),
            t.get('conf_score'),
            t.get('entry_price'),
            t.get('exit_price'),
            t.get('pnl_r'),
            t.get('win_loss_target')
        ) for t in trade_list]
        cursor.executemany('''
            INSERT INTO performance_ledger (
                timestamp, base_symbol, direction, conf_score,
                entry_price, exit_price, pnl_r, win_loss_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_tuples)
        self.conn.commit()

    def get_shadow_train_set(self, base_symbol: str, days=10000):
        """Returns joined OHLCV + Performance data STRICTLY for the target symbol."""
        query = f'''
            SELECT p.*, m.open, m.high, m.low, m.close, m.volume
            FROM performance_ledger p
            LEFT JOIN market_dna m 
              ON p.base_symbol = COALESCE(m.pair_symbol, m.base_symbol)
              AND datetime(p.timestamp) = datetime(m.timestamp)
              AND m.timeframe = '1H'
              AND m.asset_class = 'FUT'
            WHERE p.base_symbol = ?
              AND p.timestamp >= datetime('now', '-{days} days')
            ORDER BY p.timestamp ASC
        '''
        return pd.read_sql_query(query, self.conn, params=(base_symbol,))

    def _parse_ticker_dna(self, raw_ticker):
        """Parses one symbol payload into the normalized exchange-aware identity."""
        identity = parse_symbol_payload(raw_ticker)
        return (
            identity.base_symbol,
            identity.pair_symbol,
            identity.asset_class,
            identity.source_exchange,
            identity.source_symbol,
            identity.symbol_payload_json,
            identity.contract_kind,
            identity.quote_asset,
            identity.expiry_code,
        )

    def _extract_message_data(self, message):
        """Extracts the full symbol payload and the numeric bar fields."""
        message = str(message)
        base_symbol = None
        pair_symbol = None
        asset_class = None
        source_exchange = None
        source_symbol = None
        symbol_payload_json = None
        contract_kind = None
        quote_asset = None
        expiry_code = None
        timeframe = None

        symbol_payload = extract_symbol_payload(message)
        if symbol_payload is not None:
            (
                base_symbol,
                pair_symbol,
                asset_class,
                source_exchange,
                source_symbol,
                symbol_payload_json,
                contract_kind,
                quote_asset,
                expiry_code,
            ) = self._parse_ticker_dna(symbol_payload)

        try:
            tf_match = self.re_tf.search(message)
            if tf_match:
                tf_raw = tf_match.group(1).strip().strip('"')
                timeframe = self.tf_map.get(tf_raw, tf_raw)
        except Exception:
            timeframe = None

        def _parse_numeric(match: re.Match[str] | None) -> float:
            if match is None:
                return np.nan
            raw_value = match.group(1).strip()
            if raw_value == 'NA':
                return np.nan
            try:
                return float(raw_value.replace(',', ''))
            except (TypeError, ValueError):
                return np.nan

        try:
            o = _parse_numeric(self.re_open.search(message))
        except Exception:
            o = np.nan
        try:
            h = _parse_numeric(self.re_high.search(message))
        except Exception:
            h = np.nan
        try:
            lo = _parse_numeric(self.re_low.search(message))
        except Exception:
            lo = np.nan
        try:
            c = _parse_numeric(self.re_close.search(message))
        except Exception:
            c = np.nan
        try:
            v = _parse_numeric(self.re_vol.search(message))
        except Exception:
            v = np.nan

        return pd.Series(
            [
                base_symbol,
                source_exchange,
                source_symbol,
                symbol_payload_json,
                pair_symbol,
                asset_class,
                contract_kind,
                quote_asset,
                expiry_code,
                timeframe,
                o,
                h,
                lo,
                c,
                v,
            ]
        )

    def _load_inbox_file(self, path: str) -> pd.DataFrame:
        """
        Loads either a standard TradingView CSV export or a raw log file where
        each line embeds its own timestamp prefix before `SYMBOL:`.
        """
        records: list[tuple[str, str]] = []
        with open(path, encoding="utf-8", newline="") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                raw_match = RAW_LOG_LINE_RE.match(line)
                if raw_match is not None:
                    records.append(
                        (
                            raw_match.group("timestamp").strip(),
                            raw_match.group("message").strip(),
                        )
                    )
                    continue

                if "," not in raw_line:
                    continue
                first, remainder = raw_line.split(",", 1)
                first = first.strip().lstrip("\ufeff")
                second = remainder.strip()
                if first.lower() == "date" and second.lower().startswith("message"):
                    continue

                message = second
                if message.startswith('"') and message.endswith('"') and len(message) >= 2:
                    message = message[1:-1].replace('""', '"')
                if message:
                    records.append((first, message))

        if not records:
            return pd.DataFrame(columns=["Date", "Message"])
        return pd.DataFrame.from_records(records, columns=["Date", "Message"])

    def _parse_timestamp_series(self, series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce")
        if ts.notna().sum() == 0:
            return ts
        if ts.dt.tz is None:
            return ts.dt.tz_localize("Asia/Kolkata")
        return ts.dt.tz_convert("Asia/Kolkata")

    def _validate_ingestion_batch(self, df: pd.DataFrame, file_name: str) -> None:
        identity_cols = [
            "source_exchange",
            "source_symbol",
            "asset_class",
            "timeframe",
            "timestamp",
        ]
        identity_df = df.dropna(subset=identity_cols)
        if identity_df.empty:
            return

        duplicate_mask = identity_df.duplicated(identity_cols, keep=False)
        if not duplicate_mask.any():
            return

        sample_cols = [
            "pair_symbol",
            "source_exchange",
            "source_symbol",
            "asset_class",
            "timeframe",
            "timestamp",
        ]
        sample = identity_df.loc[duplicate_mask, sample_cols].head(5).to_dict("records")
        raise ValueError(
            f"Duplicate source-identity rows detected in {file_name}: {sample}"
        )

    def _enforce_physics(self, df):
        """Enforces OHLC structural bounds before any downstream repair logic."""
        df = df.copy()
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        return df

    def _interpolate_volume(self, df):
        """Surgically repairs 0 Volume using volatility-matched proxy logic."""
        df = df.copy()
        
        # Calculate True Range
        df['TR'] = df['high'] - df['low']
        
        # Identify broken volume
        zero_vol_mask = (df['volume'] == 0.0) | (df['volume'].isna())
        df['is_synthetic_vol'] = zero_vol_mask
        
        if not zero_vol_mask.any():
            return df.drop(columns=['TR'])

        # Calculate 5-bar rolling metrics (centered if possible, or forward/backward filled)
        # Using min_periods=1 to ensure edges don't become NaN
        rolling_tr = df['TR'].replace(0, np.nan).rolling(window=5, min_periods=1, center=True).mean()
        rolling_vol = df['volume'].replace(0, np.nan).rolling(window=5, min_periods=1, center=True).mean()
        
        # Volatility to Volume Ratio
        vol_tr_ratio = rolling_vol / rolling_tr
        
        # Apply synthetic volume where missing: (Current TR) * (Surrounding Vol/TR Ratio)
        synthetic_vol = df['TR'] * vol_tr_ratio
        
        # If TR was 0 (Open=High=Low=Close), default to the rolling volume average
        synthetic_vol = np.where(df['TR'] == 0, rolling_vol, synthetic_vol)
        
        df.loc[zero_vol_mask, 'volume'] = synthetic_vol[zero_vol_mask]
        
        # Final cleanup for any lingering NaNs at the extreme edges
        df['volume'] = df['volume'].ffill().bfill()
        
        return df.drop(columns=['TR'])

    def _compute_realized_volatility(self, df):
        """Vectorizes 21-period bipower variation and backfills initial gaps."""
        df = df.copy()
        df['realized_volatility'] = np.nan

        for timeframe, annualization_factor in REALIZED_VOL_ANNUALIZATION.items():
            tf_mask = df['timeframe'] == timeframe
            if not tf_mask.any():
                continue

            tf_df = df.loc[tf_mask].sort_values('timestamp').copy()
            closes = tf_df['close'].where(tf_df['close'] > 0)

            with np.errstate(divide='ignore', invalid='ignore'):
                log_returns = np.log(closes / closes.shift(1))

            bpv_component = (np.pi / 2.0) * log_returns.abs() * log_returns.shift(1).abs()
            rolling_sum = bpv_component.rolling(window=21, min_periods=2).sum()
            rv = np.sqrt(np.maximum(rolling_sum * annualization_factor, 0.0))
            rv = rv.bfill().fillna(0.0)

            df.loc[tf_df.index, 'realized_volatility'] = rv.to_numpy()

        return df

    def _generate_macro_layers(self, df_1d):
        """Forges macro regimes natively from the data that physically exists, immune to exchange schedules."""
        macro_dfs = []
        df = df_1d.copy().sort_values('timestamp')
        
        # 1. Extract Native Temporal DNA
        df['iso_year'] = df['timestamp'].dt.isocalendar().year
        df['iso_week'] = df['timestamp'].dt.isocalendar().week
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['half'] = np.where(df['month'] <= 6, 1, 2)
        
        # 2. Define Deterministic Grouping Signatures
        signatures = {
            '1W': ['iso_year', 'iso_week'],
            '1M': ['year', 'month'],
            '3M': ['year', 'quarter'],
            '6M': ['year', 'half'],
            '12M': ['year']
        }
        
        # 3. Aggregation Physics
        agg_logic = {
            'timestamp': 'first',
            'base_symbol': 'first',
            'source_exchange': 'first',
            'source_symbol': 'first',
            'symbol_payload_json': 'first',
            'pair_symbol': 'first',
            'asset_class': 'first',
            'contract_kind': 'first',
            'quote_asset': 'first',
            'expiry_code': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'is_synthetic_vol': 'max',
        }
        
        for label, group_cols in signatures.items():
            try:
                # Group strictly by data points that exist
                df_macro = df.groupby(group_cols).agg(agg_logic).reset_index(drop=True)
                
                # VAULT-1 FIX: Force the aggregated 'first' timestamp back to UTC-aware
                df_macro['timestamp'] = pd.to_datetime(df_macro['timestamp'], utc=True)
                
                df_macro['timeframe'] = label
                macro_dfs.append(df_macro)
            except Exception as e:
                print(f"  [!] Time-Cipher Forge failed for {label}: {e}")
                
        return pd.concat(macro_dfs, ignore_index=True) if macro_dfs else pd.DataFrame()

    def _run_global_schedule_sync(self):
        """Prunes any source/asset rows whose trade date is outside the 1D/1H sync set."""
        cursor = self.conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS temp_sync_dates')
        cursor.execute('''
            CREATE TEMP TABLE temp_sync_dates AS
            SELECT
                d.source_exchange_key,
                d.source_symbol_key,
                d.asset_class,
                d.trade_date
            FROM (
                SELECT DISTINCT
                    COALESCE(source_exchange, '') AS source_exchange_key,
                    COALESCE(source_symbol, base_symbol) AS source_symbol_key,
                    asset_class,
                    DATE(timestamp, '+05:30') AS trade_date
                FROM market_dna
                WHERE timeframe = '1D'
            ) AS d
            INNER JOIN (
                SELECT DISTINCT
                    COALESCE(source_exchange, '') AS source_exchange_key,
                    COALESCE(source_symbol, base_symbol) AS source_symbol_key,
                    asset_class,
                    DATE(timestamp, '+05:30') AS trade_date
                FROM market_dna
                WHERE timeframe = '1H'
            ) AS h
              ON d.source_exchange_key = h.source_exchange_key
             AND d.source_symbol_key = h.source_symbol_key
             AND d.asset_class = h.asset_class
             AND d.trade_date = h.trade_date
        ''')
        cursor.execute('''
            SELECT COUNT(*)
            FROM market_dna AS m
            WHERE NOT EXISTS (
                SELECT 1
                FROM temp_sync_dates AS s
                WHERE s.source_exchange_key = COALESCE(m.source_exchange, '')
                  AND s.source_symbol_key = COALESCE(m.source_symbol, m.base_symbol)
                  AND s.asset_class = m.asset_class
                  AND s.trade_date = DATE(m.timestamp, '+05:30')
            )
        ''')
        rows_pruned = cursor.fetchone()[0]
        cursor.execute('''
            DELETE FROM market_dna
            WHERE NOT EXISTS (
                SELECT 1
                FROM temp_sync_dates AS s
                WHERE s.source_exchange_key = COALESCE(market_dna.source_exchange, '')
                  AND s.source_symbol_key = COALESCE(
                      market_dna.source_symbol,
                      market_dna.base_symbol
                  )
                  AND s.asset_class = market_dna.asset_class
                  AND s.trade_date = DATE(market_dna.timestamp, '+05:30')
            )
        ''')
        cursor.execute('DROP TABLE temp_sync_dates')
        self.conn.commit()
        print(f"  [+] Global schedule sync pruned {rows_pruned} unsynced rows.")

    def process_inbox(self):
        """Orchestrates the ingestion, cleaning, resampling, and persistence."""
        inbox_files: list[str] = []
        for pattern in ("*.csv", "*.log", "*.txt"):
            inbox_files.extend(glob.glob(os.path.join(self.inbox_path, pattern)))
        csv_files = sorted(set(inbox_files))
        
        if not csv_files:
            print("Inbox is empty. No data to ingest.")
            return

        for file in csv_files:
            print(f"Ingesting: {os.path.basename(file)}")
            
            try:
                raw_df = self._load_inbox_file(file)
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue
                
            # Drop empty rows
            raw_df = raw_df.dropna(subset=['Date', 'Message'])
            
            # Extract data using the Regex Cleaver
            try:
                extracted = raw_df['Message'].apply(self._extract_message_data)
            except ValueError as e:
                print(f"  [!] Identity parse failed for {os.path.basename(file)}: {e}")
                print("  [!] Source left in inbox for correction.\n")
                continue
            extracted.columns = [
                'base_symbol',
                'source_exchange',
                'source_symbol',
                'symbol_payload_json',
                'pair_symbol',
                'asset_class',
                'contract_kind',
                'quote_asset',
                'expiry_code',
                'timeframe',
                'open',
                'high',
                'low',
                'close',
                'volume',
            ]
            
            # Merge timestamp securely with IST localization for naive exports,
            # while preserving explicit source offsets when they exist.
            extracted['timestamp'] = self._parse_timestamp_series(raw_df['Date'])

            
            # Preserve rows with broken volume, but reject rows missing structural inputs.
            extracted = extracted.dropna(
                subset=[
                    'base_symbol',
                    'source_symbol',
                    'pair_symbol',
                    'asset_class',
                    'contract_kind',
                    'timeframe',
                    'timestamp',
                    'open',
                    'high',
                    'low',
                    'close',
                ]
            )
            try:
                self._validate_ingestion_batch(extracted, os.path.basename(file))
            except ValueError as e:
                print(f"  [!] Integrity failure for {os.path.basename(file)}: {e}")
                print("  [!] Source left in inbox for correction.\n")
                continue
            
            # Group by exact source identity so same-symbol multi-venue rows never mix.
            grouped = extracted.groupby(
                ['source_exchange', 'source_symbol', 'asset_class'],
                dropna=False,
            )
            
            final_dfs_to_db = []
            
            for (exchange, source_symbol, asset), group_df in grouped:
                group_df = group_df.sort_values('timestamp')
                
                clean_df = self._enforce_physics(group_df)
                clean_df = self._interpolate_volume(clean_df)
                clean_df = self._compute_realized_volatility(clean_df)
                
                df_1d = clean_df[clean_df['timeframe'] == '1D']

                # Append the clean base layers (e.g., 1H, 1D)
                final_dfs_to_db.append(clean_df)
                
                # THE MACRO FORGE TRIGGER
                if not df_1d.empty:
                    print(
                        f"  [+] Forging Macro Layers for "
                        f"{exchange}:{source_symbol} {asset}..."
                    )
                    macro_df = self._generate_macro_layers(df_1d)
                    if not macro_df.empty:
                        macro_df = self._compute_realized_volatility(macro_df)
                        final_dfs_to_db.append(macro_df)

            if not final_dfs_to_db:
                print("  [!] No valid rows extracted from file. Source left in inbox for inspection.\n")
                continue

            # Concatenate all generated data
            master_df = pd.concat(final_dfs_to_db, ignore_index=True)
            
            # Execute UPSERT into SQLite using Pandas to_sql with an auxiliary temp table
            self._upsert_dataframe(master_df)
            
            # Remove file after successful ingestion
            os.remove(file)
            print("Successfully processed and archived into database.\n")

        self._run_global_schedule_sync()

    def _upsert_dataframe(self, df):
        """Performs a highly efficient SQLite INSERT OR REPLACE."""
        df_db = df.copy()
        for col in (
            'source_exchange',
            'source_symbol',
            'symbol_payload_json',
            'pair_symbol',
            'contract_kind',
            'quote_asset',
            'expiry_code',
        ):
            if col not in df_db.columns:
                df_db[col] = np.nan
        if 'realized_volatility' not in df_db.columns:
            df_db['realized_volatility'] = np.nan
        
        # Ensure timestamp is correctly formatted as UTC ISO8601 for SQLite storage
        ts = df_db['timestamp']
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, utc=True)
        
        # Defensive check: if it somehow arrives naive, assume it's IST from the TV export
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize('Asia/Kolkata')
            
        # VAULT-1 FIX: Convert to UTC and strictly format with the +00:00 offset
        df_db['timestamp'] = ts.dt.tz_convert('UTC').dt.strftime(UTC_TIMESTAMP_FORMAT)
        self._validate_ingestion_batch(df_db, "upsert_batch")
        
        # Write to temporary table
        df_db.to_sql('temp_market_dna', self.conn, if_exists='replace', index=False)
        
        # Execute UPSERT
        cursor = self.conn.cursor()
        cursor.execute('''
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
            FROM temp_market_dna
        ''')
        self.conn.commit()
        
        # Drop temp table
        cursor.execute('DROP TABLE temp_market_dna')
        self.conn.commit()

# --- Execution Block ---
if __name__ == "__main__":
    vault = DataVault()
    print("Initiating TOON v5 Data Sovereign...")
    vault.process_inbox()
    print("Ingestion Cycle Complete. Database locked and ready.")
