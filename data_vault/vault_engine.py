import os
import glob
import sqlite3
import re
import pandas as pd
import numpy as np

class DataVault:
    def __init__(self, db_path='ohlcv.db', inbox_path='inbox'):
        """Initializes the Data Vault, ensuring paths and database schemas exist."""
        self.db_path = db_path
        self.inbox_path = inbox_path
        self.conn = sqlite3.connect(self.db_path)
        
        # Ensure inbox exists
        if not os.path.exists(self.inbox_path):
            os.makedirs(self.inbox_path)
            
        self._build_schema()
        
        # Pre-compile regex for maximum CPU efficiency on i7
        self.re_symbol = re.compile(r'SYMBOL:\s*([^,]+)')
        self.re_open = re.compile(r'OPEN:\s*([\d,\.]+)')
        self.re_high = re.compile(r'HIGH:\s*([\d,\.]+)')
        self.re_low = re.compile(r'LOW:\s*([\d,\.]+)')
        self.re_close = re.compile(r'CLOSE:\s*([\d,\.]+)')
        self.re_vol = re.compile(r'VOLUME:\s*([\d,\.]+)')
        self.re_tf = re.compile(r'TIME FRAME:\s*([^,]+)')
        
        # The Universal Timeframe Rosetta Stone
        self.tf_map = {
            "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
            "60": "1H", "120": "2H", "240": "4H",
            "1D": "1D", "D": "1D", "1W": "1W", "W": "1W", "1M": "1M", "M": "1M"
        }

    def _build_schema(self):
        """Constructs the normalized SQLite relational matrix."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_dna (
                base_symbol TEXT,
                asset_class TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                is_synthetic_vol BOOLEAN,
                PRIMARY KEY (base_symbol, asset_class, timeframe, timestamp)
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
        if not trade_list: return
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
              ON p.base_symbol = m.base_symbol 
              AND p.timestamp = m.timestamp
              AND m.timeframe = '1H'
            WHERE p.base_symbol = ?
              AND p.timestamp >= datetime('now', '-{days} days')
            ORDER BY p.timestamp ASC
        '''
        return pd.read_sql_query(query, self.conn, params=(base_symbol,))

    def _parse_ticker_dna(self, raw_ticker):
        """Assumes TradingView convention: '!' or '1!' suffix = futures. Non-TV tickers ending in '!' will be misclassified."""
        raw_ticker = raw_ticker.strip()
        
        # Refined Regex: Matches an optional '1' followed by '!' at the VERY end
        # This prevents stripping '50' from 'NIFTY50!'
        cleaned_ticker = re.sub(r'1?!$', '', raw_ticker) 
        
        asset_class = 'FUT' if cleaned_ticker != raw_ticker else 'SPOT'
        return cleaned_ticker, asset_class

    def _extract_message_data(self, message):
        """The brutal string parser. Hunts and extracts values, stripping commas."""
        try:
            symbol_raw = self.re_symbol.search(message).group(1).strip()
            base_symbol, asset_class = self._parse_ticker_dna(symbol_raw)
            
            tf_raw = self.re_tf.search(message).group(1).strip().strip('"')
            timeframe = self.tf_map.get(tf_raw, tf_raw) # Default to raw if not in map
            
            # Extract and clean floats
            o = float(self.re_open.search(message).group(1).replace(',', ''))
            h = float(self.re_high.search(message).group(1).replace(',', ''))
            l = float(self.re_low.search(message).group(1).replace(',', ''))
            c = float(self.re_close.search(message).group(1).replace(',', ''))
            v = float(self.re_vol.search(message).group(1).replace(',', ''))
            
            return pd.Series([base_symbol, asset_class, timeframe, o, h, l, c, v])
        except AttributeError:
            # Failsafe for corrupted rows
            return pd.Series([None, None, None, None, None, None, None, None])

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
            'base_symbol': 'first', 'asset_class': 'first',
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'is_synthetic_vol': 'max'
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

    def process_inbox(self):
        """Orchestrates the ingestion, cleaning, resampling, and persistence."""
        csv_files = glob.glob(os.path.join(self.inbox_path, '*.csv'))
        
        if not csv_files:
            print("Inbox is empty. No data to ingest.")
            return

        for file in csv_files:
            print(f"Ingesting: {os.path.basename(file)}")
            
            # Read CSV. Using header=0 to handle standard TradingView export headers
            try:
                raw_df = pd.read_csv(file, names=['Date', 'Message'], header=0)
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue
                
            # Drop empty rows
            raw_df = raw_df.dropna(subset=['Date', 'Message'])
            
            # Extract data using the Regex Cleaver
            extracted = raw_df['Message'].apply(self._extract_message_data)
            extracted.columns = ['base_symbol', 'asset_class', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
            
            # Merge timestamp securely with IST localization (TradingView exports are typically exchange-local)
            ts = pd.to_datetime(raw_df['Date'])
            if ts.dt.tz is None:
                extracted['timestamp'] = ts.dt.tz_localize('Asia/Kolkata')
            else:
                extracted['timestamp'] = ts.dt.tz_convert('Asia/Kolkata')

            
            # Drop rows where regex failed
            extracted = extracted.dropna(subset=['base_symbol'])
            
            # Group by asset only, allowing cross-timeframe examination
            grouped = extracted.groupby(['base_symbol', 'asset_class'])
            
            final_dfs_to_db = []
            
            for (sym, asset), group_df in grouped:
                group_df = group_df.sort_values('timestamp')
                
                # Apply Volume Interpolation (Crucial for FUT)
                clean_df = self._interpolate_volume(group_df)
                
                df_1d = clean_df[clean_df['timeframe'] == '1D']
                df_1h = clean_df[clean_df['timeframe'] == '1H']
                
                # THE SCHEDULE MATRIX CRUCIBLE
                if not df_1d.empty and not df_1h.empty:
                    # Build the Active Date Matrix
                    active_dates = set(df_1d['timestamp'].dt.date)
                    hourly_dates = set(df_1h['timestamp'].dt.date)
                    
                    missing_in_1h = active_dates - hourly_dates
                    if missing_in_1h:
                        print(f"  [!] FATAL INGESTION ERROR: {sym} {asset} Matrix Fragmented.")
                        print(f"      1D file reports {len(missing_in_1h)} active days completely missing from the 1H file.")
                        print("      Aborting ingestion for this asset to preserve data sovereignty.")
                        continue # Skip this asset entirely
                        
                    print(f"  [+] Schedule Matrix Verified: Intraday parity perfectly matches Daily grid for {sym}.")

                # Append the clean base layers (e.g., 1H, 1D)
                final_dfs_to_db.append(clean_df)
                
                # THE MACRO FORGE TRIGGER
                if not df_1d.empty:
                    print(f"  [+] Forging Macro Layers for {sym} {asset}...")
                    macro_df = self._generate_macro_layers(df_1d)
                    if not macro_df.empty:
                        final_dfs_to_db.append(macro_df)

            # Concatenate all generated data
            master_df = pd.concat(final_dfs_to_db, ignore_index=True)
            
            # Execute UPSERT into SQLite using Pandas to_sql with an auxiliary temp table
            self._upsert_dataframe(master_df)
            
            # Remove file after successful ingestion
            os.remove(file)
            print("Successfully processed and archived into database.\n")

    def _upsert_dataframe(self, df):
        """Performs a highly efficient SQLite INSERT OR REPLACE."""
        df_db = df.copy()
        
        # Ensure timestamp is correctly formatted as UTC ISO8601 for SQLite storage
        ts = df_db['timestamp']
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, utc=True)
        
        # Defensive check: if it somehow arrives naive, assume it's IST from the TV export
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize('Asia/Kolkata')
            
        # VAULT-1 FIX: Convert to UTC and strictly format with the +00:00 offset
        df_db['timestamp'] = ts.dt.tz_convert('UTC').dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # Write to temporary table
        df_db.to_sql('temp_market_dna', self.conn, if_exists='replace', index=False)
        
        # Execute UPSERT
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO market_dna (base_symbol, asset_class, timeframe, timestamp, open, high, low, close, volume, is_synthetic_vol)
            SELECT base_symbol, asset_class, timeframe, timestamp, open, high, low, close, volume, is_synthetic_vol
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
