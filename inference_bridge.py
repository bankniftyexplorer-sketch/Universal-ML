import sqlite3
import pandas as pd
import os

class InferenceBridge:
    def __init__(self, db_path='data_vault/ohlcv.db'):
        """
        Initializes the connection to the Data Sovereign.
        Ensure the path accurately points to your new ohlcv.db file.
        """
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"[!] CRITICAL: Database not found at {self.db_path}")

    def fetch_holographic_stack(self, base_symbol: str, asset_class: str) -> dict:
        """
        Pulls the complete multi-timeframe dataset for a specific asset.
        Converts the UTC database time back to IST ('Asia/Kolkata') for ML execution.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # The Brutally Perfect SQL Query: Pull everything for this specific asset
                query = """
                    SELECT timeframe, timestamp, open, high, low, close, volume, is_synthetic_vol 
                    FROM market_dna 
                    WHERE base_symbol = ? AND asset_class = ?
                    ORDER BY timestamp ASC
                """
                
                # Read directly into a highly-optimized Pandas DataFrame
                raw_df = pd.read_sql_query(query, conn, params=(base_symbol, asset_class))

            if raw_df.empty:
                print(f"[!] DATABASE WARNING: No data found for {base_symbol} {asset_class}.")
                return {}

            # VAULT-1 FIX: Force UTC parsing. Pandas will read the +00:00 and make it TZ-aware.
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], utc=True)
            
            # CRITICAL TRANSLATION: Rename 'timestamp' to 'time' 
            # This ensures the ML engine requires ZERO changes to its internal math.
            raw_df = raw_df.rename(columns={'timestamp': 'time'})
            
            # Safely convert to IST for local execution
            raw_df['time'] = raw_df['time'].dt.tz_convert('Asia/Kolkata')

            # Break the master DataFrame into the Holographic Stack dictionary
            holographic_stack = {}
            timeframes = raw_df['timeframe'].unique()
            
            for tf in timeframes:
                tf_df = raw_df[raw_df['timeframe'] == tf].copy()
                tf_df = tf_df.drop(columns=['timeframe']).reset_index(drop=True)
                holographic_stack[tf] = tf_df
                
            return holographic_stack

        except sqlite3.Error as e:
            print(f"[!] DATABASE ERROR: {e}")
            return {}

# --- Diagnostic Test Block ---
if __name__ == "__main__":
    # Adjust path if you are running this from inside a subfolder
    bridge = InferenceBridge(db_path='ohlcv.db') 
    
    print("Initiating Database Uplink...")
    stack = bridge.fetch_holographic_stack("BANKNIFTY", "FUT")
    
    if stack:
        print(f"Uplink Successful. Timeframes loaded: {list(stack.keys())}")
        for tf, df in stack.items():
            print(f"[{tf}] Matrix Shape: {df.shape} | Latest Timestamp: {df['time'].iloc[-1]}")
    else:
        print("Uplink Failed. Check base_symbol and asset_class exact matches.")
