import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb

# Dynamically add data_vault into path
sys.path.append(os.path.join(os.path.dirname(__file__), "data_vault"))
try:
    from vault_engine import DataVault
except ImportError:
    pass


class ShadowBrain:
    def __init__(self, base_symbol: str, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(__file__)
            db_path = os.path.join(base_dir, "data_vault", "ohlcv.db")

        self.base_symbol = base_symbol
        self.db_path = db_path
        self.model = None
        self.is_trained = False
        self.feature_cols = []

    def _build_model(self):
        return lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
            n_jobs=4,
            verbose=-1,
            objective="binary",
        )

    def train(self, days=10000):
        """Pulls from vault using strict Symbol Quarantine."""
        try:
            vault = DataVault(db_path=self.db_path)
            df = vault.get_shadow_train_set(base_symbol=self.base_symbol, days=days)
        except Exception as e:
            print(f"  [SHADOW] Error accessing vault: {e}")
            return False

        if df.empty or len(df) < 15:
            print(
                "  [SHADOW] Insufficient trade records for meta-modeling. Bypass engaged."
            )
            return False

        # Target: win_loss_target == 1 means success, 0 means loss
        if "win_loss_target" not in df.columns:
            print("  [SHADOW] Warning: No target found in performance ledger.")
            return False

        # Drop rows where execution hasn't completed yet
        df = df.dropna(subset=["win_loss_target"]).reset_index(drop=True)

        if len(df) < 15 or df["win_loss_target"].nunique() < 2:
            print(
                "  [SHADOW] Target requires both wins and losses (>15). Bypass engaged."
            )
            return False

        # Build feature set dynamically from joined OHLCV
        base_exclusions = [
            "timestamp",
            "base_symbol",
            "direction",
            "conf_score",
            "entry_price",
            "exit_price",
            "pnl_r",
            "win_loss_target",
        ]
        self.feature_cols = [c for c in df.columns if c not in base_exclusions]

        if not self.feature_cols:
            print("  [SHADOW] No OHLCV features found via vault join.")
            return False

        X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df["win_loss_target"].astype(int)

        # Apply dynamic exponential temporal decay weighting
        # Half-life of 30 days
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            current_time = df["timestamp"].max()
            delta_days = (current_time - df["timestamp"]).dt.days
            weights = np.exp(-np.log(2) * delta_days / 30.0)
        except Exception:
            weights = np.ones(len(df))

        self.model = self._build_model()
        try:
            self.model.fit(X, y, sample_weight=weights)
            self.is_trained = True
            print(
                f"  [SHADOW] Meta-model trained on {len(df)} records. Features: {len(self.feature_cols)}"
            )
            return True
        except Exception as e:
            print(f"  [SHADOW] Model fitting failed: {e}")
            return False

    def predict_veto(self, current_features: dict) -> bool:
        """
        RETURN True IF Shadow_P(Success) < 0.45 ELSE False.
        """
        if not self.is_trained or self.model is None or not self.feature_cols:
            return False  # Fail open if no shadow model exists

        row_dict = {}
        for col in self.feature_cols:
            row_dict[col] = current_features.get(col, 0.0)

        X_inf = pd.DataFrame([row_dict]).astype(float)

        try:
            prob_success = float(self.model.predict_proba(X_inf)[0, 1])
        except (IndexError, ValueError, Exception) as e:
            print(f"  [SHADOW] Prediction error ({e}), failing open.")
            prob_success = 1.0

        if prob_success < 0.45:
            print(f"  [SHADOW] VETO TRIGGERED: P(Success) = {prob_success:.2f} < 0.45")
            return True

        print(f"  [SHADOW] APPROVED: P(Success) = {prob_success:.2f}")
        return False
