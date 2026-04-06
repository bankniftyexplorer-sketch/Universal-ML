"""
Compatibility shim for the automated Yahoo Finance vault.

The canonical ingestion implementation now lives in ``yfinance_vault.py``.
This module preserves the historical ``data_vault/vault_engine.py`` entrypoint.
"""

from __future__ import annotations

try:
    from yfinance_vault import YFinanceVault
except ImportError:
    from data_vault.yfinance_vault import YFinanceVault

DataVault = YFinanceVault

__all__ = ["DataVault", "YFinanceVault"]


def main() -> int:
    try:
        from yfinance_vault import main as _main
    except ImportError:
        from data_vault.yfinance_vault import main as _main
    return _main()


if __name__ == "__main__":
    try:
        from yfinance_vault import main as _main
    except ImportError:
        from data_vault.yfinance_vault import main as _main

    raise SystemExit(_main())
