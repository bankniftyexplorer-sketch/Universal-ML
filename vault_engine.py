"""
Compatibility shim for the data-vault engine.

The canonical implementation lives in ``data_vault/vault_engine.py``.
Several root-level scripts still import ``vault_engine`` directly, so this
module keeps those imports stable regardless of where ``--outdir`` points.
"""

from data_vault.vault_engine import DataVault

__all__ = ["DataVault"]
