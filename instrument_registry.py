from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from data_vault.symbol_identity import CANONICAL_SYMBOL_ALIASES, canonical_pair_symbol


def _normalize_symbol(value: str) -> str:
    return str(value).strip().upper()


@lru_cache(maxsize=1)
def _reverse_aliases() -> dict[str, tuple[str, ...]]:
    reverse: dict[str, set[str]] = {}
    for alias, canonical in CANONICAL_SYMBOL_ALIASES.items():
        reverse.setdefault(str(canonical).upper(), set()).add(str(alias).upper())
    return {
        canonical: tuple(sorted(aliases))
        for canonical, aliases in reverse.items()
    }


@dataclass(frozen=True)
class InstrumentIdentity:
    request_symbol: str
    canonical_symbol: str
    display_symbol: str
    alias_set: tuple[str, ...]
    artifact_prefix: str
    market_data_symbol: str

    def symbol_dir(self, outdir: str) -> str:
        return os.path.join(outdir, self.canonical_symbol)


def _scan_alias_dirs(
    outdir: str | None,
    *,
    canonical_symbol: str,
    asset_class: str,
) -> set[str]:
    aliases: set[str] = set()
    if not outdir or not os.path.isdir(outdir):
        return aliases

    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if not os.path.isdir(path):
            continue
        candidate = _normalize_symbol(name)
        if canonical_pair_symbol(candidate, asset_class=asset_class) == canonical_symbol:
            aliases.add(candidate)
    return aliases


def resolve_instrument_identity(
    raw_symbol: str,
    *,
    outdir: str | None = None,
    asset_class: str = "SPOT",
) -> InstrumentIdentity:
    request_symbol = _normalize_symbol(raw_symbol)
    canonical_symbol = (
        canonical_pair_symbol(request_symbol, asset_class=asset_class) or request_symbol
    )
    canonical_symbol = _normalize_symbol(canonical_symbol)

    aliases = {request_symbol, canonical_symbol}
    aliases.update(_reverse_aliases().get(canonical_symbol, ()))
    aliases.update(
        _scan_alias_dirs(
            outdir,
            canonical_symbol=canonical_symbol,
            asset_class=asset_class,
        )
    )

    return InstrumentIdentity(
        request_symbol=request_symbol,
        canonical_symbol=canonical_symbol,
        display_symbol=canonical_symbol,
        alias_set=tuple(sorted(aliases)),
        artifact_prefix=canonical_symbol.lower().replace(" ", "_"),
        market_data_symbol=canonical_symbol,
    )
