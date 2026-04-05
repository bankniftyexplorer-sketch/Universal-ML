from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import time

_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)
_MIN_SESSION_BARS = 78


def bpv_estimator(
    rows: Sequence[tuple[object, object, object, object, object]],
) -> float | None:
    closes: list[float] = []
    session_dates: set[object] = set()

    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            return None
        ts = row[0]
        if not hasattr(ts, "date") or not hasattr(ts, "time"):
            return None
        if ts.time() < _MARKET_OPEN or ts.time() > _MARKET_CLOSE:
            continue
        try:
            close = float(row[4])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(close) or close <= 0:
            continue
        session_dates.add(ts.date())
        closes.append(close)

    if len(session_dates) != 1 or len(closes) < _MIN_SESSION_BARS:
        return None

    log_returns: list[float] = []
    for prev_close, close in zip(closes, closes[1:]):
        if prev_close <= 0 or close <= 0:
            continue
        log_returns.append(math.log(close / prev_close))

    if len(log_returns) < 2:
        return None

    bpv_variance = (
        (math.pi / 2.0)
        * sum(abs(curr) * abs(prev) for prev, curr in zip(log_returns, log_returns[1:]))
        * 252.0
    )
    return math.sqrt(max(0.0, bpv_variance))


def har_rv_j(
    daily_bpv_history: Sequence[tuple[object, object]],
) -> tuple[float | None, bool]:
    cleaned: list[tuple[object, float]] = []
    for session_date, bpv_value in daily_bpv_history:
        try:
            value = float(bpv_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        cleaned.append((session_date, value))

    if len(cleaned) < 22:
        return None, False

    values = [bpv for _, bpv in cleaned]
    latest_daily = values[-1]
    latest_weekly = _mean(values[-5:])
    latest_monthly = _mean(values[-22:])
    latest_jump = 0.0
    fallback = _mean([latest_daily, latest_weekly, latest_monthly])

    features: list[list[float]] = []
    targets: list[float] = []
    for idx in range(21, len(values) - 1):
        rv_daily = values[idx]
        rv_weekly = _mean(values[idx - 4 : idx + 1])
        rv_monthly = _mean(values[idx - 21 : idx + 1])
        jump_component = 0.0
        features.append([1.0, rv_daily, rv_weekly, rv_monthly, jump_component])
        targets.append(values[idx + 1])

    coeffs = _ols_coefficients(features, targets) if features else None
    if coeffs is None:
        return fallback, True

    forecast = (
        coeffs[0]
        + coeffs[1] * latest_daily
        + coeffs[2] * latest_weekly
        + coeffs[3] * latest_monthly
        + coeffs[4] * latest_jump
    )
    if not math.isfinite(forecast) or forecast <= 0:
        return fallback, True
    return forecast, True


def vrp(atm_iv: float | None, rv_forecast: float | None) -> tuple[float, str] | None:
    if atm_iv is None or rv_forecast is None:
        return None
    if not math.isfinite(atm_iv) or not math.isfinite(rv_forecast):
        return None
    if atm_iv <= 0 or rv_forecast <= 0:
        return None

    value = (atm_iv * atm_iv) - (rv_forecast * rv_forecast)
    if abs(value) <= 1e-12:
        state = "FAIR"
    elif value > 0:
        state = "IV_RICH"
    else:
        state = "IV_CHEAP"
    return value, state


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _ols_coefficients(
    features: Sequence[Sequence[float]], targets: Sequence[float]
) -> list[float] | None:
    if not features or not targets or len(features) != len(targets):
        return None

    width = len(features[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, target in zip(features, targets):
        if len(row) != width:
            return None
        for i in range(width):
            xty[i] += row[i] * target
            for j in range(width):
                xtx[i][j] += row[i] * row[j]

    ridge = 1e-9
    for i in range(width):
        xtx[i][i] += ridge

    return _solve_linear_system(xtx, xty)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float] | None:
    size = len(vector)
    if len(matrix) != size:
        return None

    augmented = [list(row) + [vector[idx]] for idx, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row_idx: abs(augmented[row_idx][col]))
        if abs(augmented[pivot][col]) <= 1e-12:
            return None
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot_value

        for row_idx in range(size):
            if row_idx == col:
                continue
            factor = augmented[row_idx][col]
            if factor == 0:
                continue
            for j in range(col, size + 1):
                augmented[row_idx][j] -= factor * augmented[col][j]

    return [augmented[row_idx][size] for row_idx in range(size)]
