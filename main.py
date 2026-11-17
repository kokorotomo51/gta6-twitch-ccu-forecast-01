"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecast the Twitch concurrent-viewer count of Grand Theft Auto VI in each
hour 2..24 of its release day (2026-11-19 UTC), as five quantiles an hour,
from everything recorded up to the end of hour 1, hour 1's print included.

Model: a launch-day *shape* model, fitted at run time from the case's own
`comparables` (past big game launches' hourly Twitch curves, SullyGnome).
For a past launch let p1 be its hour-1 print and y_h its hour-h print.
Across launches, log(y_h / p1) depends strongly on log(p1): a launch whose
first hour is already huge (the world already unlocked: UK, Europe,
Asia-Pacific in hour 1) stays near flat and peaks ~1.3x in the evening,
while a launch whose first hour is tiny (its main markets had not unlocked
yet) grows 10-2000x by evening (corr(log p1, log peak ratio) = -0.94 over
the kit's 37 usable past launches). So for each hour h the forecast is a
*local* fit in log p1 around the case's own log p1:

  - local OLS of log(y_h / p1) on log p1 over launches within +-2.0 dex of
    the case's log p1 (window widens until at least 8 launches), and
  - the local median log-ratio over the same launches,
  averaged (ensemble) for the centre, with the spread the average of the
  local OLS residual sd and the local sd of the log-ratios, scaled by 0.85.

Quantiles are log-normal: q_tau = p1 * exp(mu_h + z_tau * sigma_h), with
z_tau the standard normal quantile at the case's levels (Acklam's
approximation below), rounded to 2 decimals half up. A launch is usable
only if its release day has an hour-1 print and at least 12 of the 24
hours (this also drops the GTA VI pre-release comparable, which has no
release-day hours yet). An hour with fewer than 5 usable launches falls
back to the house persistence baseline (spread 0.15 * sqrt(h - 1)).

Backtest (leave-one-launch-out over the kit's comparables, pinball loss
on the log scale, scored on each launch's recorded hours only):
  launches with p1 >= 50,000 (the regime GTA VI will be in; its hour-1
  print will be in the hundreds of thousands):
    this model  0.199  (90% coverage 0.911, typical miss 39.4%)
    house baseline (persistence)  0.251
  launches with p1 >= 20,000:
    this model  0.248 vs baseline 0.321
  all 37 launches: this model 0.384 vs baseline 0.930 (small launches are
  intrinsically far harder: an hour-1 print of a few hundred viewers
  says little about the evening peak).
The house baseline scores ~0.387 on the four practice cases; this model
is expected to score ~0.20-0.25 there (practice answers are not in the
kit, so that is an estimate from the leave-one-out numbers above, not a
measured practice score). No parameter was fitted to the final case: the
final's answer is not recorded yet, and the fit is re-done from the
comparables inside whatever case file the program is handed.

Reads every <id>.in in the input directory and writes <id>.out to the
output directory: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]},
...]}, exactly 23 rows, hours 2..24 each once, no other key, each q five
finite numbers above 0 and at most 1e9, non-decreasing. Deterministic:
no randomness, no network, no time dependence, so two runs write the
same bytes. Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file, but keep the
# starter's guard so the layout stays gradeable either way.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- local shape fit ---
# Half-width (in natural-log units of the hour-1 print) of the window of
# past launches the fit looks at around the case's own log hour-1 print.
WINDOW = 2.0
WINDOW_STEP = 0.5
WINDOW_MAX = 4.0
MIN_LAUNCHES = 8
# A usable past launch needs an hour-1 print and this many release-day hours.
MIN_HOURS = 12
# An hour needs this many usable launches or it falls back to persistence.
MIN_PAIRS = 5
# Spread calibration: local sds are shrunk by this much (leave-one-out
# optimum on big launches is flat between 0.8 and 0.9; 0.85 is the middle).
SIGMA_SCALE = 0.85
SIGMA_FLOOR = 0.10
SIGMA_CAP = 1.0
# Fallback persistence spread (the house baseline's), log units per sqrt hour.
BASE_SIGMA = 0.15


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, ~1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers, keyed by hour 1..24."""
    day_start = parse_ts(comp["day_start"])
    hours: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        h = int((parse_ts(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            hours[h] = float(viewers)
    return hours


def usable_launches(case: dict) -> list[tuple[float, dict[int, float]]]:
    """(log hour-1 print, release-day hours) for each usable comparable."""
    launches = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if 1 in hours and len(hours) >= MIN_HOURS:
            launches.append((math.log(hours[1]), hours))
    return launches


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares y ~ a + b*x; returns (a, b, residual sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.3
    return a, b, sd


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _pstdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.3
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def local_shape(pairs: list[tuple[float, float]], x0: float) -> tuple[float, float] | None:
    """Ensemble centre and spread of log(y_h / p1) near the case's log p1.

    pairs are (log p1 of a past launch, its log ratio at this hour).
    Returns (mu, sigma) in log units, or None when too few pairs exist.
    """
    if len(pairs) < MIN_PAIRS:
        return None
    window = WINDOW
    use = pairs
    while window <= WINDOW_MAX:
        near = [(x, y) for x, y in pairs if abs(x - x0) <= window]
        if len(near) >= MIN_LAUNCHES:
            use = near
            break
        window += WINDOW_STEP
    else:
        use = pairs
    xs = [x for x, _ in use]
    ys = [y for _, y in use]
    a, b, sd_ols = _ols(xs, ys)
    mu_ols = a + b * x0
    mu_med = _median(ys)
    sd_med = _pstdev(ys)
    mu = (mu_ols + mu_med) / 2
    sd = (sd_ols + sd_med) / 2
    sd = min(SIGMA_CAP, max(SIGMA_FLOOR, sd)) * SIGMA_SCALE
    return mu, sd


def hour_quantiles(p1: float, h: int, shape: tuple[float, float] | None,
                   zs: list[float]) -> list[float | int]:
    """The five quantiles of hour h, rounded half up and non-decreasing."""
    if shape is None:
        mu, sd = 0.0, BASE_SIGMA * math.sqrt(h - 1)
    else:
        mu, sd = shape
    qs = [p1 * math.exp(mu + z * sd) for z in zs]
    rounded = [round2(max(0.01, min(1e9, q))) for q in qs]
    # Rounding can only break monotonicity by a cent; enforce it outright.
    for i in range(1, len(rounded)):
        if rounded[i] < rounded[i - 1]:
            rounded[i] = rounded[i - 1]
    return rounded


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(float(tau)) for tau in levels]
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None or not first.get("viewers"):
        # No hour-1 print: fall back to the last history point, else 1.
        history = case.get("history") or []
        p1 = float(history[-1]["viewers"]) if history and history[-1].get("viewers") else 1.0
    else:
        p1 = float(first["viewers"])
    p1 = max(1.0, p1)
    x0 = math.log(p1)
    launches = usable_launches(case)
    # Per hour, the (log p1, log ratio) pairs across usable launches.
    pairs_by_hour: dict[int, list[tuple[float, float]]] = {}
    for log_p1, hours in launches:
        for h in range(2, 25):
            if h in hours:
                pairs_by_hour.setdefault(h, []).append((log_p1, math.log(hours[h] / hours[1])))
    rows = []
    forecast_rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    for row in forecast_rows:
        h = int(row["hour"])
        shape = local_shape(pairs_by_hour.get(h, []), x0)
        rows.append({"hour": h, "q": hour_quantiles(p1, h, shape, zs)})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
