# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kokorotomo51/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kokorotomo51/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Write a program that reads the Twitch viewer count of Grand Theft Auto VI up to the end of the first hour of its release day, 2026-11-19 (UTC), and forecasts how many people will be watching it live on Twitch in each of the other 23 hours, as five quantiles an hour, for the lowest pinball loss on the log scale.

The forecaster (`main.py`, Python 3.13, standard library only) fits a launch-day shape model at run time from the case's own comparables: past big game launches' hourly Twitch curves. For each hour it locally regresses log(viewers / hour-1 print) on log(hour-1 print) over the past launches nearest in scale to the case, ensembled with the local median, with a calibrated log-normal spread. Leave-one-launch-out backtest on the kit's comparables: pinball loss 0.199 on launches with hour-1 prints ≥ 50,000 (house persistence baseline: 0.251) and 0.248 on launches ≥ 20,000 (baseline: 0.321).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=7bd7933c&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
