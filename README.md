# Kronos-4h paper bot (KuCoin Futures)

Paper trading only: public KuCoin data, **no API keys, no real orders**.

## What it does
Every 4 hours (at the 00/04/08/12/16/20 UTC candle close), Kronos-mini reads the last 400 one-hour
candles of each coin. Instead of using Kronos's own price forecast (which tested no better than a coin
flip), it uses Kronos's **internal summary** of those candles. A small model trained on top of that
summary gives a score for the next 4 hours. Only the strongest ~10% of scores are traded: long if
positive, short if negative, held exactly 4 hours.

Two paper accounts run on the same signals, so you can see what the order type costs:
- **limit:** post-only entry at the signal price, filled only if the next candle trades through it
  (maker fee); market exit after 4h.
- **market:** market orders both ways (taker fee + slippage).

## Backtest it came from
- **Development data:** the strongest 10% of calls beat the average coin by +0.26% per 4h trade,
  right 56% of the time, on longs and shorts alike.
- **Locked newest 30 days:** +0.8% per trade, right 54–55%. But most of that came from IRYS's ~8× pump
  at the end. **Without IRYS: +0.13% per trade**, which pays with limit orders but not with market
  orders. This is the weaker of the two new bots; paper trading decides.
- `kronos4h_model.json` is the trained model (on all 150 days). The bot's Kronos summaries were checked
  to be identical to the research ones (difference 0.0).

## Reading the results

Two numbers matter, and the equity curve is not one of them.

**Net exposure.** The book is not dollar-neutral: it can be 40% net long or 10% net short depending on
which way the scores point. `kronos4h_exposure.csv` records net and gross notional every hour, and the
log line shows net as a percentage of equity. A book that is quietly net long in a rising market will
look skilful when it is only exposed.

**Excess vs the basket.** Every closed trade records `excess_after_fees` in `kronos4h_trades.csv`: the
trade's return after removing the equal-weight move of the same coins over the same window
(`ret - basket` for a long, `ret + basket` for a short). This is the only number that says whether the
model picks coins better than chance.

`python bot.py --report` prints the running excess, its t-statistic, and how many trades are still
needed before the result means anything.

**Kill criterion, fixed in advance:** at ~600 closed trades, if mean excess per trade is below zero
(or t is below 2), the model has no cross-sectional skill and the bot is dead - regardless of how the
equity curve looks. At ~8 trades a day that is about three months.

**Caveat on the backtest above:** `kronos4h_model.json` was trained on 2026-05-27..2026-10-23, which
is the same window the "locked newest 30 days" result covers. That number is therefore in-sample, and
the paper run is the first genuinely out-of-sample test.

`python test_excess.py` checks the accounting on synthetic data with no network: a book of coins that
all move together must show zero excess, and a single coin that moves alone must show almost all of it.

## Render
Background worker. It runs Kronos-mini like KB does, so the Starter plan (512 MB) should fit; keep
`TORCH_THREADS=1`. Add a disk at `/data` and set `DATA_DIR=/data`. `render.yaml` has a blueprint.
The `model/` folder is the Kronos model code (same as in KB); the weights download from Hugging Face.

## Settings (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `paper` | only paper is supported |
| `PAIRS` | your 12 coins | the model was trained on these |
| `HOLD_HOURS` | `4` | holding time (the model predicts 4h) |
| `DECISION_HOUR_OFFSET` | `0` | decide at UTC hours where hour % 4 == this |
| `THRESHOLD_MULT` | `1.0` | >1 = fewer, stronger signals |
| `POSITION_FRAC` | `0.10` | size per trade as a fraction of equity |
| `MAX_POSITIONS` | `12` | |
| `TAKER_FEE` / `MAKER_FEE` / `SLIPPAGE` | `0.0006` / `0.0002` / `0.0005` | |
| `FILL_BUFFER` | `0.0002` | limit counts as filled only if price trades this far through it |
| `START_CASH` | `1000` | paper balance per account |
| `TORCH_THREADS` | `1` | CPU threads for Kronos |
| `DATA_DIR` | `.` | state, trades, signals and equity CSVs |

Run `python bot.py --report` for results. Kill switch: create a file named `STOP`.
