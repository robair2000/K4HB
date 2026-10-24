"""Kronos-4h paper bot for KuCoin Futures (public data only, no API keys, no real orders).

Every 4 hours, at a candle close, Kronos-mini reads the last 400 one-hour candles of each coin. A small
model trained on top of Kronos's internal summary turns that into a score for the next 4 hours. Only the
strongest scores (top ~10%, threshold from training) are traded: long if positive, short if negative,
held for exactly 4 hours, no stop / take-profit (as backtested).

Two paper accounts run side by side on the same signals:
  limit  : entry = post-only limit at the signal candle's close, only filled if the next candle trades
           through it (maker fee); exit = market order after 4h (taker fee + slippage)
  market : entry and exit with market orders (taker fee + slippage both ways)

    python bot.py            # runs forever (wakes just after every hour)
    python bot.py --once     # one cycle, then exits
    python bot.py --report   # paper results so far
    touch STOP               # kill switch
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

import kucoin_data as kd


def _f(name, default):
    return float(os.getenv(name, default))


MODE = os.getenv("MODE", "paper").lower()
PAIRS = [p.strip().upper() for p in os.getenv(
    "PAIRS", "ETH,SOL,ZEC,IRYS,XRP,NEAR,XMR,USELESS,ASTER,LINK,ADA,DOGE").split(",") if p.strip()]
HOLD_H = int(os.getenv("HOLD_HOURS", "4"))
DECISION_HOUR_OFFSET = int(os.getenv("DECISION_HOUR_OFFSET", "0"))   # decide when (UTC close hour % 4) == this
THRESHOLD_MULT = _f("THRESHOLD_MULT", "1.0")         # >1 = fewer, stronger signals
POSITION_FRAC = _f("POSITION_FRAC", "0.10")          # notional per trade / equity
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "12"))
TAKER_FEE = _f("TAKER_FEE", "0.0006")
MAKER_FEE = _f("MAKER_FEE", "0.0002")
SLIPPAGE = _f("SLIPPAGE", "0.0005")
FILL_BUFFER = _f("FILL_BUFFER", "0.0002")            # limit counts as filled only if price trades this far through
START_CASH = _f("START_CASH", "1000")
DATA_DIR = os.getenv("DATA_DIR", ".")
STATE_PATH = os.path.join(DATA_DIR, os.getenv("STATE_FILE", "kronos4h_state.json"))
TRADES_CSV = os.path.join(DATA_DIR, "kronos4h_trades.csv")
SIGNALS_CSV = os.path.join(DATA_DIR, "kronos4h_signals.csv")
EQUITY_CSV = os.path.join(DATA_DIR, "kronos4h_equity.csv")
H = 3600


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def contract(p):
    return p if p.endswith("USDTM") else ("XBT" if p == "BTC" else p) + "USDTM"


def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    acct = lambda: {"cash": START_CASH, "pos": {}, "trades": 0, "wins": 0, "fees": 0.0}
    return {"accounts": {"limit": acct(), "market": acct()}, "pending": {}, "last_decision": None,
            "missed": 0, "signals": 0, "started": time.time()}


def save_state(st):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_PATH)


def acct_equity(a, prices):
    return a["cash"] + sum(p["dir"] * p["qty"] * (prices.get(s, p["entry"]) - p["entry"]) for s, p in a["pos"].items())


def close_pos(st, name, sym, px, why, now_iso):
    a = st["accounts"][name]
    p = a["pos"].pop(sym)
    fill = px * (1 - p["dir"] * SLIPPAGE)
    fee = p["qty"] * fill * TAKER_FEE
    pnl = p["dir"] * p["qty"] * (fill - p["entry"]) - fee
    a["cash"] += pnl
    a["fees"] += fee
    a["trades"] += 1
    ret = p["dir"] * (fill / p["entry"] - 1) - p["fee_in"] / (p["qty"] * p["entry"]) - TAKER_FEE
    a["wins"] += ret > 0
    append_csv(TRADES_CSV, ["closed", "account", "symbol", "side", "entry", "exit", "return_after_fees", "reason", "score"],
               [now_iso, name, sym, "LONG" if p["dir"] == 1 else "SHORT", p["entry"], round(fill, 10), round(ret, 5), why, round(p["score"], 4)])
    return ret


def cycle(st, scorer, now=None, fetch=None, summarize=None):
    """One hourly cycle. `fetch(sym, n)` and `summarize(frames)` can be swapped for tests."""
    now = now or time.time()
    fetch = fetch or (lambda s, n: kd.candles_1h(s, n, end_ts=now))
    if summarize is None:
        from features import summaries as summarize
    now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
    close_hour = int(now // H) * H                       # close time of the newest closed 1h candle
    decide = st["last_decision"] != close_hour and (close_hour // H) % 4 == DECISION_HOUR_OFFSET % 4
    syms = [contract(p) for p in PAIRS]
    need = set(st["pending"]) | set(st["accounts"]["limit"]["pos"]) | set(st["accounts"]["market"]["pos"])
    frames, prices = {}, {}
    for s in (syms if decide else sorted(need)):
        try:
            frames[s] = fetch(s, 450 if decide else 8)
            prices[s] = float(frames[s].close.iloc[-1])
        except Exception as e:
            log(f"{s}: data error, skipped this cycle ({e})")

    # 1) pending limit entries: filled if the candle after the signal traded through the limit price
    for s, o in list(st["pending"].items()):
        df = frames.get(s)
        if df is None:
            continue
        after = df[df.t_ms // 1000 + H > o["placed"]]      # candles that closed after the order was placed
        if after.empty:
            continue
        c = after.iloc[0]
        hit = (o["dir"] == 1 and c.low <= o["limit"] * (1 - FILL_BUFFER)) or \
              (o["dir"] == -1 and c.high >= o["limit"] * (1 + FILL_BUFFER))
        st["pending"].pop(s)
        if hit:
            a = st["accounts"]["limit"]
            fee = o["qty"] * o["limit"] * MAKER_FEE
            a["cash"] -= fee
            a["fees"] += fee
            a["pos"][s] = dict(dir=o["dir"], qty=o["qty"], entry=o["limit"], fee_in=fee, exit_at=o["exit_at"], score=o["score"])
        else:
            st["missed"] += 1
            log(f"{s}: limit {'buy' if o['dir'] == 1 else 'sell'} at {o['limit']:.6g} not filled - missed")

    # 2) exits after HOLD_H hours (market order at that candle's close)
    for name in ("limit", "market"):
        for s, p in list(st["accounts"][name]["pos"].items()):
            if close_hour >= p["exit_at"] and s in prices:
                df = frames[s]
                row = df[df.t_ms // 1000 + H == p["exit_at"]]
                px = float(row.close.iloc[0]) if len(row) else prices[s]
                r = close_pos(st, name, s, px, f"{HOLD_H}h exit", now_iso)
                log(f"[{name}] {s}: closed {'LONG' if p['dir'] == 1 else 'SHORT'} after {HOLD_H}h: {r:+.2%} after fees")

    # 3) new signals
    if decide:
        st["last_decision"] = close_hour
        ok = [s for s in syms if s in frames and len(frames[s]) >= 400]
        if ok:
            scores = scorer.score(summarize([frames[s] for s in ok]))
            thr = scorer.thr * THRESHOLD_MULT
            picked = []
            for s, sc in sorted(zip(ok, scores), key=lambda x: -abs(x[1])):
                append_csv(SIGNALS_CSV, ["time", "symbol", "price", "score", "threshold", "traded"],
                           [now_iso, s, prices[s], round(float(sc), 5), round(thr, 5), abs(sc) >= thr])
                if abs(sc) < thr:
                    continue
                d = 1 if sc > 0 else -1
                for name in ("limit", "market"):
                    a = st["accounts"][name]
                    busy = s in a["pos"] or (name == "limit" and s in st["pending"])
                    n_open = len(a["pos"]) + (len(st["pending"]) if name == "limit" else 0)
                    if busy or n_open >= MAX_POSITIONS:
                        continue
                    eq = acct_equity(a, prices)
                    qty = eq * POSITION_FRAC / prices[s]
                    if name == "limit":
                        st["pending"][s] = dict(dir=d, qty=qty, limit=prices[s], placed=close_hour,
                                                exit_at=close_hour + HOLD_H * H, score=float(sc))
                    else:
                        fill = prices[s] * (1 + d * SLIPPAGE)
                        fee = qty * fill * TAKER_FEE
                        a["cash"] -= fee
                        a["fees"] += fee
                        a["pos"][s] = dict(dir=d, qty=qty, entry=fill, fee_in=fee, exit_at=close_hour + HOLD_H * H, score=float(sc))
                picked.append(f"{s.replace('USDTM', '')} {'LONG' if d == 1 else 'SHORT'} ({sc:+.2f})")
                st["signals"] += 1
            log(f"decision: {len(ok)} coins scored, threshold {thr:.3f} | "
                + (", ".join(picked) if picked else "no signal strong enough"))

    for name in ("limit", "market"):
        a = st["accounts"][name]
        eq = acct_equity(a, prices)
        wr = a["wins"] / a["trades"] if a["trades"] else float("nan")
        log(f"[{name}] equity {eq:.2f} ({eq / START_CASH - 1:+.2%}) | trades {a['trades']} | win rate {wr:.0%} | open {len(a['pos'])} | fees {a['fees']:.2f}")
        append_csv(EQUITY_CSV, ["time", "account", "equity", "trades", "wins", "open", "fees"],
                   [now_iso, name, round(eq, 4), a["trades"], a["wins"], len(a["pos"]), round(a["fees"], 4)])
    save_state(st)


def report():
    if not os.path.exists(STATE_PATH):
        print("no results yet")
        return
    st = load_state()
    for name, a in st["accounts"].items():
        wr = a["wins"] / a["trades"] if a["trades"] else float("nan")
        print(f"{name:6s}: cash {a['cash']:.2f} ({a['cash'] / START_CASH - 1:+.2%} closed P&L) | trades {a['trades']} | win rate {wr:.0%} | fees {a['fees']:.2f}")
    print(f"signals {st['signals']} | limit entries missed {st['missed']}")


def main():
    if "--report" in sys.argv:
        return report()
    if MODE != "paper":
        raise SystemExit("This bot is paper-only for now (MODE=paper). It never places real orders.")
    from features import Scorer, load
    scorer = Scorer()
    log(f"KRONOS-4H PAPER BOT | {len(PAIRS)} coins | decide every 4h | hold {HOLD_H}h | threshold x{THRESHOLD_MULT:g} "
        f"({scorer.thr * THRESHOLD_MULT:.3f}) | {POSITION_FRAC:.0%} of equity per trade | max {MAX_POSITIONS} open")
    load()
    st = load_state()
    while True:
        if os.path.exists("STOP"):
            log("STOP file found, exiting")
            break
        try:
            cycle(st, scorer)
        except Exception as e:
            log(f"cycle error: {e}")
        if "--once" in sys.argv:
            break
        time.sleep(H - time.time() % H + 90)


if __name__ == "__main__":
    main()
