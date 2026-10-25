"""Plumbing test for the excess-vs-basket accounting. No Kronos, no network.

Scenario A: every coin rises exactly 1% per hour. The equal-weight basket rises 4.06% over 4h.
            A long therefore earns ~+4% but has ~0 excess - it was the market, not the pick,
            so all that is left is the round-trip cost.
Scenario B: only the traded coin rises; the other 11 are flat. The basket barely moves, so the
            same long should show almost all of its return as excess.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

DATA = os.path.join(tempfile.gettempdir(), "k4hb_test")
shutil.rmtree(DATA, ignore_errors=True)
os.makedirs(DATA, exist_ok=True)
os.environ["DATA_DIR"] = DATA
os.environ["START_CASH"] = "1000"

import bot  # noqa: E402

BASE = 1767225600          # 2026-01-01 00:00 UTC, a multiple of 4h
H = 3600
PAIRS = bot.PAIRS
START = BASE // H - 500    # array index 0 is 500 hours before BASE
N = 1200


class Clock:
    now = BASE


def make_prices(growth):
    """{sym: closes indexed by (absolute_hour - START)}; growth[sym] is the per-hour multiplier."""
    return {bot.contract(s): 100.0 * growth.get(s, 1.0) ** np.arange(N) for s in PAIRS}


class FakeScorer:
    thr = 0.1
    next_scores = None

    def score(self, F):
        return np.array(self.next_scores, dtype=float)


def make_fetch(px):
    def fetch(sym, n):                       # the bot calls fetch(sym, n) - no end_ts
        cur = Clock.now // H                 # the newest candle CLOSES at this hour
        idx = np.arange(cur - 450, cur)      # so it opened an hour earlier
        c = px[sym][idx + 1 - START]         # close price = price at the close time
        return pd.DataFrame({"t_ms": (idx * H * 1000).astype("int64"), "open": c,
                             "high": c * 1.001, "low": c * 0.999, "close": c,
                             "volume": np.ones(len(c))})
    return fetch


def run(growth, score_sym, label, break_bench=False):
    if os.path.exists(bot.STATE_PATH):
        os.remove(bot.STATE_PATH)
    st = bot.load_state()
    sc = FakeScorer()
    fetch = make_fetch(make_prices(growth))
    summarize = lambda frames: np.zeros((len(frames), 768))
    target = bot.contract(score_sym)

    for step, scores in ((0, [0.0] * len(PAIRS)),
                         (4, [0.5 if bot.contract(p) == target else 0.0 for p in PAIRS]),
                         (8, [0.0] * len(PAIRS))):
        Clock.now = BASE + step * H
        sc.next_scores = scores
        if break_bench and step == 8:
            # simulate a position that opened before the benchmark history existed (e.g. pre-restart)
            st["bench_hist"].pop(str(BASE + 4 * H), None)
        bot.cycle(st, sc, now=Clock.now, fetch=fetch, summarize=summarize)

    a = st["accounts"]["market"]
    n, m = a["trades"], a.get("excess_n", 0)
    ex = a.get("excess", 0.0)
    print(f"\n{label}\n  trades {n} | measured {m} | excess total {ex:+.4%} | "
          f"per measured trade {ex / m if m else float('nan'):+.4%}")
    return a, st


a1, st1 = run({p: 1.01 for p in PAIRS}, "ETH", "A. all 12 coins +1%/h (no dispersion)")
a2, st2 = run({**{p: 1.0 for p in PAIRS}, "ETH": 1.01}, "ETH",
              "B. only ETH +1%/h, the other 11 flat")
a3, st3 = run({**{p: 1.0 for p in PAIRS}, "ETH": 1.01}, "ETH",
              "C. same as B but the entry-hour benchmark is missing (pre-restart position)",
              break_bench=True)

print("\n--- report() ---")
bot.report()

print("\n--- trades csv ---")
print(open(os.path.join(DATA, "kronos4h_trades.csv")).read().strip())

ok = True
e1 = a1.get("excess", 0.0) / a1["trades"] if a1["trades"] else 9
e2 = a2.get("excess", 0.0) / a2["trades"] if a2["trades"] else 0
if not (-0.005 < e1 < 0.001):
    print(f"\nFAIL A: per-trade excess should be ~0 minus costs, got {e1:+.4%}")
    ok = False
if not (e2 > 0.03):
    print(f"\nFAIL B: per-trade excess should be most of the move, got {e2:+.4%}")
    ok = False
if a1["trades"] == 0 or a2["trades"] == 0:
    print("\nFAIL: no trades were taken, the harness is not exercising anything")
    ok = False
if a3["trades"] == 0 or a3.get("excess_n", 0) != 0:
    print(f"\nFAIL C: a trade with no benchmark must be excluded from excess "
          f"(trades {a3['trades']}, measured {a3.get('excess_n', 0)})")
    ok = False
print("\nPASS" if ok else "\nFAILED")
