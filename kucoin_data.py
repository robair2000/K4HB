"""Public KuCoin Futures data (no API keys): contract list, 1-hour candles, funding rate."""
import time

import pandas as pd
import requests

BASE = "https://api-futures.kucoin.com"
H = 3600
PAGE = 200                       # candles per request


def _get(path, params=None, session=requests, tries=4):
    for i in range(tries):
        try:
            r = session.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(5 + 5 * i)
                continue
            r.raise_for_status()
            out = r.json()
            if out.get("code") not in (None, "200000"):
                raise RuntimeError(f"KuCoin error on {path}: {out}")
            return out.get("data")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 2 * i)


def active_contracts(session=requests):
    """USDT-margined perpetuals with their 24h turnover and first-open time (ms)."""
    rows = []
    for c in _get("/api/v1/contracts/active", session=session) or []:
        if c.get("symbol", "").endswith("USDTM") and c.get("isInverse") is False:
            rows.append(dict(symbol=c["symbol"], base=c.get("baseCurrency", ""),
                             turnover=float(c.get("turnoverOf24h") or 0), listed=int(c.get("firstOpenDate") or 0)))
    return rows


def candles_1h(symbol, n, end_ts=None, session=requests, pause=0.12):
    """Last `n` CLOSED 1-hour candles (oldest -> newest). Raises if the newest closed candle is missing
    or recent candles have holes (stale data must never be traded on)."""
    end_ts = int(end_ts or time.time())
    rows, cursor = {}, end_ts * 1000
    for _ in range(n // PAGE + 4):
        if len(rows) >= n + 2:
            break
        data = _get("/api/v1/kline/query", {"symbol": symbol, "granularity": 60,
                                            "from": cursor - PAGE * H * 1000, "to": cursor}, session=session) or []
        if not data:
            break
        for x in data:
            rows[int(x[0])] = x[:6]
        cursor = min(min(int(x[0]) for x in data), cursor - PAGE * H * 1000) - 1
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"no candles for {symbol}")
    df = pd.DataFrame(list(rows.values()), columns=["t_ms", "open", "high", "low", "close", "volume"]).astype(float)
    df["t_ms"] = df.t_ms.astype("int64")
    df = df.drop_duplicates("t_ms").sort_values("t_ms")
    df = df[df.t_ms // 1000 + H <= end_ts].tail(n).reset_index(drop=True)
    want = (end_ts // H) * H - H
    if df.empty or int(df.t_ms.iloc[-1] // 1000) < want:
        raise RuntimeError(f"stale candles for {symbol} (newest {df.t_ms.iloc[-1] if len(df) else 'none'}, want {want * 1000})")
    recent = df.t_ms.values[-min(len(df), 48):] // 1000
    if len(recent) > 1 and ((recent[1:] - recent[:-1]) != H).any():
        raise RuntimeError(f"gaps in recent candles for {symbol}")
    return df


def funding_rate(symbol, session=requests):
    """Current funding rate (per 8 hours) for a contract, or 0.0 if unavailable."""
    try:
        d = _get(f"/api/v1/funding-rate/{symbol}/current", session=session) or {}
        return float(d.get("value") or 0.0)
    except Exception:
        return 0.0
