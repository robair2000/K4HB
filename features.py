"""Kronos-mini as a pattern reader: its internal summary of the last 400 one-hour candles -> a trained
ridge model -> score for the next 4 hours. Identical to the backtest (checked against the research vectors)."""
import json
import os

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosTokenizer

LOOKBACK = 400
_model = None


def load(model_dir=None):
    """Kronos-mini + its tokenizer (from Hugging Face, or MODEL_DIR with Kronos-mini/ and Kronos-Tokenizer-2k/)."""
    global _model
    if _model is None:
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
        d = model_dir or os.getenv("MODEL_DIR")
        tok_id = f"{d}/Kronos-Tokenizer-2k" if d else "NeoQuasar/Kronos-Tokenizer-2k"
        mdl_id = f"{d}/Kronos-mini" if d else "NeoQuasar/Kronos-mini"
        _model = (KronosTokenizer.from_pretrained(tok_id).eval(), Kronos.from_pretrained(mdl_id).eval())
    return _model


def summaries(candle_frames):
    """candle_frames: list of 1h candle DataFrames (t_ms, open, high, low, close, volume), >= 400 rows each.
    Returns an array [n, 768]: Kronos context at the last candle, mean of the last 8, mean of the last 48."""
    tok, mdl = load()
    xs, ss = [], []
    for df in candle_frames:
        w = df.tail(LOOKBACK)
        x = np.c_[w[["open", "high", "low", "close", "volume"]].values, w.volume.values * w.close.values].astype(np.float32)
        mu, sd = x.mean(0), x.std(0)
        xs.append(np.clip((x - mu) / (sd + 1e-5), -5, 5))
        ts = pd.to_datetime(w.t_ms.values, unit="ms")
        ss.append(np.c_[ts.minute, ts.hour, ts.weekday, ts.day, ts.month].astype(np.float32))
    with torch.no_grad():
        x = torch.tensor(np.array(xs)); s = torch.tensor(np.array(ss))
        t1, t2 = tok.encode(x, half=True)
        _, ctx = mdl.decode_s1(t1, t2, s)
    return torch.cat([ctx[:, -1], ctx[:, -8:].mean(1), ctx[:, -48:].mean(1)], dim=1).numpy()


class Scorer:
    """The trained ridge model on top of the Kronos summaries."""

    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos4h_model.json")
        m = json.load(open(path))
        self.mu, self.sd, self.w = np.array(m["mu"]), np.array(m["sd"]), np.array(m["w"])
        self.thr = float(m["thr"])

    def score(self, F):
        A = (np.asarray(F, dtype=np.float64) - self.mu) / self.sd
        return np.c_[A, np.ones(len(A))] @ self.w
