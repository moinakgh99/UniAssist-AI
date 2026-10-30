"""Shared test helpers. Tests never touch the real data/chatbot.db or models/intent.joblib: the DB is copied
to a temp folder first, and the two downloadable neural models are replaced by deterministic stand-ins."""
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nlp  # noqa: E402

_RealBot = nlp.Bot  # captured before any test replaces nlp.Bot with make_bot


def temp_db_copy(empty=False) -> Path:
    d = Path(tempfile.mkdtemp(prefix="uniassist_test_"))
    dest = d / "chatbot.db"
    if not empty:
        shutil.copy(ROOT / "data" / "chatbot.db", dest)
    return dest


class HashEmbedder:
    """Deterministic bag-of-words 'sentence embedder' (stand-in for all-MiniLM-L6-v2)."""
    DIM = 512

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        out = np.zeros((len(texts), self.DIM))
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, zlib.crc32(w.encode()) % self.DIM] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class FakeZeroShot:
    """Stand-in for the BART zero-shot pipeline: probability on the best retrieval category grows with similarity."""
    def __init__(self, retriever):
        self.r = retriever

    def __call__(self, text, candidate_labels, multi_label=False):
        s = self.r.scores(text)
        best = self.r.cats[int(s.argmax())]
        top = min(0.9, 0.15 + 1.5 * float(s.max()))   # unsure when nothing in the FAQ resembles the query
        rest = (1 - top) / max(1, len(candidate_labels) - 1)
        scores = [top if c == best else rest for c in candidate_labels]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {"labels": [candidate_labels[i] for i in order], "scores": [scores[i] for i in order]}


class FakeSIA:
    def __init__(self):
        self.lexicon = {}

    def polarity_scores(self, text):
        low = text.lower()
        return {"compound": -0.9 if any(w in low for w in ("useless", "pathetic", "nightmare")) else 0.0}


def make_bot():
    """A real nlp.Bot whose two neural models are stand-ins. respond()/understand()/escalation logic is the
    project's real code."""
    bot = _RealBot.__new__(_RealBot)
    bot.embedder = HashEmbedder()
    bot.retriever = nlp.Retriever(pd.read_csv(nlp.FAQ_PATH), embedder=bot.embedder)
    bot.intent_labels = sorted(set(bot.retriever.cats))
    bot.clf = FakeZeroShot(bot.retriever)
    bot.sia = FakeSIA()
    return bot
