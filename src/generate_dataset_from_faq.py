#!/usr/bin/env python
"""Generate a synthetic chat log dataset from the FAQ dataset.
Creates `data/dataset.csv` with columns expected by clean_and_eda.py.
"""
import uuid
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = ROOT / "data" / "faq_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "dataset.csv"

def main():
    faq = pd.read_csv(FAQ_PATH)
    df = pd.DataFrame({
        "session_id": [uuid.uuid4().hex[:12] for _ in range(len(faq))],
        "user_query": faq["question"],
        "intent": faq["category"],
        "confidence_score": 1.0,
        "bot_response": faq["answer"],
        "escalated": "N",
        "rating": pd.NA,
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {OUTPUT_PATH} with {len(df)} rows")

if __name__ == "__main__":
    main()
