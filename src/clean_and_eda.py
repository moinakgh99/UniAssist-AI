"""Clean data/dataset.csv, make EDA figures, and write data_dictionary.csv.

Usage:
    python src/clean_and_eda.py                       # reads data/dataset.csv
    python src/clean_and_eda.py --input path/to/file.csv

Expected columns (from the project brief):
    session_id, user_query, intent, confidence_score, bot_response, escalated (Y/N), rating
Outputs:
    data/dataset_clean.csv, report/data_dictionary.csv, report/figures/eda_*.png, report/cleaning_log.txt
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from nlp import ROOT, scrub  # noqa: E402

EXPECTED = ["session_id", "user_query", "intent", "confidence_score", "bot_response", "escalated", "rating"]
FIG = ROOT / "report" / "figures"


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\W+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df


def clean(df, log):
    n0 = len(df)
    missing_cols = [c for c in EXPECTED if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing expected columns: {missing_cols}. Found: {list(df.columns)}")
    log.append(f"Rows loaded: {n0}")
    log.append("Missing values per column (raw):\n" + df[EXPECTED].isna().sum().to_string())

    for c in ["session_id", "user_query", "intent", "bot_response", "escalated"]:
        df[c] = df[c].astype("string").str.strip()
    df["intent"] = df["intent"].str.lower().str.replace(r"[\s-]+", "_", regex=True)

    before = len(df)
    df = df[df["user_query"].notna() & (df["user_query"] != "")]
    log.append(f"Dropped {before - len(df)} rows with empty user_query")

    before = len(df)
    df = df.drop_duplicates()
    log.append(f"Dropped {before - len(df)} exact duplicate rows")

    yes = {"y", "yes", "true", "1", "t"}
    no = {"n", "no", "false", "0", "f"}
    esc = df["escalated"].str.lower()
    df["escalated"] = esc.map(lambda v: "Y" if v in yes else "N" if v in no else pd.NA)
    log.append(f"Rows with unrecognised escalated value (set to missing): {int(df['escalated'].isna().sum())}")

    conf = pd.to_numeric(df["confidence_score"], errors="coerce")
    scaled = int((conf > 1).sum())
    conf = conf.where(conf <= 1, conf / 100)          # 0-100 scale -> 0-1
    df["confidence_score"] = conf.clip(0, 1)
    log.append(f"confidence_score: rescaled {scaled} values from 0-100 to 0-1; {int(conf.isna().sum())} non-numeric")

    rating = pd.to_numeric(df["rating"], errors="coerce")
    bad = int(((rating < 1) | (rating > 5)).sum())
    df["rating"] = rating.where((rating >= 1) & (rating <= 5)).astype("Int64")
    log.append(f"rating: {bad} values outside 1-5 set to missing; {int(df['rating'].isna().sum())} missing (unrated)")

    for c in ["user_query", "bot_response"]:
        masked = df[c].map(lambda t: scrub(t) if isinstance(t, str) else t)
        log.append(f"PII masked in {c}: {int((masked != df[c]).sum())} rows")
        df[c] = masked

    df["query_length"] = df["user_query"].str.len()
    df["word_count"] = df["user_query"].str.split().str.len()
    log.append(f"Final rows: {len(df)}")
    return df.reset_index(drop=True)


def make_figures(df):
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    # 1. intent distribution
    plt.figure(figsize=(8, 5))
    order = df["intent"].value_counts().index
    sns.countplot(y="intent", data=df, order=order, color="#4C72B0")
    plt.title("Number of queries per intent"); plt.tight_layout()
    plt.savefig(FIG / "eda_1_intent_counts.png", dpi=150); plt.close()
    # 2. confidence distribution by escalation
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df.dropna(subset=["escalated"]), x="confidence_score", hue="escalated", bins=20, multiple="layer")
    plt.title("Confidence score distribution (escalated Y/N)"); plt.tight_layout()
    plt.savefig(FIG / "eda_2_confidence_hist.png", dpi=150); plt.close()
    # 3. escalation rate per intent
    rate = (df.dropna(subset=["escalated"]).assign(esc=lambda d: (d["escalated"] == "Y").astype(int))
              .groupby("intent")["esc"].mean().sort_values())
    plt.figure(figsize=(8, 5)); rate.plot.barh(color="#DD8452")
    plt.xlabel("Escalation rate"); plt.title("Escalation rate by intent"); plt.tight_layout()
    plt.savefig(FIG / "eda_3_escalation_by_intent.png", dpi=150); plt.close()
    # 4. rating distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="rating", data=df.dropna(subset=["rating"]), color="#55A868")
    plt.title("User rating distribution"); plt.tight_layout()
    plt.savefig(FIG / "eda_4_rating_dist.png", dpi=150); plt.close()
    # 5. rating by escalation
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="escalated", y="rating", data=df.dropna(subset=["escalated", "rating"]).astype({"rating": float}))
    plt.title("Rating vs escalation"); plt.tight_layout()
    plt.savefig(FIG / "eda_5_rating_by_escalation.png", dpi=150); plt.close()


DESCRIPTIONS = {
    "session_id": ("string", "Identifier of the chat session (several turns can share one session).", "any unique text"),
    "user_query": ("string", "Student's message, PII masked.", "free text"),
    "intent": ("string (categorical)", "Intent label assigned to the query.", "lower_snake_case category names"),
    "confidence_score": ("float", "Model confidence for the predicted intent.", "0.0 - 1.0"),
    "bot_response": ("string", "Text the bot replied with.", "free text"),
    "escalated": ("string (categorical)", "Whether the chat was handed to a human agent.", "Y / N"),
    "rating": ("integer (nullable)", "Student satisfaction rating; empty if not rated.", "1 - 5"),
    "query_length": ("integer", "Derived: number of characters in user_query.", ">= 1"),
    "word_count": ("integer", "Derived: number of words in user_query.", ">= 1"),
}


def write_dictionary(df, out):
    rows = []
    for col, (dtype, desc, allowed) in DESCRIPTIONS.items():
        example = df[col].dropna().iloc[0] if col in df and df[col].notna().any() else ""
        rows.append({"column": col, "data_type": dtype, "description": desc, "allowed_values_or_range": allowed,
                     "example": str(example)[:80], "missing_count": int(df[col].isna().sum()) if col in df else ""})
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "dataset.csv"))
    args = ap.parse_args()
    log = []
    df = clean(load_raw(args.input), log)
    (ROOT / "report").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "dataset_clean.csv", index=False)
    write_dictionary(df, ROOT / "report" / "data_dictionary.csv")
    make_figures(df)
    (ROOT / "report" / "cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print("\nSaved: data/dataset_clean.csv, report/data_dictionary.csv, report/cleaning_log.txt, report/figures/eda_*.png")


if __name__ == "__main__":
    main()
