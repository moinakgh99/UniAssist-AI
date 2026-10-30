"""Train the intent classifier and save it to models/intent.joblib.

Usage:
    python src/train.py                 # train on the FAQ questions only
    python src/train.py --use-chatlog   # also add rows from data/dataset_clean.csv whose
                                        # intent names match the FAQ categories
    python src/train.py --output /tmp/intent.joblib   # write somewhere else (keeps models/intent.joblib untouched)
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from nlp import FAQ_PATH, MODEL_PATH, ROOT, Retriever, normalise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-chatlog", action="store_true")
    ap.add_argument("--output", type=Path, default=MODEL_PATH,
                    help=f"where to save the model (default: {MODEL_PATH})")
    args = ap.parse_args()

    faq = pd.read_csv(FAQ_PATH)
    faq = faq[faq["language"] == "en"]
    X, y = list(faq["question"]), list(faq["category"])
    # Retriever(faq) is built without an embedder: only its parsed topics/keywords are needed here.
    # add short keyword documents so one-word / slangy queries ("wifi", "cv", "scholarship") are learnt too
    for topic in Retriever(faq).topics:
        for kw in sorted(topic["kw"]):
            X.append(kw)
            y.append(topic["category"])
    print(f"Training rows (questions + keywords): {len(X)} | intents: {sorted(set(y))}")

    if args.use_chatlog:
        path = ROOT / "data" / "dataset_clean.csv"
        log = pd.read_csv(path)
        known = set(y)
        match = log[log["intent"].isin(known)]
        print(f"Chat log rows: {len(log)} | matching FAQ intents: {len(match)}")
        unmatched = sorted(set(log["intent"]) - known)
        if unmatched:
            print("  Intent names in dataset.csv with no FAQ category (rename or map them):", unmatched)
        X += list(match["user_query"])
        y += list(match["intent"])

    pipe = make_pipeline(
        TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5),
    )
    pipe.fit(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.output)
    print(f"Saved {args.output}")
    print("NOTE: do not report accuracy on the training data. Run src/evaluate.py on your own test queries.")


if __name__ == "__main__":
    main()
