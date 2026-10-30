"""Evaluate the chatbot on hand-written test queries (data/test_queries.csv).

Columns: user_query, intent, expected_subcategory, should_escalate (0/1)
Rows with intent 'out_of_scope' or 'smalltalk' are skipped for intent/retrieval metrics
but still count for escalation metrics.

Outputs: report/metrics.json, report/errors.csv, report/figures/confusion_matrix.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).parent))
from nlp import ROOT, Bot, scrub  # noqa: E402


def main(path=ROOT / "data" / "test_queries.csv"):
    bot = Bot()
    df = pd.read_csv(path).fillna({"expected_subcategory": ""})
    rows, times = [], []
    for _, r in df.iterrows():
        t0 = time.perf_counter()
        res = bot.respond(r["user_query"])
        times.append((time.perf_counter() - t0) * 1000)
        clean = scrub(r["user_query"])
        top3 = [t["subcategory"] for t in bot.rank(clean, k=3)]
        rows.append({**r.to_dict(), "pred_intent": res["intent"], "clf_only_intent": bot.predict_intent(clean)[0],
                     "confidence": round(res["confidence"], 3), "fused": round(res.get("fused", 0.0), 3),
                     "retrieval_score": round(res["retrieval_score"], 3), "sentiment": round(res["sentiment"], 3),
                     "pred_escalate": int(res["escalated"]), "reason": res["reason"], "top3_sub": top3})
    out = pd.DataFrame(rows)

    # intent metrics (in-scope rows, excluding crisis/handoff-only rows that skip the classifier)
    ins = out[~out["intent"].isin(["out_of_scope", "smalltalk"]) & out["expected_subcategory"].ne("")]
    acc = accuracy_score(ins["intent"], ins["pred_intent"])
    macro_f1 = f1_score(ins["intent"], ins["pred_intent"], average="macro")
    clf_only_acc = accuracy_score(ins["intent"], ins["clf_only_intent"])  # ablation: classifier alone

    # retrieval: is the expected subcategory the top-1 / within the top-3 retrieved topics?
    top1 = sum(1 for _, r in ins.iterrows() if r["top3_sub"] and r["top3_sub"][0] == r["expected_subcategory"]) / len(ins)
    top3 = sum(1 for _, r in ins.iterrows() if r["expected_subcategory"] in r["top3_sub"]) / len(ins)

    # escalation metrics
    p = precision_score(out["should_escalate"], out["pred_escalate"], zero_division=0)
    rc = recall_score(out["should_escalate"], out["pred_escalate"], zero_division=0)
    f1 = f1_score(out["should_escalate"], out["pred_escalate"], zero_division=0)
    crisis = out[out["user_query"].str.contains("ending my life|self harm", case=False)]

    metrics = {
        "n_test_queries": len(out), "n_in_scope": len(ins),
        "intent_accuracy_fused": round(acc, 3), "intent_macro_f1_fused": round(macro_f1, 3),
        "intent_accuracy_classifier_only": round(clf_only_acc, 3),
        "retrieval_top1": round(top1, 3), "retrieval_top3": round(top3, 3),
        "escalation_precision": round(p, 3), "escalation_recall": round(rc, 3), "escalation_f1": round(f1, 3),
        "crisis_queries_escalated": f"{int(crisis['pred_escalate'].sum())}/{len(crisis)}",
        "latency_ms_median": round(float(pd.Series(times).median()), 2),
        "latency_ms_p95": round(float(pd.Series(times).quantile(0.95)), 2),
    }
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nPer-intent report:\n", classification_report(ins["intent"], ins["pred_intent"], zero_division=0))

    labels = sorted(set(ins["intent"]) | set(ins["pred_intent"]))
    cm = confusion_matrix(ins["intent"], ins["pred_intent"], labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted intent"); plt.ylabel("True intent"); plt.title("Intent confusion matrix")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(ROOT / "report" / "figures" / "confusion_matrix.png", dpi=150)

    err = out[(out["intent"].isin(bot.intent_labels) & (out["intent"] != out["pred_intent"]))
              | (out["should_escalate"] != out["pred_escalate"])
              | ((out["expected_subcategory"] != "") & out.apply(lambda r: not r["top3_sub"] or r["top3_sub"][0] != r["expected_subcategory"], axis=1))]
    err.to_csv(ROOT / "report" / "errors.csv", index=False)
    print(f"\n{len(err)} rows with at least one error saved to report/errors.csv (use them for error analysis).")


if __name__ == "__main__":
    main()
