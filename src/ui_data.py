"""Streamlit-free helpers for the UniAssist AI front end.

Everything here is pure presentation logic (formatting, FAQ browsing, exports). Nothing here talks to the
database or to the NLP models, and nothing imports streamlit, so it can be unit-tested on its own.
"""
from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- categories
# Display names/icons for the FAQ categories that exist in data/faq_dataset.csv.
CATEGORY_META = {
    "admissions": ("Admissions", ":material/how_to_reg:"),
    "financial_aid": ("Fees & Financial Aid", ":material/payments:"),
    "finance": ("Fees & Payments", ":material/receipt_long:"),
    "housing": ("Housing", ":material/apartment:"),
    "academics": ("Academics", ":material/school:"),
    "advising": ("Advising", ":material/forum:"),
    "attendance": ("Attendance", ":material/event_available:"),
    "exams": ("Exams", ":material/edit_note:"),
    "it_services": ("IT Services", ":material/devices:"),
    "library": ("Library", ":material/local_library:"),
    "mental_health": ("Wellbeing", ":material/self_improvement:"),
    "health_services": ("Health Services", ":material/medical_services:"),
    "career_services": ("Careers", ":material/work:"),
    "international_students": ("International Students", ":material/public:"),
    "campus_life": ("Campus Life", ":material/groups:"),
}


def category_label(category: str) -> str:
    return CATEGORY_META.get(category, (category.replace("_", " ").title(), ""))[0]


# --------------------------------------------------------------------------- prompt library
# Every prompt is a real question from data/faq_dataset.csv (checked by tests/test_ui_data.py).
PROMPT_LIBRARY = {
    "Admissions": [
        "How do I apply for undergraduate admission?",
        "How do I check my application status?",
        "How do I request a deferral of my admission?",
    ],
    "Fees & Aid": [
        "How do I apply for a scholarship?",
        "How do I set up a tuition payment plan?",
        "How do I check my tuition balance?",
    ],
    "Housing": [
        "How do I apply for on-campus housing?",
        "How do I request a roommate change?",
        "How do I report a maintenance problem in my room?",
    ],
    "Academics": [
        "How do I register for courses?",
        "How do I appeal a grade?",
        "How do I request an official transcript?",
    ],
    "IT & Library": [
        "How do I reset my university account password?",
        "How do I connect to the campus Wi-Fi?",
        "How do I book a study room?",
    ],
    "Wellbeing": [
        "How do I book a counseling appointment?",
        "How do I find resources for managing exam stress?",
        "How do I book a medical appointment on campus?",
    ],
    "Careers": [
        "How do I get my resume reviewed?",
        "How do I find internships?",
        "How do I book a mock interview?",
    ],
}
HUMAN_PROMPT = "I'd like to talk to a human agent."  # handled by the bot's existing "user_requested" rule


def all_library_prompts() -> list[str]:
    return [p for prompts in PROMPT_LIBRARY.values() for p in prompts]


# --------------------------------------------------------------------------- FAQ library
def load_faq_library(path) -> pd.DataFrame:
    """English, general-audience FAQ entries (one row per distinct question), sorted for browsing."""
    df = pd.read_csv(path)
    df = df[df["language"] == "en"]
    if "audience" in df.columns:
        df = df[df["audience"] == "general"]
    cols = [c for c in ("id", "question", "answer", "category", "subcategory", "difficulty_level") if c in df.columns]
    df = df[cols].drop_duplicates(subset="question").reset_index(drop=True)
    return df.sort_values(["category", "subcategory", "id"]).reset_index(drop=True)


def faq_category_counts(df: pd.DataFrame) -> dict[str, int]:
    return df.groupby("category").size().sort_values(ascending=False).to_dict()


def filter_faq(df: pd.DataFrame, category: str | None = None, query: str = "") -> pd.DataFrame:
    """Filter by category (None / 'all' = every category) and a case-insensitive text search."""
    out = df
    if category and category != "all":
        out = out[out["category"] == category]
    q = (query or "").strip().lower()
    if q:
        mask = (out["question"].str.lower().str.contains(q, regex=False)
                | out["answer"].str.lower().str.contains(q, regex=False)
                | out["subcategory"].astype(str).str.lower().str.contains(q, regex=False))
        out = out[mask]
    return out


def suggest_questions(df: pd.DataFrame, category: str | None, asked: list[str] | None = None, n: int = 3) -> list[str]:
    """Follow-up suggestions from the same FAQ category, skipping questions already asked."""
    asked_norm = {a.strip().lower() for a in (asked or [])}
    pool = df[df["category"] == category] if category else df.iloc[0:0]
    pool = pool[pool["question"].str.startswith("How")]  # short, self-contained questions read best as chips
    out = [q for q in pool["question"] if q.strip().lower() not in asked_norm]
    if len(out) < n:  # category unknown / exhausted -> fall back to the curated library
        out += [q for q in all_library_prompts() if q.strip().lower() not in asked_norm and q not in out]
    # deterministic spread through the category rather than always the first few
    step = max(1, len(out) // (n * 2)) if len(out) > n else 1
    return out[::step][:n]


# --------------------------------------------------------------------------- chat formatting
def fmt_time(ts) -> str:
    return ts.strftime("%H:%M") if isinstance(ts, datetime.datetime) else ""


def escape_text(text: str) -> str:
    """HTML-escape user/bot text for embedding in custom markup (newlines become <br>)."""
    return html.escape(str(text), quote=True).replace("\r\n", "\n").replace("\n", "<br>")


def one_line(markup: str) -> str:
    """Collapse indented multi-line HTML into one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines())


def bubble_html(msg: dict) -> str:
    """Markup for one chat bubble. All dynamic text is escaped."""
    role = "user" if msg.get("role") == "user" else "assistant"
    who = "You" if role == "user" else "UniAssist"
    badge = ""
    if msg.get("escalated"):
        reason = escape_text(msg.get("reason_text") or "")
        badge = f'<div class="ua-badge ua-badge-warn">Handed to a human agent{(": " + reason) if reason else ""}</div>'
    rated = ""
    if msg.get("rated"):
        rated = '<span class="ua-meta-chip">Feedback sent</span>'
    stamp = fmt_time(msg.get("timestamp"))
    return one_line(f"""
        <div class="ua-msg ua-msg-{role}">
          <div class="ua-avatar ua-avatar-{role}">{'You' if role == 'user' else 'UA'}</div>
          <div class="ua-bubble-wrap">
            <div class="ua-msg-meta"><span>{who}</span><span>{stamp}</span>{rated}</div>
            <div class="ua-bubble ua-bubble-{role}">{badge}{escape_text(msg.get('content', ''))}</div>
          </div>
        </div>""")


TYPING_HTML = one_line("""
    <div class="ua-msg ua-msg-assistant">
      <div class="ua-avatar ua-avatar-assistant">UA</div>
      <div class="ua-bubble-wrap">
        <div class="ua-bubble ua-bubble-assistant ua-typing" role="status" aria-label="UniAssist is thinking">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>""")


# --------------------------------------------------------------------------- local export
def _export_rows(messages: list[dict]) -> list[dict]:
    rows = []
    for m in messages:
        ts = m.get("timestamp")
        rows.append({
            "role": m.get("role", ""),
            "time": ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime.datetime) else "",
            "text": m.get("content", ""),
            "escalated": bool(m.get("escalated", False)),
        })
    return rows


def export_text(messages: list[dict]) -> str:
    lines = ["UniAssist AI chat transcript", "=" * 28, ""]
    for r in _export_rows(messages):
        who = "You" if r["role"] == "user" else "UniAssist"
        flag = "  [handed to a human agent]" if r["escalated"] else ""
        lines += [f"[{r['time']}] {who}{flag}", r["text"], ""]
    return "\n".join(lines)


def export_markdown(messages: list[dict]) -> str:
    out = ["# UniAssist AI chat transcript", ""]
    for r in _export_rows(messages):
        who = "**You**" if r["role"] == "user" else "**UniAssist**"
        flag = " _(handed to a human agent)_" if r["escalated"] else ""
        out += [f"{who} · {r['time']}{flag}", "", r["text"], ""]
    return "\n".join(out)


def export_json(messages: list[dict]) -> str:
    return json.dumps({"app": "UniAssist AI", "messages": _export_rows(messages)}, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- dashboard / agent helpers
def session_stats(messages: list[dict]) -> dict:
    asked = sum(1 for m in messages if m.get("role") == "user")
    answered = [m for m in messages if m.get("role") == "assistant"]
    return {
        "asked": asked,
        "answered": len(answered),
        "escalated": sum(1 for m in answered if m.get("escalated")),
        "rated": sum(1 for m in answered if m.get("rated")),
    }


def log_insights(log_df: pd.DataFrame) -> dict:
    """Aggregate-only numbers from db.export_chat_log() (no message text is surfaced)."""
    if log_df is None or len(log_df) == 0:
        return {"total": 0, "escalation_rate": None, "avg_rating": None, "n_rated": 0, "top_intents": []}
    ratings = pd.to_numeric(log_df["rating"], errors="coerce").dropna()
    intents = (log_df["intent"].dropna().value_counts().head(5))
    return {
        "total": int(len(log_df)),
        "escalation_rate": float((log_df["escalated"] == "Y").mean()),
        "avg_rating": float(ratings.mean()) if len(ratings) else None,
        "n_rated": int(len(ratings)),
        "top_intents": [(str(k), int(v)) for k, v in intents.items()],
    }


def escalation_summary(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"total": 0, "open": 0, "by_reason": {}}
    return {
        "total": int(len(df)),
        "open": int((df["status"] == "open").sum()),
        "by_reason": {str(k): int(v) for k, v in df["reason"].fillna("unknown").value_counts().items()},
    }
