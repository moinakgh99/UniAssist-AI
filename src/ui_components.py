"""Reusable Streamlit components for UniAssist AI.

Presentation only: these functions draw widgets and *report* what the person did (by returning a value);
they never call the database or the NLP bot. src/app.py performs the actual backend operations.
"""
from __future__ import annotations

import datetime
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from ui_data import (HUMAN_PROMPT, PROMPT_LIBRARY, TYPING_HTML, bubble_html, category_label, escape_text,
                     export_json, export_markdown, export_text, one_line)

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "styles.css"

PAGES = [  # (state key, label, material icon)
    ("dashboard", "Dashboard", ":material/dashboard:"),
    ("assistant", "AI Assistant", ":material/chat:"),
    ("faq", "FAQ Library", ":material/menu_book:"),
    ("agent", "Agent Support", ":material/support_agent:"),
    ("privacy", "Privacy & Settings", ":material/shield:"),
    ("feedback", "Feedback", ":material/thumb_up:"),
]
PAGE_LABELS = {key: label for key, label, _ in PAGES}

AGENT_REASON_LABELS = {
    "crisis": "Wellbeing concern",
    "user_requested": "Asked for a human",
    "negative_sentiment": "Frustrated student",
    "low_confidence": "Low confidence",
}


# ------------------------------------------------------------------ theming & navigation
@lru_cache(maxsize=1)
def _css_text() -> str:
    try:
        return CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_css(theme: str = "light", density: str = "comfortable") -> None:
    """Inject the design system and the marker element that selects the theme/density variables."""
    theme = theme if theme in ("light", "dark") else "light"
    density = density if density in ("comfortable", "compact") else "comfortable"
    css = _css_text()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(f'<div class="ua-marker" data-theme="{theme}" data-density="{density}"></div>',
                unsafe_allow_html=True)


def goto(page: str) -> None:
    """Switch page. Safe to call from a button callback."""
    if page in PAGE_LABELS:
        st.session_state["ua_page"] = page


def _toggle_theme() -> None:
    new = "light" if st.session_state.get("ui_theme") == "dark" else "dark"
    st.session_state["ui_theme"] = new
    st.session_state["w_dark_mode"] = new == "dark"  # keep the sidebar toggle in sync


def _sync_theme_from_toggle() -> None:
    st.session_state["ui_theme"] = "dark" if st.session_state.get("w_dark_mode") else "light"


def _set_density(widget_key: str) -> None:
    st.session_state["ui_density"] = "compact" if st.session_state.get(widget_key) == "Compact" else "comfortable"


# ------------------------------------------------------------------ small building blocks
def render_header(title: str, subtitle: str = "") -> None:
    sub = f'<p class="ua-page-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(one_line(f'<div class="ua-page-head"><h1 class="ua-page-title">{html.escape(title)}</h1>{sub}</div>'),
                unsafe_allow_html=True)


def render_empty_state(mark: str, title: str, body: str) -> None:
    st.markdown(one_line(f"""
        <div class="ua-empty"><div class="ua-empty-mark">{html.escape(mark)}</div>
        <h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>"""), unsafe_allow_html=True)


def _stat_card(value, label: str) -> str:
    text = str(value)
    size = " ua-stat-sm" if len(text) > 8 else ""   # words / timestamps get a smaller size than numbers
    return (f'<div class="ua-card"><div class="ua-stat-value{size}">{html.escape(text)}</div>'
            f'<div class="ua-stat-label">{html.escape(label)}</div></div>')


# ------------------------------------------------------------------ sidebar & session controls
def render_session_controls(prefix: str) -> set:
    """Clear chat / New session / two-step Delete. Returns the set of requested actions."""
    actions: set = set()
    c1, c2 = st.columns(2)
    if c1.button("Clear chat", key=f"{prefix}_clear", icon=":material/mop:",
                 help="Empties the conversation on screen. Saved records are not touched."):
        actions.add("clear")
    if c2.button("New session", key=f"{prefix}_new", icon=":material/add_comment:",
                 help="Starts a fresh conversation under a new random session ID."):
        actions.add("new")
    confirm_key = f"{prefix}_confirm_delete"
    if not st.session_state.get(confirm_key):
        if st.button("Delete my session data", key=f"{prefix}_delete", icon=":material/delete:"):
            st.session_state[confirm_key] = True
            st.rerun()
    else:
        st.warning("This permanently removes this session's saved chat and escalation records.")
        a, b = st.columns(2)
        if a.button("Yes, delete", key=f"{prefix}_delete_yes", type="primary"):
            st.session_state[confirm_key] = False
            actions.add("delete")
        if b.button("Cancel", key=f"{prefix}_delete_no"):
            st.session_state[confirm_key] = False
            st.rerun()
    return actions


def render_sidebar(session_id: str, n_messages: int) -> tuple[bool, set]:
    """Brand, navigation, session card, theme + privacy controls. Returns (opted_out, actions)."""
    with st.sidebar:
        st.markdown(one_line("""
            <div class="ua-brand"><div class="ua-brand-mark">UA</div>
            <div><div class="ua-brand-name">UniAssist AI</div>
            <div class="ua-brand-sub">Student support, any hour</div></div></div>"""), unsafe_allow_html=True)

        current = st.session_state.get("ua_page", "dashboard")
        for key, label, icon in PAGES:
            st.button(label, key=f"nav_{key}", icon=icon, on_click=goto, args=(key,),
                      type="primary" if key == current else "secondary")

        st.markdown('<div class="ua-side-title">Appearance</div>', unsafe_allow_html=True)
        st.toggle("Dark mode", value=st.session_state.get("ui_theme") == "dark", key="w_dark_mode",
                  on_change=_sync_theme_from_toggle)

        st.markdown('<div class="ua-side-title">Privacy</div>', unsafe_allow_html=True)
        opted_out = st.checkbox("Don't save my chat (opt out)", key="chat_opt_out_checkbox",
                                help="Nothing from this chat is written to the database.")
        status = "Not saving this chat" if opted_out else "Saving scrubbed text only"
        st.markdown(one_line(f"""
            <div class="ua-side-card"><b>{html.escape(status)}</b><br>
            Session <b>{html.escape(session_id)}</b> · {n_messages} message{'s' if n_messages != 1 else ''}<br>
            Emails, phone numbers and ID numbers are masked before saving.</div>"""), unsafe_allow_html=True)

        st.markdown('<div class="ua-side-title">Session</div>', unsafe_allow_html=True)
        actions = render_session_controls("sb")
    return bool(opted_out), actions


# ------------------------------------------------------------------ dashboard
def _greeting() -> str:
    h = datetime.datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 18 else "Good evening"


def render_dashboard(stats: dict, insights: dict, faq_total: int, category_total: int):
    """Returns (kind, value) for a clicked action: ('goto', page) or ('prompt', text); otherwise None."""
    st.markdown(one_line(f"""
        <div class="ua-hero"><h1>{_greeting()}. What can we sort out today?</h1>
        <p>UniAssist AI answers questions about admissions, fees, housing, courses, IT and wellbeing,
        and passes you to a real person when it is not sure it has the right answer.</p>
        <div class="ua-hero-pills"><span class="ua-hero-pill">Emails, phone and ID numbers masked</span>
        <span class="ua-hero-pill">Opt out any time</span><span class="ua-hero-pill">Human hand-off</span></div>
        </div>"""), unsafe_allow_html=True)

    result = None
    a, b, c = st.columns(3)
    if a.button("Ask the assistant", key="dash_ask", icon=":material/chat:", type="primary"):
        result = ("goto", "assistant")
    if b.button("Browse the FAQ library", key="dash_faq", icon=":material/menu_book:"):
        result = ("goto", "faq")
    if c.button("Talk to a human", key="dash_human", icon=":material/support_agent:"):
        result = ("prompt", HUMAN_PROMPT)

    st.markdown('<div class="ua-section-title">This session</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [(stats["asked"], "Questions asked"), (stats["escalated"], "Handed to a human"),
             (faq_total, f"FAQ answers across {category_total} topics"),
             (f"{insights['avg_rating']:.1f} / 5" if insights["avg_rating"] is not None else "–",
              f"Average rating ({insights['n_rated']} rated)")]
    for col, (value, label) in zip(cols, cards):
        col.markdown(_stat_card(value, label), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        rows = ""
        top = insights["top_intents"]
        if top:
            peak = max(n for _, n in top)
            for name, n in top:
                rows += (f'<div class="ua-bar-row"><span>{html.escape(category_label(name))}</span>'
                         f'<div class="ua-bar-track"><div class="ua-bar-fill" style="width:{max(6, round(100 * n / peak))}%">'
                         f'</div></div><span>{n}</span></div>')
        else:
            rows = '<p class="ua-muted">No saved conversations yet. Topics will appear here once people start asking.</p>'
        st.markdown(f'<div class="ua-card"><h3>Most asked topics</h3>{rows}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ua-card"><h3>Try asking</h3></div>', unsafe_allow_html=True)
        for i, prompt in enumerate([PROMPT_LIBRARY["Fees & Aid"][0], PROMPT_LIBRARY["IT & Library"][1],
                                    PROMPT_LIBRARY["Housing"][0]]):
            if st.button(prompt, key=f"dash_try_{i}"):
                result = ("prompt", prompt)
    return result


# ------------------------------------------------------------------ chat
def render_chat_input(placeholder: str):
    return st.chat_input(placeholder)


def render_prompt_library(key_prefix: str):
    """Quick Prompt Library: tabs by topic. Returns the clicked prompt text, or None."""
    chosen = None
    tabs = st.tabs(list(PROMPT_LIBRARY) + ["Human help"])
    for tab, (topic, prompts) in zip(tabs, PROMPT_LIBRARY.items()):
        with tab:
            for i, p in enumerate(prompts):
                if st.button(p, key=f"{key_prefix}_{topic}_{i}"):
                    chosen = p
    with tabs[-1]:
        if st.button(HUMAN_PROMPT, key=f"{key_prefix}_human"):
            chosen = HUMAN_PROMPT
    return chosen


def render_suggestions(questions: list[str], key_prefix: str):
    """Suggested follow-up questions as buttons. Returns the clicked question or None."""
    if not questions:
        return None
    st.markdown('<div class="ua-side-title">Suggested next questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(questions))
    chosen = None
    for i, (col, q) in enumerate(zip(cols, questions)):
        if col.button(q, key=f"{key_prefix}_{i}"):
            chosen = q
    return chosen


def render_chat_message(msg: dict, idx: int, key_prefix: str = "chat"):
    """Draw one message. For assistant messages also draw rate/copy actions.
    Returns 'up' / 'down' when a rating button was clicked, else None."""
    st.markdown(bubble_html(msg), unsafe_allow_html=True)
    if msg.get("role") != "assistant":
        return None
    action = None
    can_rate = bool(msg.get("log_id")) and not msg.get("rated")
    c_up, c_down, c_copy, _ = st.columns([1, 1, 1.4, 6])
    if can_rate:
        if c_up.button("", key=f"up{msg['log_id']}", icon=":material/thumb_up:", help="Helpful"):
            action = "up"
        if c_down.button("", key=f"down{msg['log_id']}", icon=":material/thumb_down:", help="Not helpful"):
            action = "down"
    with c_copy:
        with st.popover("Copy", icon=":material/content_copy:", help="Show the reply as copyable text"):
            st.caption("Use the copy icon at the top right of the box.")
            st.code(msg.get("content", ""), language=None)
    return action


def render_typing_placeholder():
    """Returns an st.empty() showing the animated typing indicator; call .empty() when done."""
    ph = st.empty()
    ph.markdown(TYPING_HTML, unsafe_allow_html=True)
    return ph


def render_chat_toolbar(messages: list[dict]) -> None:
    """Layout switch (comfortable/compact), prompt-library hint and local export."""
    c1, c2 = st.columns([3, 2])
    with c1:
        st.radio("Chat layout", ["Comfortable", "Compact"], horizontal=True, key="w_density_chat",
                 index=1 if st.session_state.get("ui_density") == "compact" else 0,
                 on_change=_set_density, args=("w_density_chat",))
    with c2:
        with st.popover("Export chat", icon=":material/download:"):
            if messages:
                st.caption("Saved on your device only. Nothing is uploaded.")
                st.download_button("Plain text (.txt)", export_text(messages), "uniassist_chat.txt", "text/plain",
                                   key="dl_txt")
                st.download_button("Markdown (.md)", export_markdown(messages), "uniassist_chat.md", "text/markdown",
                                   key="dl_md")
                st.download_button("JSON (.json)", export_json(messages), "uniassist_chat.json", "application/json",
                                   key="dl_json")
            else:
                st.caption("Nothing to export yet. Ask a question first.")


def render_chat_welcome() -> None:
    render_empty_state("UA", "Ask me anything about student life",
                       "Pick a prompt below or type your own question. I'm an automated assistant, so ask for a human "
                       "at any time.")


# ------------------------------------------------------------------ FAQ library
def render_faq_library(df, counts: dict, total_all: int, page_size: int = 10):
    """Category buttons + search box. Returns (selected_category, search_text)."""
    cats = ["all"] + list(counts)
    cur = st.session_state.get("faq_category", "all")
    if cur not in cats:
        cur = "all"
    st.markdown('<div class="ua-side-title">Categories</div>', unsafe_allow_html=True)
    per_row = 5
    for start in range(0, len(cats), per_row):
        row = cats[start:start + per_row]
        cols = st.columns(per_row)
        for col, cat in zip(cols, row):
            label = f"All ({total_all})" if cat == "all" else f"{category_label(cat)} ({counts[cat]})"
            if col.button(label, key=f"faq_cat_{cat}", type="primary" if cat == cur else "secondary"):
                st.session_state["faq_category"] = cat
                st.session_state["faq_limit"] = page_size
                st.rerun()
    query = st.text_input("Search the library", key="faq_search", placeholder="e.g. scholarship, Wi-Fi, deferral")
    return cur, query


def render_faq_entries(view, page_size: int = 10):
    """Draw the filtered FAQ rows. Returns ('ask', question) or None."""
    if len(view) == 0:
        render_empty_state("?", "No matching questions",
                           "Try a different word or pick another category. You can also ask the assistant directly.")
        return None
    limit = st.session_state.get("faq_limit", page_size)
    st.caption(f"Showing {min(limit, len(view))} of {len(view)} questions")
    result = None
    for _, row in view.head(limit).iterrows():
        with st.expander(row["question"]):
            st.markdown(f'<span class="ua-tag">{html.escape(category_label(row["category"]))} · '
                        f'{html.escape(str(row.get("subcategory", "")).replace("_", " "))}</span>',
                        unsafe_allow_html=True)
            st.write(row["answer"])
            c1, c2, _ = st.columns([2, 2, 4])
            if c1.button("Ask the assistant", key=f"faq_ask_{row['id']}", icon=":material/chat:"):
                result = ("ask", row["question"])
            with c2:
                with st.popover("Copy answer", icon=":material/content_copy:"):
                    st.code(row["answer"], language=None)
    if len(view) > limit:
        if st.button("Show more", key="faq_more"):
            st.session_state["faq_limit"] = limit + page_size
            st.rerun()
    return result


# ------------------------------------------------------------------ agent support
def render_agent_dashboard(df, error: str | None = None, max_rows: int = 100) -> None:
    """Escalation records as an actual table (theme-aware HTML, all values escaped)."""
    if error:
        st.error(f"The escalation records could not be loaded: {error}")
        return
    if df is None or len(df) == 0:
        render_empty_state("0", "No escalations yet",
                           "When the assistant hands a conversation to a human, it appears here with the reason.")
        return

    total, open_n = len(df), int((df["status"] == "open").sum())
    by_reason = df["reason"].fillna("unknown").value_counts()
    cols = st.columns(4)
    cols[0].markdown(_stat_card(total, "Escalations"), unsafe_allow_html=True)
    cols[1].markdown(_stat_card(open_n, "Open"), unsafe_allow_html=True)
    top_reason = AGENT_REASON_LABELS.get(by_reason.index[0], by_reason.index[0])
    cols[2].markdown(_stat_card(top_reason, "Most common reason"), unsafe_allow_html=True)
    cols[3].markdown(_stat_card(str(df["created_at"].iloc[0]), "Latest"), unsafe_allow_html=True)

    st.markdown('<div class="ua-section-title">Records</div>', unsafe_allow_html=True)
    options = list(by_reason.index)
    picked = st.multiselect("Filter by reason", options, default=options, key="agent_reasons",
                            format_func=lambda r: AGENT_REASON_LABELS.get(r, r))
    view = df[df["reason"].fillna("unknown").isin(picked)] if picked else df.iloc[0:0]
    if len(view) == 0:
        render_empty_state("–", "Nothing matches this filter", "Select at least one reason to see records.")
        return

    body = ""
    for _, r in view.head(max_rows).iterrows():
        reason = str(r["reason"]) if r["reason"] is not None else "unknown"
        pill = "ua-pill ua-pill-crisis" if reason == "crisis" else "ua-pill"
        status_cls = "ua-pill ua-pill-open" if r["status"] == "open" else "ua-pill"
        body += (f'<tr><td>{html.escape(str(r["id"]))}</td><td>{html.escape(str(r["created_at"]))}</td>'
                 f'<td><span class="{pill}">{html.escape(AGENT_REASON_LABELS.get(reason, reason))}</span></td>'
                 f'<td>{escape_text(r["user_query"])}</td>'
                 f'<td><span class="{status_cls}">{html.escape(str(r["status"]))}</span></td></tr>')
    st.markdown(one_line(f"""
        <div class="ua-table-wrap"><table class="ua-table"><thead><tr><th>ID</th><th>Time</th><th>Reason</th>
        <th>Student message (masked)</th><th>Status</th></tr></thead><tbody>{body}</tbody></table></div>"""),
        unsafe_allow_html=True)
    if len(view) > max_rows:
        st.caption(f"Showing the newest {max_rows} of {len(view)} records. Download the CSV for the full list.")
    st.download_button("Download records (CSV)", view.to_csv(index=False), "escalations.csv", "text/csv",
                       key="dl_escalations")


# ------------------------------------------------------------------ settings & feedback
def render_settings(scrub_fn, use_flan: bool, opted_out: bool, session_id: str) -> set:
    """Privacy & Settings page body. Returns requested session actions."""
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ua-section-title">Your privacy</div>', unsafe_allow_html=True)
        st.markdown(one_line(f"""
            <div class="ua-card"><b>{'Not saving this chat' if opted_out else 'Saving scrubbed text only'}</b>
            <p class="ua-muted">Chats are stored as masked text under the random session ID
            <b>{html.escape(session_id)}</b>, never linked to your name. Emails, phone numbers and long ID
            numbers are masked before saving. Names and addresses are <b>not</b> detected, so please don't type
            them. Use the opt-out checkbox in the sidebar to stop saving altogether.</p></div>"""),
            unsafe_allow_html=True)
        st.markdown('<div class="ua-section-title">Session controls</div>', unsafe_allow_html=True)
        actions = render_session_controls("pv")
    with right:
        st.markdown('<div class="ua-section-title">Appearance</div>', unsafe_allow_html=True)
        is_dark = st.session_state.get("ui_theme") == "dark"
        st.button("Switch to light mode" if is_dark else "Switch to dark mode", key="pv_theme",
                  icon=":material/light_mode:" if is_dark else ":material/dark_mode:", on_click=_toggle_theme)
        st.radio("Chat layout", ["Comfortable", "Compact"], horizontal=True, key="w_density_settings",
                 index=1 if st.session_state.get("ui_density") == "compact" else 0,
                 on_change=_set_density, args=("w_density_settings",))

        st.markdown('<div class="ua-section-title">See the masking work</div>', unsafe_allow_html=True)
        sample = st.text_input("Type something with an email or phone number", key="mask_demo",
                               placeholder="Reach me at jo@example.com or +91 98765 43210")
        if sample:
            st.code(scrub_fn(sample), language=None)
        else:
            st.caption("The same masking is applied to every message before it is saved.")

        st.markdown('<div class="ua-section-title">Experimental</div>', unsafe_allow_html=True)
        st.checkbox("Use FLAN-T5 for answers (experimental)", value=use_flan, key="w_use_flan",
                    on_change=lambda: st.session_state.update(use_flan=bool(st.session_state.get("w_use_flan"))),
                    help="Generates free-text answers with google/flan-t5-small. Those answers are not logged, "
                         "rated or escalated. Crisis and human-request messages still use the standard assistant.")
    return actions


def render_feedback(items: list[dict], insights: dict):
    """Feedback page. `items` are answered turns of this session. Returns (log_id, rating) or None."""
    cols = st.columns(3)
    cols[0].markdown(_stat_card(len(items), "Answers this session"), unsafe_allow_html=True)
    cols[1].markdown(_stat_card(sum(1 for i in items if i["rated"]), "You rated"), unsafe_allow_html=True)
    avg = f"{insights['avg_rating']:.1f} / 5" if insights["avg_rating"] is not None else "–"
    cols[2].markdown(_stat_card(avg, f"Average rating, all users ({insights['n_rated']})"), unsafe_allow_html=True)

    if not items:
        render_empty_state("★", "Nothing to rate yet",
                           "Ask the assistant a question, then come back to tell us whether the answer helped.")
        return None
    st.markdown('<div class="ua-section-title">Rate this session\'s answers</div>', unsafe_allow_html=True)
    result = None
    for it in items:
        with st.container(border=True):
            st.markdown(f"**{html.escape(it['question'])}**")
            st.write(it["answer"])
            if it["log_id"] is None:
                st.caption("This answer was not saved (opt-out or experimental mode), so it can't be rated.")
            elif it["rated"]:
                st.caption("Thanks, your feedback was recorded: "
                           + ("helpful" if it.get("rating") == 5 else "not helpful"))
            else:
                a, b, _ = st.columns([1.3, 1.6, 5])
                if a.button("Helpful", key=f"fb_up_{it['log_id']}", icon=":material/thumb_up:"):
                    result = (it["log_id"], 5)
                if b.button("Not helpful", key=f"fb_down_{it['log_id']}", icon=":material/thumb_down:"):
                    result = (it["log_id"], 1)
    return result
