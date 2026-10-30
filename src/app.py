"""UniAssist AI - Streamlit front end.  Run from the chatbot/ folder with:  streamlit run src/app.py

This file owns every backend call (Bot.respond, db.log_turn, db.rate, db.delete_session, db.escalations_df,
db.export_chat_log). All drawing lives in ui_components.py; pure helpers live in ui_data.py.
"""
import datetime
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import ui_components as ui  # noqa: E402
import ui_data  # noqa: E402
from nlp import FAQ_PATH, HELPDESK_CONTACT, REASON_TEXT, Bot, scrub  # noqa: E402

st.set_page_config(page_title="UniAssist AI", page_icon=":material/school:", layout="wide",
                   initial_sidebar_state="expanded")


# ------------------------------------------------------------------ cached backend objects
@st.cache_resource
def get_bot():
    return Bot()


@st.cache_resource
def get_conn():
    return db.connect()


@st.cache_data
def get_faq_library():
    return ui_data.load_faq_library(FAQ_PATH)


# ------------------------------------------------------------------ session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]   # random, not linked to the student
    st.session_state.messages = []
for _key, _default in (("ui_theme", "light"), ("ui_density", "comfortable"), ("use_flan", False),
                       ("ua_page", "dashboard")):
    st.session_state.setdefault(_key, _default)

ui.load_css(st.session_state.ui_theme, st.session_state.ui_density)

try:
    conn = get_conn()
except Exception as exc:  # the database is essential: stop with a readable message
    ui.render_header("UniAssist AI", "The database could not be opened.")
    st.error(f"Could not open the chat database: {exc}")
    st.stop()

bot, bot_error = None, None
try:
    with st.spinner("Loading language models (the first start can take a minute)..."):
        bot = get_bot()
except Exception as exc:  # keep the DB-only pages usable and explain what is wrong on the chat page
    bot_error = str(exc)

try:
    faq_df, faq_error = get_faq_library(), None
except Exception as exc:
    faq_df, faq_error = None, str(exc)


# ------------------------------------------------------------------ helpers
def flash(message: str) -> None:
    st.session_state["_flash"] = message


def apply_session_actions(actions: set) -> None:
    """Delete / clear / new session. Each action is independent (the original nested one inside the other)."""
    if not actions:
        return
    if "delete" in actions:
        db.delete_session(conn, st.session_state.session_id)
        st.session_state.messages = []
        flash("Your saved chat data for this session was deleted.")
    elif "clear" in actions:
        st.session_state.messages = []
        flash("Chat history cleared.")
    elif "new" in actions:
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        flash("Started a new session.")
    st.rerun()


def go(page: str, prompt: str | None = None) -> None:
    if prompt:
        st.session_state["_pending_prompt"] = prompt
    ui.goto(page)
    st.rerun()


def get_insights() -> dict:
    try:
        return ui_data.log_insights(db.export_chat_log(conn))
    except Exception:
        return ui_data.log_insights(None)


def rate(log_id: int, value: int) -> None:
    db.rate(conn, log_id, value)
    for m in st.session_state.messages:
        if m.get("log_id") == log_id:
            m["rated"] = True
            m["rating"] = value
    st.rerun()


def generate_reply(prompt: str, opted_out: bool) -> tuple:
    """Run the existing pipeline for one prompt. Returns (user_message, assistant_message)."""
    res = bot.respond(prompt)
    reply, escalated, reason = res["reply"], res["escalated"], res["reason"]
    flan_used = False
    if st.session_state.use_flan and reason not in ("crisis", "user_requested"):
        # Experimental path (as in the old frontend/app.py): free-text FLAN-T5 answer on the masked text.
        # Crisis / human-request messages never take this path, so those safeguards always apply.
        from model import get_answer
        reply, escalated, reason, flan_used = get_answer(res["user_query"]), False, None, True

    log_id = None
    if not opted_out and not flan_used:
        try:
            log_id = db.log_turn(conn, st.session_state.session_id, res)
        except Exception as exc:
            st.toast(f"Could not save this turn: {exc}")
    if res["escalated"] and opted_out:
        reply += (f" Because you opted out, nothing is saved, so no one will see this chat. "
                  f"Please contact the helpdesk directly: {HELPDESK_CONTACT}.")

    user_msg = {"role": "user", "content": prompt, "timestamp": datetime.datetime.now()}
    bot_msg = {"role": "assistant", "content": reply, "log_id": log_id, "escalated": escalated,
               "reason_text": REASON_TEXT.get(reason, ""), "timestamp": datetime.datetime.now(),
               "intent": res["intent"]}
    return user_msg, bot_msg


# ------------------------------------------------------------------ sidebar (always rendered)
opted_out, side_actions = ui.render_sidebar(st.session_state.session_id, len(st.session_state.messages))
apply_session_actions(side_actions)
if st.session_state.get("_flash"):
    st.toast(st.session_state.pop("_flash"))
page = st.session_state.ua_page


# ------------------------------------------------------------------ pages
def page_dashboard():
    ui.render_header("Dashboard")
    stats = ui_data.session_stats(st.session_state.messages)
    total = len(faq_df) if faq_df is not None else 0
    cats = faq_df["category"].nunique() if faq_df is not None else 0
    click = ui.render_dashboard(stats, get_insights(), total, cats)
    if click:
        kind, value = click
        if kind == "prompt":
            go("assistant", value)
        else:
            go(value)


def page_assistant():
    ui.render_header("AI Assistant", "Ask in your own words. Answers come from the university FAQ knowledge base.")
    st.info("I'm an automated assistant, not a person. Please don't share personal details such as "
            "your name, ID number or phone number. Ask me to talk to a human at any time.")
    if bot is None:
        st.error(f"The assistant could not start. {bot_error}")
        ui.render_empty_state("!", "Assistant unavailable",
                              "The rest of the app still works. Fix the issue above and reload this page.")
        st.chat_input("The assistant is unavailable", disabled=True)
        return

    messages = st.session_state.messages
    ui.render_chat_toolbar(messages)
    chosen = None

    if not messages:
        ui.render_chat_welcome()
        chosen = ui.render_prompt_library("qp_empty")
    else:
        for i, m in enumerate(messages):
            action = ui.render_chat_message(m, i)
            if action and m.get("log_id"):
                rate(m["log_id"], 5 if action == "up" else 1)
        last = messages[-1]
        if last["role"] == "assistant" and faq_df is not None:
            asked = [m["content"] for m in messages if m["role"] == "user"]
            chosen = ui.render_suggestions(ui_data.suggest_questions(faq_df, last.get("intent"), asked), "sugg")
        with st.expander("Quick prompt library"):
            chosen = ui.render_prompt_library("qp_chat") or chosen

    typed = ui.render_chat_input("Ask about admissions, fees, housing, courses, IT, wellbeing...")
    prompt = typed or chosen or st.session_state.pop("_pending_prompt", None)
    if not prompt:
        return

    st.markdown(ui_data.bubble_html({"role": "user", "content": prompt, "timestamp": datetime.datetime.now()}),
                unsafe_allow_html=True)
    typing = ui.render_typing_placeholder()
    try:
        user_msg, bot_msg = generate_reply(prompt, opted_out)
    except Exception as exc:
        typing.empty()
        st.error(f"Sorry, something went wrong while answering: {exc}")
        return
    messages.append(user_msg)
    messages.append(bot_msg)
    st.rerun()


def page_faq():
    ui.render_header("FAQ Library", "Browse the same answers the assistant uses, by topic.")
    if faq_df is None:
        st.error(f"The FAQ data could not be loaded: {faq_error}")
        return
    counts = ui_data.faq_category_counts(faq_df)
    category, query = ui.render_faq_library(faq_df, counts, len(faq_df))
    if st.session_state.get("faq_sig") != (category, query):
        st.session_state["faq_sig"] = (category, query)
        st.session_state["faq_limit"] = 10
    click = ui.render_faq_entries(ui_data.filter_faq(faq_df, category, query))
    if click:
        go("assistant", click[1])


def page_agent():
    ui.render_header("Agent Support", "Escalated conversations waiting for a human. Shows only masked text from "
                                      "students who did not opt out.")
    try:
        df, error = db.escalations_df(conn), None
    except Exception as exc:
        df, error = None, str(exc)
    ui.render_agent_dashboard(df, error)


def page_privacy():
    ui.render_header("Privacy & Settings", "Control what is saved and how the app looks.")
    apply_session_actions(ui.render_settings(scrub, st.session_state.use_flan, opted_out,
                                             st.session_state.session_id))


def page_feedback():
    ui.render_header("Feedback", "Tell us whether answers helped. Ratings are stored with the saved chat turn.")
    items, msgs = [], st.session_state.messages
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        q = msgs[i - 1]["content"] if i > 0 and msgs[i - 1]["role"] == "user" else ""
        items.append({"question": q, "answer": m["content"], "log_id": m.get("log_id"),
                      "rated": bool(m.get("rated")), "rating": m.get("rating")})
    click = ui.render_feedback(list(reversed(items)), get_insights())
    if click:
        rate(*click)


PAGE_FUNCS = {"dashboard": page_dashboard, "assistant": page_assistant, "faq": page_faq,
              "agent": page_agent, "privacy": page_privacy, "feedback": page_feedback}
PAGE_FUNCS.get(page, page_dashboard)()
