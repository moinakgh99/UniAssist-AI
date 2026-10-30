"""SQLite storage. Only scrubbed text is stored, keyed by a random session id."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chatbot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, user_query TEXT, intent TEXT, confidence_score REAL,
    bot_response TEXT, escalated TEXT CHECK (escalated IN ('Y','N')), rating INTEGER,
    escalation_reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, reason TEXT, user_query TEXT,
    status TEXT DEFAULT 'open', created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_turn(conn, session_id, res):
    cur = conn.execute(
        "INSERT INTO chat_log (session_id,user_query,intent,confidence_score,bot_response,escalated,escalation_reason)"
        " VALUES (?,?,?,?,?,?,?)",
        (session_id, res["user_query"], res["intent"], round(res["confidence"], 4),
         res["reply"], "Y" if res["escalated"] else "N", res["reason"]))
    if res["escalated"]:
        conn.execute("INSERT INTO escalations (session_id,reason,user_query) VALUES (?,?,?)",
                     (session_id, res["reason"], res["user_query"]))
    conn.commit()
    return cur.lastrowid


def rate(conn, log_id, rating):
    conn.execute("UPDATE chat_log SET rating=? WHERE id=?", (rating, log_id))
    conn.commit()


def delete_session(conn, session_id):
    conn.execute("DELETE FROM chat_log WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM escalations WHERE session_id=?", (session_id,))
    conn.commit()


def escalations_df(conn):
    return pd.read_sql_query(
        "SELECT id, created_at, reason, user_query, status FROM escalations ORDER BY id DESC", conn)


def export_chat_log(conn):
    """Logged chats in the same column layout as the project's dataset.csv."""
    return pd.read_sql_query(
        "SELECT session_id, user_query, intent, confidence_score, bot_response, escalated, rating FROM chat_log", conn)
