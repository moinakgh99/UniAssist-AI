import hashlib
import unittest

import sqlite3

from tests import support

import db  # noqa: E402


def sha(p):
    return hashlib.sha256(__import__("pathlib").Path(p).read_bytes()).hexdigest()


def fake_res(**kw):
    base = dict(user_query="How do I apply for housing?", intent="housing", confidence=0.66, reply="answer",
                escalated=False, reason=None)
    base.update(kw)
    return base


class DbTests(unittest.TestCase):
    def setUp(self):
        self.path = support.temp_db_copy()
        self.conn = db.connect(self.path)

    def test_existing_records_preserved(self):
        self.assertEqual(self.conn.execute("select count(*) from chat_log").fetchone()[0], 25)
        self.assertEqual(self.conn.execute("select count(*) from escalations").fetchone()[0], 10)

    def test_log_rate_delete_roundtrip(self):
        lid = db.log_turn(self.conn, "sess_test", fake_res())
        row = self.conn.execute("select * from chat_log where id=?", (lid,)).fetchone()
        self.assertEqual(row["session_id"], "sess_test"); self.assertEqual(row["escalated"], "N")
        db.rate(self.conn, lid, 5)
        self.assertEqual(self.conn.execute("select rating from chat_log where id=?", (lid,)).fetchone()[0], 5)
        db.delete_session(self.conn, "sess_test")
        self.assertEqual(self.conn.execute("select count(*) from chat_log where session_id='sess_test'").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("select count(*) from chat_log").fetchone()[0], 25)  # others untouched

    def test_escalation_logged_and_listed(self):
        db.log_turn(self.conn, "sess_esc", fake_res(escalated=True, reason="user_requested", user_query="human please"))
        df = db.escalations_df(self.conn)
        self.assertEqual(list(df.columns), ["id", "created_at", "reason", "user_query", "status"])
        self.assertEqual(len(df), 11); self.assertEqual(df.iloc[0]["user_query"], "human please")
        db.delete_session(self.conn, "sess_esc")
        self.assertEqual(len(db.escalations_df(self.conn)), 10)

    def test_export_chat_log_columns(self):
        df = db.export_chat_log(self.conn)
        self.assertEqual(list(df.columns), ["session_id", "user_query", "intent", "confidence_score",
                                            "bot_response", "escalated", "rating"])

    def test_fresh_database_schema(self):
        conn = db.connect(support.temp_db_copy(empty=True))
        self.assertEqual(len(db.escalations_df(conn)), 0)

    def test_real_database_never_modified_by_tests(self):
        before = sha(support.ROOT / "data" / "chatbot.db")
        db.log_turn(self.conn, "x", fake_res())
        self.assertEqual(before, sha(support.ROOT / "data" / "chatbot.db"))


if __name__ == "__main__":
    unittest.main()
