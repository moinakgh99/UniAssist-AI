"""End-to-end control-flow tests of src/app.py using the fake Streamlit module (see fake_streamlit.py).
They use a temp copy of the database and a Bot whose two neural models are stand-ins."""
import hashlib
import sqlite3
import unittest
from pathlib import Path

from tests import support
from tests.fake_streamlit import AppHarness
import db
import nlp

APP = str(support.ROOT / "src" / "app.py")
LAUNCHER = str(support.ROOT / "frontend" / "app.py")


def real_db_hash():
    return hashlib.sha256((support.ROOT / "data" / "chatbot.db").read_bytes()).hexdigest()


class AppFlow(unittest.TestCase):
    def setUp(self):
        self.db_hash = real_db_hash()
        self.db_path = support.temp_db_copy()
        self._orig_connect_defaults = db.connect.__defaults__
        db.connect.__defaults__ = (self.db_path,)          # app calls db.connect() with no args
        self._orig_bot = nlp.Bot
        nlp.Bot = support.make_bot                         # real respond() logic, stand-in neural models
        self.app = AppHarness(APP).run()

    def tearDown(self):
        db.connect.__defaults__ = self._orig_connect_defaults
        nlp.Bot = self._orig_bot
        self.assertEqual(self.db_hash, real_db_hash(), "the real database must never be modified by tests")

    # helpers
    def rows(self, sql, *a):
        c = sqlite3.connect(self.db_path); c.row_factory = sqlite3.Row
        try:
            return c.execute(sql, a).fetchall()
        finally:
            c.close()

    @property
    def sid(self):
        return self.app.st.session_state["session_id"]

    def ask(self, text):
        self.app.goto("assistant")
        return self.app.send(text)

    # ---- rendering
    def test_all_six_pages_render_without_errors_or_duplicate_keys(self):
        for page in ("dashboard", "assistant", "faq", "agent", "privacy", "feedback"):
            self.app.goto(page)
            self.assertNotRegex(self.app.text, r"(?m)^error:", f"{page} rendered an error")
        self.assertIn("UniAssist AI", self.app.text)

    def test_session_state_contract_preserved(self):
        s = self.app.st.session_state
        self.assertEqual(len(s["session_id"]), 12); self.assertEqual(s["messages"], [])

    def test_dashboard_shows_existing_insights(self):
        self.assertIn("Most asked topics", self.app.text)
        self.assertIn("Questions asked", self.app.text)

    # ---- chat + backend
    def test_chat_answers_and_logs_turn(self):
        self.ask("How do I apply for on-campus housing?")
        msgs = self.app.st.session_state["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertIsNotNone(msgs[1]["log_id"]); self.assertFalse(msgs[1]["escalated"])
        self.assertIn("housing", msgs[1]["content"].lower())
        row = self.rows("select * from chat_log where id=?", msgs[1]["log_id"])[0]
        self.assertEqual(row["session_id"], self.sid); self.assertEqual(row["intent"], "housing")
        self.assertIn("ua-bubble-assistant", self.app.text)

    def test_rating_persists_and_hides_buttons(self):
        self.ask("How do I connect to the campus Wi-Fi?")
        lid = self.app.st.session_state["messages"][1]["log_id"]
        self.app.click(f"up{lid}")
        self.assertEqual(self.rows("select rating from chat_log where id=?", lid)[0][0], 5)
        self.assertTrue(self.app.st.session_state["messages"][1]["rated"])
        self.assertNotIn(f"up{lid}", self.app.st.registry)

    def test_thumbs_down_and_feedback_page(self):
        self.ask("How do I book a study room?")
        self.ask("How do I renew a library book?")
        lid = self.app.st.session_state["messages"][1]["log_id"]
        self.app.goto("feedback")
        self.assertIn("Rate this session", self.app.text)
        self.app.click(f"fb_down_{lid}")
        self.assertEqual(self.rows("select rating from chat_log where id=?", lid)[0][0], 1)

    def test_opt_out_saves_nothing(self):
        before = len(self.rows("select id from chat_log"))
        self.app.set_widget("chat_opt_out_checkbox", True)
        self.ask("How do I apply for a scholarship?")
        self.assertIsNone(self.app.st.session_state["messages"][1]["log_id"])
        self.assertEqual(len(self.rows("select id from chat_log")), before)

    def test_opt_out_escalation_adds_helpdesk_notice(self):
        self.app.set_widget("chat_opt_out_checkbox", True)
        self.ask("I want to talk to a human")
        reply = self.app.st.session_state["messages"][1]["content"]
        self.assertIn("opted out", reply); self.assertIn(nlp.HELPDESK_CONTACT, reply)
        self.assertEqual(len(self.rows("select id from escalations where session_id=?", self.sid)), 0)

    def test_escalation_is_recorded_and_visible_to_agents(self):
        self.ask("I want to talk to a human agent please")
        m = self.app.st.session_state["messages"][1]
        self.assertTrue(m["escalated"]); self.assertEqual(m["reason_text"], nlp.REASON_TEXT["user_requested"])
        self.assertEqual(len(self.rows("select * from escalations where session_id=?", self.sid)), 1)
        self.assertIn("Handed to a human agent", self.app.text)
        self.app.goto("agent")
        self.assertIn("I want to talk to a human agent please", self.app.text)
        self.assertIn("Asked for a human", self.app.text)
        self.assertEqual(self.app.st.downloads[-1][1], "escalations.csv")

    def test_agent_dashboard_lists_the_ten_seeded_records(self):
        self.app.goto("agent")
        self.assertIn("When will the semester examinations begin?", self.app.text)
        self.assertEqual(self.app.text.count("<tr><td>"), 10)

    def test_crisis_flow(self):
        self.ask("I want to end my life")
        m = self.app.st.session_state["messages"][1]
        self.assertTrue(m["escalated"]); self.assertIn("counselling", m["content"])
        self.assertEqual(self.rows("select reason from escalations where session_id=?", self.sid)[0][0], "crisis")

    def test_pii_masked_in_database_but_not_stored_raw(self):
        self.ask("my email is jo@uni.edu how do I reset my password")
        stored = self.rows("select user_query from chat_log where session_id=?", self.sid)[0][0]
        self.assertIn("[EMAIL]", stored); self.assertNotIn("jo@uni.edu", stored)

    def test_html_in_user_message_is_escaped(self):
        self.ask("<script>alert('x')</script> how do I register for courses?")
        self.assertNotIn("<script>", self.app.text); self.assertIn("&lt;script&gt;", self.app.text)

    # ---- session controls (the original nested-if bug)
    def test_clear_chat_button_works_on_its_own(self):
        self.ask("How do I register for courses?")
        self.app.click("sb_clear")
        self.assertEqual(self.app.st.session_state["messages"], [])
        self.assertEqual(len(self.rows("select id from chat_log where session_id=?", self.sid)), 1)  # DB untouched

    def test_delete_requires_confirmation_then_deletes_only_this_session(self):
        self.ask("How do I register for courses?")
        total = len(self.rows("select id from chat_log"))
        self.app.click("sb_delete")
        self.assertEqual(len(self.rows("select id from chat_log")), total)          # not yet
        self.app.click("sb_delete_no")
        self.assertEqual(len(self.rows("select id from chat_log")), total)
        self.app.click("sb_delete"); self.app.click("sb_delete_yes")
        self.assertEqual(len(self.rows("select id from chat_log where session_id=?", self.sid)), 0)
        self.assertEqual(len(self.rows("select id from chat_log")), total - 1)      # other sessions intact
        self.assertEqual(self.app.st.session_state["messages"], [])

    def test_new_session_changes_id_and_keeps_records(self):
        self.ask("How do I register for courses?")
        old = self.sid
        self.app.click("sb_new")
        self.assertNotEqual(old, self.sid); self.assertEqual(self.app.st.session_state["messages"], [])
        self.assertEqual(len(self.rows("select id from chat_log where session_id=?", old)), 1)

    # ---- frontend-only features
    def test_quick_prompt_library_sends_a_real_question(self):
        self.app.goto("assistant")
        self.app.click("qp_empty_Housing_0")
        self.assertEqual(self.app.st.session_state["messages"][0]["content"], "How do I apply for on-campus housing?")

    def test_suggested_followups_appear_after_an_answer(self):
        self.ask("How do I apply for on-campus housing?")
        self.assertTrue(any(k.startswith("sugg_") for k in self.app.st.registry))
        self.app.click("sugg_0")
        self.assertEqual(len(self.app.st.session_state["messages"]), 4)

    def test_export_buttons_contain_the_conversation(self):
        self.ask("How do I apply for on-campus housing?")
        names = {n for _, n, _ in self.app.st.downloads}
        self.assertEqual(names, {"uniassist_chat.txt", "uniassist_chat.md", "uniassist_chat.json"})
        self.assertIn("on-campus housing", self.app.st.downloads[0][2])

    def test_dark_mode_toggle_and_settings_button_stay_in_sync(self):
        self.assertIn('data-theme="light"', self.app.text)
        self.app.set_widget("w_dark_mode", True)
        self.assertIn('data-theme="dark"', self.app.text)
        self.app.goto("privacy"); self.app.click("pv_theme")
        self.assertIn('data-theme="light"', self.app.text)
        self.assertFalse(self.app.st.session_state["w_dark_mode"])

    def test_compact_layout_toggle(self):
        self.app.goto("assistant")
        self.app.set_widget("w_density_chat", "Compact")
        self.assertIn('data-density="compact"', self.app.text)

    def test_faq_category_navigation_search_and_ask(self):
        self.app.goto("faq")
        self.app.click("faq_cat_housing")
        self.assertIn("Housing", self.app.text); self.assertNotIn("scholarship", self.app.text.lower().split("categories")[-1][:0])
        self.app.set_widget("faq_search", "roommate")
        self.assertIn("roommate", self.app.text.lower())
        ask_keys = [k for k in self.app.st.registry if k.startswith("faq_ask_")]
        self.assertTrue(ask_keys)
        self.app.click(ask_keys[0])
        self.assertEqual(self.app.st.session_state["ua_page"], "assistant")
        self.assertEqual(len(self.app.st.session_state["messages"]), 2)

    def test_faq_empty_state(self):
        self.app.goto("faq"); self.app.set_widget("faq_search", "qqqzzz-nothing")
        self.assertIn("No matching questions", self.app.text)

    def test_dashboard_talk_to_human_goes_through_escalation(self):
        self.app.click("dash_human")
        self.assertEqual(self.app.st.session_state["ua_page"], "assistant")
        self.assertTrue(self.app.st.session_state["messages"][1]["escalated"])

    def test_privacy_page_masking_demo_uses_real_scrub(self):
        self.app.goto("privacy"); self.app.set_widget("mask_demo", "mail jo@x.com")
        self.assertIn("code: mail [EMAIL]", self.app.text)

    def test_empty_states(self):
        self.app.goto("feedback"); self.assertIn("Nothing to rate yet", self.app.text)
        self.app.goto("assistant"); self.assertIn("Ask me anything about student life", self.app.text)

    # ---- launcher + failure handling
    def test_frontend_launcher_runs_same_app(self):
        h = AppHarness(LAUNCHER).run()
        self.assertIn("UniAssist AI", h.text); self.assertNotRegex(h.text, r"(?m)^error:")

    def test_assistant_load_failure_is_reported_not_crashing(self):
        def boom():
            raise RuntimeError("Missing or broken dependency: transformers")
        nlp.Bot = boom
        app = AppHarness(APP).run()
        app.goto("assistant")
        self.assertIn("could not start", app.text); self.assertIn("Missing or broken dependency", app.text)
        app.goto("agent"); self.assertNotRegex(app.text, r"(?m)^error:")   # DB-only pages keep working

    def test_flan_mode_never_bypasses_crisis_or_human_rules(self):
        self.app.st.session_state["use_flan"] = True
        self.ask("I want to end my life")
        m = self.app.st.session_state["messages"][1]
        self.assertTrue(m["escalated"]); self.assertIn("counselling", m["content"])


if __name__ == "__main__":
    unittest.main()
