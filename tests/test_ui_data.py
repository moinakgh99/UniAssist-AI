import datetime
import json
import unittest

import pandas as pd

from tests import support
from tests.support import nlp

from ui_data import (PROMPT_LIBRARY, HUMAN_PROMPT, all_library_prompts, bubble_html, escape_text, export_json,
                     export_markdown, export_text, faq_category_counts, filter_faq, load_faq_library,
                     log_insights, session_stats, suggest_questions, escalation_summary)


class UiData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_faq_library(nlp.FAQ_PATH)

    def test_every_library_prompt_is_a_real_faq_question(self):
        known = set(self.lib["question"])
        for p in all_library_prompts():
            self.assertIn(p, known, p)

    def test_library_loaded(self):
        self.assertGreater(len(self.lib), 400); self.assertEqual(self.lib["question"].duplicated().sum(), 0)
        self.assertIn("housing", faq_category_counts(self.lib))

    def test_filter(self):
        self.assertTrue((filter_faq(self.lib, "housing")["category"] == "housing").all())
        hits = filter_faq(self.lib, "all", "scholarship")
        self.assertGreater(len(hits), 0)
        self.assertEqual(len(filter_faq(self.lib, "housing", "zzzz-no-match")), 0)

    def test_suggestions_exclude_asked_and_have_fallback(self):
        asked = ["How do I apply for on-campus housing?"]
        s = suggest_questions(self.lib, "housing", asked, n=3)
        self.assertEqual(len(s), 3); self.assertNotIn(asked[0], s)
        self.assertEqual(len(suggest_questions(self.lib, "smalltalk", [], n=3)), 3)

    def test_html_escaping(self):
        self.assertNotIn("<script>", escape_text("<script>alert(1)</script>"))
        out = bubble_html({"role": "user", "content": "<img src=x onerror=alert(1)>", "timestamp": datetime.datetime.now()})
        self.assertNotIn("<img", out); self.assertIn("&lt;img", out); self.assertNotIn("\n", out)

    def test_escalation_badge_and_reason_escaped(self):
        out = bubble_html({"role": "assistant", "content": "hi", "escalated": True, "reason_text": "<b>x</b>"})
        self.assertIn("Handed to a human agent", out); self.assertNotIn("<b>x</b>", out)

    def test_exports(self):
        msgs = [{"role": "user", "content": "hi", "timestamp": datetime.datetime(2026, 1, 2, 3, 4, 5)},
                {"role": "assistant", "content": "hello", "escalated": True}]
        self.assertIn("hello", export_text(msgs)); self.assertIn("handed to a human agent", export_text(msgs))
        self.assertIn("**You**", export_markdown(msgs))
        data = json.loads(export_json(msgs)); self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["time"], "2026-01-02 03:04:05")

    def test_stats_and_insights(self):
        msgs = [{"role": "user"}, {"role": "assistant", "escalated": True, "rated": True}, {"role": "user"}]
        self.assertEqual(session_stats(msgs), {"asked": 2, "answered": 1, "escalated": 1, "rated": 1})
        ins = log_insights(pd.DataFrame({"intent": ["a", "a", "b"], "escalated": ["Y", "N", "N"], "rating": [5, None, 1]}))
        self.assertEqual(ins["total"], 3); self.assertEqual(ins["avg_rating"], 3.0); self.assertEqual(ins["top_intents"][0], ("a", 2))
        self.assertEqual(log_insights(None)["total"], 0)
        self.assertEqual(escalation_summary(pd.DataFrame({"status": ["open"], "reason": ["crisis"]}))["open"], 1)


if __name__ == "__main__":
    unittest.main()
