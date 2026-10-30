import unittest

import pandas as pd

from tests import support
from tests.support import nlp


class ScrubAndText(unittest.TestCase):
    def test_pii_masked(self):
        out = nlp.scrub("mail jo.smith@uni.edu or call +91 98765 43210, id AB1234567")
        self.assertIn("[EMAIL]", out); self.assertIn("[PHONE]", out); self.assertIn("[ID]", out)
        self.assertNotIn("@", out); self.assertNotIn("98765", out)

    def test_plain_text_untouched(self):
        self.assertEqual(nlp.scrub("How do I apply for housing?"), "How do I apply for housing?")

    def test_normalise_synonyms(self):
        self.assertEqual(nlp.normalise("Hostel FEES?"), "housing residence tuition")

    def test_crisis_human_smalltalk_patterns(self):
        self.assertTrue(nlp.CRISIS.search("I want to end my life"))
        self.assertTrue(nlp.HUMAN.search("let me talk to a human"))
        self.assertTrue(nlp.SMALLTALK.match("Thanks!"))
        self.assertFalse(nlp.SMALLTALK.match("thanks but how do I pay fees"))


class ImportAndErrors(unittest.TestCase):
    def test_module_imports_without_heavy_dependencies(self):
        # nlp.py must import even if transformers / sentence-transformers are missing (they are optional here)
        self.assertTrue(hasattr(nlp, "Bot"))

    def test_single_import_block(self):
        src = (support.SRC / "nlp.py").read_text(encoding="utf-8")
        self.assertEqual(src.count("from transformers import pipeline"), 1)
        self.assertEqual(src.count("W_RETR = "), 1)

    def test_bot_reports_missing_dependencies_clearly(self):
        if nlp.pipeline is not None and nlp.SentenceTransformer is not None:
            self.skipTest("heavy dependencies are installed")
        with self.assertRaises(RuntimeError) as cm:
            nlp.Bot()
        self.assertIn("requirements.txt", str(cm.exception))

    def test_bad_model_path_is_a_clear_error(self):
        orig = (nlp.pipeline, nlp.SentenceTransformer)
        nlp.pipeline, nlp.SentenceTransformer = object, object
        try:
            with self.assertRaises(RuntimeError) as cm:
                nlp.Bot(model_path="/nonexistent/intent.joblib")
            self.assertIn("intent model", str(cm.exception))
        finally:
            nlp.pipeline, nlp.SentenceTransformer = orig


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faq = pd.read_csv(nlp.FAQ_PATH)

    def test_retriever_without_embedder_builds_topics(self):  # the train.py regression
        r = nlp.Retriever(self.faq)
        self.assertGreater(len(r.topics), 50)
        with self.assertRaises(RuntimeError):
            r.scores("hello")

    def test_retrieval_with_embedder_finds_housing(self):
        r = nlp.Retriever(self.faq, embedder=support.HashEmbedder())
        top, score = r.search("How do I apply for on-campus housing?", k=1)[0]
        self.assertEqual(top["category"], "housing"); self.assertGreater(score, 0.3)

    def test_answer_styles_and_audience(self):
        r = nlp.Retriever(self.faq, embedder=support.HashEmbedder())
        topic = r.search("apply for on-campus housing", k=1)[0][0]
        self.assertTrue(r.answer(topic, "how do I apply for housing"))
        self.assertIn("international student", r.answer(topic, "as an international student how do I apply"))


class BotRespond(unittest.TestCase):
    """Real Bot.respond() logic with stand-in neural models."""
    @classmethod
    def setUpClass(cls):
        cls.bot = support.make_bot()

    def test_normal_question_not_escalated(self):
        res = self.bot.respond("How do I apply for on-campus housing?")
        self.assertFalse(res["escalated"], res); self.assertEqual(res["intent"], "housing")
        self.assertTrue(res["reply"]); self.assertGreater(res["retrieval_score"], 0.3)

    def test_pii_is_scrubbed_in_result(self):
        res = self.bot.respond("my email is a.b@uni.edu how do I reset my password")
        self.assertIn("[EMAIL]", res["user_query"]); self.assertNotIn("a.b@uni.edu", res["user_query"])

    def test_crisis_escalates_with_resources(self):
        res = self.bot.respond("I want to end my life")
        self.assertTrue(res["escalated"]); self.assertEqual(res["reason"], "crisis")
        self.assertEqual(res["intent"], "mental_health"); self.assertIn("counselling", res["reply"])

    def test_smalltalk(self):
        res = self.bot.respond("hello")
        self.assertEqual(res["intent"], "smalltalk"); self.assertFalse(res["escalated"])

    def test_user_requested_human(self):
        res = self.bot.respond("I want to speak to a human agent")
        self.assertTrue(res["escalated"]); self.assertEqual(res["reason"], "user_requested")

    def test_negative_sentiment_escalates(self):
        res = self.bot.respond("this housing application is useless")
        self.assertTrue(res["escalated"]); self.assertEqual(res["reason"], "negative_sentiment")

    def test_low_confidence_escalates(self):
        res = self.bot.respond("zzqx blorp wibble")
        self.assertTrue(res["escalated"]); self.assertEqual(res["reason"], "low_confidence")

    def test_result_contract_keys(self):
        res = self.bot.respond("How do I connect to the campus Wi-Fi?")
        for k in ("user_query", "intent", "confidence", "retrieval_score", "sentiment", "fused", "reason",
                  "escalated", "reply"):
            self.assertIn(k, res)


if __name__ == "__main__":
    unittest.main()
