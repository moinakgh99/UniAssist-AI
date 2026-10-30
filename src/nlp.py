"""Core NLP: PII scrub, intent classification, FAQ retrieval, sentiment, escalation rules."""
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Heavy / optional dependencies. They are imported defensively so that this module (PII scrubbing,
# regexes, FAQ topic parsing, ...) can still be imported without them; Bot() reports a clear error
# if they are needed but missing.
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - depends on the environment
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - depends on the environment
    SentenceTransformer = None

ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = ROOT / "data" / "faq_dataset.csv"
MODEL_PATH = ROOT / "models" / "intent.joblib"

# ---- tunable settings (report the values you finally choose) ----
W_RETR = 0.6      # weight of FAQ similarity vs classifier probability in the fused score
FUSED_MIN = 0.35  # below this fused score        -> escalate (low_confidence)
SENT_MIN = -0.5   # VADER compound <= this        -> escalate (negative_sentiment)

# TODO: fill these in with your campus's real contacts before the demo.
HELPDESK_CONTACT = "[add your university helpdesk phone/email]"
CRISIS_RESOURCES = (
    "Please reach out right now to your campus counselling centre ([add number]) or a "
    "national helpline ([add number]). If you are in immediate danger, call your local "
    "emergency number."
)

# ---------------------------------------------------------------- PII scrub
PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?<!\w)(?:\+?\d[\s-]?){10,}(?!\w)"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z]{0,3}\d{6,}\b"), "[ID]"),
]


def scrub(text: str) -> str:
    """Replace emails, phone numbers and long ID numbers. (Names are NOT detected - see report.)"""
    for rx, tag in PII_PATTERNS:
        text = rx.sub(tag, text)
    return text


# ---------------------------------------------------------- normalisation
SYNONYMS = {
    "uni": "university", "hostel": "housing residence", "fees": "tuition", "fee": "tuition",
    "instalments": "installments", "instalment": "installments", "cv": "resume",
    "marksheet": "transcripts", "subject": "course", "subjects": "courses",
    "prof": "faculty", "professor": "faculty", "counselor": "counseling",
    "counsellor": "counseling", "counselling": "counseling", "doctor": "medical",
    "wifi": "wi-fi", "postpone": "deferral", "aid": "financial aid",
    "vaccine": "immunization", "vaccines": "immunization", "vaccination": "immunization",
    "meeting": "appointment", "eat": "dining food", "jobs": "job employment", "marks": "grade",
    "challenge": "appeal", "contest": "appeal", "switch": "change", "credits": "degree requirements",
    "verification": "authentication multi-factor", "break": "leave of absence",
    "connecting": "connect", "btech": "undergraduate",
}


def normalise(text: str) -> str:
    """Lower-case, strip punctuation, expand a few slang/synonym words."""
    text = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    return " ".join(SYNONYMS.get(w, w) for w in text.split())


# ------------------------------------------------------------- sentiment
def _load_sia():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:  # fall back to the packaged VADER (same algorithm)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


SENTIMENT_ADDITIONS = {  # student-support words VADER misses (document these in the report)
    "useless": -2.5, "joke": -1.5, "nightmare": -2.0, "pathetic": -2.5, "ridiculous": -2.2,
    "unhelpful": -2.2, "nobody": -1.0, "waiting": -0.8, "ignoring": -2.0, "scam": -2.5,
    "rejected": -1.0, "waste": -1.8,
}

# ---------------------------------------------------------- FAQ retrieval
PATTERNS = [
    ("deadline", re.compile(r"^Is there a deadline to (.+)\?$")),
    ("docs", re.compile(r"^What do I need before I (.+)\?$")),
    ("process", re.compile(r"^What is the process to (.+)\?$")),
    ("where", re.compile(r"^Where can I get help if I want to (.+)\?$")),
    ("late", re.compile(r"^What if I miss the window to (.+)\?$")),
    ("online", re.compile(r"^Can I (.+) online\?$")),
    ("how", re.compile(r"^How (?:do|can) I (.+)\?$")),
]
PATTERN_KW = {"how-to", "process", "contact", "requirements", "deadline", "online", "late"}

QUERY_TYPE = [
    ("late", re.compile(r"\b(miss(ed)?|too late|after the deadline|deadline (has )?passed)\b")),
    ("deadline", re.compile(r"\b(deadline|last date|due date|by when|until when)\b")),
    ("docs", re.compile(r"\b(documents?|required|requirements?|bring|papers)\b")),
    ("online", re.compile(r"\b(online|remotely|from home)\b")),
    ("where", re.compile(r"\b(where|who)\b")),
]

AUDIENCE = [
    (re.compile(r"\bfirst[- ]year|freshers?\b"), "As a first-year student, start with your orientation materials and ask your advisor if anything is unclear."),
    (re.compile(r"\btransfer\b"), "As a transfer student, keep your previous institution's records ready, since they can affect eligibility or timing."),
    (re.compile(r"\binternational|foreign\b"), "As an international student, also check with the international student office, because visa or immigration rules can affect this."),
    (re.compile(r"\bgraduate student|masters|m\.?tech|phd\b"), "As a graduate student, your department or graduate school may add steps beyond the general process."),
    (re.compile(r"\bpart[- ]time student\b"), "As a part-time student, check whether your enrolment level changes eligibility, fees or deadlines."),
    (re.compile(r"\bcommut(er|e)\b"), "As a commuter, ask which steps can be completed remotely so you can save a trip to campus."),
    (re.compile(r"\bdisabilit(y|ies)|disabled\b"), "If you use accommodations, contact Disability Services early so arrangements are in place."),
    (re.compile(r"\bfinal[- ]year|last year\b"), "In your final year, check how this interacts with graduation deadlines."),
    (re.compile(r"\bonline student|distance learning\b"), "As an online student, ask about remote options and the best hours to reach staff in your time zone."),
]


class Retriever:
    """Finds the best FAQ topic using sentence embeddings for semantic similarity.
    The original TF‑IDF approach is replaced by a lightweight transformer model.
    """

    def __init__(self, faq: pd.DataFrame, embedder=None):
        # `embedder` is a SentenceTransformer. It may be None when only the parsed topics are needed
        # (e.g. src/train.py); scores()/search() then raise a clear error instead of failing obscurely.
        self.embedder = embedder
        gen = faq[faq["language"] == "en"]
        if "audience" in gen.columns:
            gen = gen[gen["audience"] == "general"]
        topics = {}
        for _, r in gen.iterrows():
            for name, rx in PATTERNS:
                m = rx.match(r["question"])
                if m:
                    key = (r["category"], m.group(1))
                    t = topics.setdefault(key, {
                        "category": r["category"],
                        "subcategory": r["subcategory"],
                        "action": m.group(1),
                        "kw": set(),
                        "answers": {}
                    })
                    t["answers"][name] = r["answer"]
                    t["kw"].update(k for k in str(r["keywords"]).split(";") if k and k not in PATTERN_KW)
                    break
        self.topics = list(topics.values())
        # Build canonical documents for each topic and embed them
        docs = [normalise(f"{t['action']} {' '.join(sorted(t['kw']))}") for t in self.topics]
        self.topic_embeddings = (
            self.embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            if self.embedder is not None else None
        )
        self.cats = np.array([t["category"] for t in self.topics])
        self.topic_docs = docs

    def scores(self, query: str) -> np.ndarray:
        """Return cosine similarity between query embedding and topic embeddings."""
        if self.embedder is None:
            raise RuntimeError("Retriever was created without an embedder, so it cannot score queries.")
        q_norm = self.embedder.encode([normalise(query)], convert_to_numpy=True, normalize_embeddings=True)
        # cosine similarity via dot product (embeddings already normalized)
        return np.dot(self.topic_embeddings, q_norm.squeeze())

    def search(self, query: str, category: str = None, k: int = 3):
        s = self.scores(query)
        if category:
            s = np.where(self.cats == category, s, 0.0)
        order = s.argsort()[::-1][:k]
        return [(self.topics[i], float(s[i])) for i in order if s[i] > 0]

    @staticmethod
    def answer(topic, query: str) -> str:
        q = query.lower()
        style = next((n for n, rx in QUERY_TYPE if rx.search(q)), "how")
        text = topic["answers"].get(style) or topic["answers"].get("how") or next(iter(topic["answers"].values()))
        for rx, addendum in AUDIENCE:
            if rx.search(q):
                text += " " + addendum
                break
        return text


# ------------------------------------------------------------- the bot
CRISIS = re.compile(r"suicid|kill myself|end my life|ending my life|self[- ]?harm|hurt myself|want to die|don'?t want to live", re.I)
HUMAN = re.compile(r"\b(human|real person|live agent|agent|representative|speak to (someone|a person)|talk to (someone|a person))\b", re.I)
SMALLTALK = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?( a lot| you)?|thank you|ok(ay)?|bye|goodbye)\b[\s!.?]*$", re.I)

REASON_TEXT = {
    "crisis": "urgent wellbeing concern",
    "user_requested": "you asked for a human",
    "negative_sentiment": "you seem frustrated",
    "low_confidence": "I'm not confident I can answer correctly",
}


class Bot:
    def __init__(self, faq_path=FAQ_PATH, model_path=MODEL_PATH):
        missing = [name for name, obj in (("transformers", pipeline), ("sentence-transformers", SentenceTransformer))
                   if obj is None]
        if missing:
            raise RuntimeError(
                f"Missing or broken dependency: {', '.join(missing)}. "
                "Install the project requirements with: pip install -r requirements.txt")
        # Load legacy model to extract intent class labels
        try:
            legacy_clf = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Could not load the intent model at {model_path}: {exc}. "
                               "Run `python src/train.py` to create it.") from exc
        self.intent_labels = list(legacy_clf.classes_)
        # Zero-shot intent classifier (no fine-tuning needed)
        try:
            self.clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            # Sentence-Transformer for FAQ semantic search
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the language models ({exc}). The first start needs internet access "
                "to download facebook/bart-large-mnli and all-MiniLM-L6-v2.") from exc
        # Retriever now uses the embedder
        self.retriever = Retriever(pd.read_csv(faq_path), embedder=self.embedder)
        self.sia = _load_sia()
        self.sia.lexicon.update(SENTIMENT_ADDITIONS)

    # Helper: make a short sentence look friendly (capitalize & ensure punctuation)
    def _friendly(self, text: str) -> str:
        txt = text.strip()
        if not txt.endswith('.'):  # add period if missing
            txt += '.'
        return txt[0].upper() + txt[1:]

    # Helper: add varied phrasing to answers based on intent
    def _polish(self, answer: str, intent: str) -> str:
        templates = {
            "academics": [
                "Sure thing! {}",
                "Here’s what you need to know: {}",
                "Let me explain: {}",
            ],
            "admissions": [
                "You’ll want to know that {}",
                "Regarding admissions: {}",
            ],
            "default": [
                "Here’s the info you asked for: {}",
                "Let me share what I found: {}",
                "{}",
            ],
        }
        pool = templates.get(intent, templates["default"])
        tmpl = random.choice(pool)
        return self._friendly(tmpl.format(answer.strip()))
    def predict_intent(self, text):
        """Predict intent using zero‑shot transformer.
        Returns top label and its confidence score.
        """
        result = self.clf(text, candidate_labels=self.intent_labels, multi_label=False)
        top_label = result["labels"][0]
        confidence = float(result["scores"][0])
        return top_label, confidence

    def _fused(self, text):
        # Zero‑shot classification probabilities
        result = self.clf(text, candidate_labels=self.intent_labels, multi_label=False)
        probs = dict(zip(result["labels"], result["scores"]))
        sims = self.retriever.scores(text)
        p = np.array([probs.get(c, 0.0) for c in self.retriever.cats])
        return p, sims, W_RETR * sims + (1 - W_RETR) * p

    def understand(self, text):
        """Fuse the intent classifier with FAQ retrieval.

        fused(topic) = W_RETR * retrieval_similarity + (1 - W_RETR) * P(category of topic)
        The best fused topic decides the intent, so a strong FAQ match can correct a weak classifier
        (and vice-versa). Returns intent, classifier probability of that intent, topic, retrieval
        similarity of that topic and the fused score used for the escalation decision.
        """
        p, sims, fused = self._fused(text)
        i = int(fused.argmax())
        return self.retriever.topics[i]["category"], float(p[i]), self.retriever.topics[i], float(sims[i]), float(fused[i])

    def rank(self, text, k=3):
        """Top-k FAQ topics by fused score (used for retrieval top-k evaluation)."""
        _, _, fused = self._fused(text)
        return [self.retriever.topics[i] for i in fused.argsort()[::-1][:k]]

    def respond(self, raw: str) -> dict:
        text = scrub(raw.strip())
        res = dict(user_query=text, intent="unknown", confidence=0.0, retrieval_score=0.0,
                   sentiment=0.0, fused=0.0, reason=None, escalated=False, reply="")

        if CRISIS.search(text):
            res.update(intent="mental_health", confidence=1.0, reason="crisis", escalated=True,
                       reply="I'm really sorry you're going through this, and I'm glad you told me. "
                             + CRISIS_RESOURCES + " I'm also flagging this to a human member of staff.")
            return res
        # ----- smalltalk handling -----
        if SMALLTALK.match(text):
            low = text.lower()
            reply = (
                "You're welcome! Anything else I can help with?"
                if "thank" in low
                else ("Goodbye! Come back any time."
                      if "bye" in low
                      else "Hello! I can help with admissions, fees, housing, courses, IT, wellbeing and more. What do you need?")
            )
            res.update(intent="smalltalk", confidence=1.0, reply=self._friendly(reply))
            return res

        # ----- core processing -----
        res["sentiment"] = float(self.sia.polarity_scores(text)["compound"])
        intent, conf, topic, score, fused = self.understand(text)
        res.update(intent=intent, confidence=conf, retrieval_score=score, fused=fused)
        answer = self.retriever.answer(topic, text)
        confident = fused >= FUSED_MIN

        # ----- escalation rules -----
        if HUMAN.search(text):
            res["reason"] = "user_requested"
        elif res["sentiment"] <= SENT_MIN:
            res["reason"] = "negative_sentiment"
        elif not confident:
            res["reason"] = "low_confidence"

        if res["reason"]:
            res["escalated"] = True
            lead = {
                "user_requested": "Of course – I'm handing you over to a human agent.",
                "negative_sentiment": "I'm sorry this has been frustrating. I'm escalating this to a human agent.",
                "low_confidence": "I'm not sure I can answer that correctly, so I'm passing it to a human agent.",
            }[res["reason"]]
            extra = (
                f" In the meantime, this may help: {answer}" if confident and res["reason"] != "user_requested" else ""
            )
            res["reply"] = self._friendly(lead + extra)
        else:
            # Non‑escalated, polished reply
            res["reply"] = self._polish(answer, intent)
        return res
