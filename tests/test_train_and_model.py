import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib

from tests import support

ROOT = support.ROOT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TrainPipeline(unittest.TestCase):
    def test_train_runs_and_does_not_touch_existing_model(self):
        before = sha(ROOT / "models" / "intent.joblib")
        out = Path(tempfile.mkdtemp()) / "intent.joblib"
        r = subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--output", str(out)],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(before, sha(ROOT / "models" / "intent.joblib"))
        model = joblib.load(out)
        self.assertIn("housing", list(model.classes_))
        self.assertEqual(model.predict(["how do I reset my password"])[0], "it_services")

    def test_shipped_model_labels_match_faq_categories(self):
        model = joblib.load(ROOT / "models" / "intent.joblib")
        self.assertEqual(len(model.classes_), 15)


class ModelLazyLoading(unittest.TestCase):
    def test_import_does_not_load_flan(self):
        code = ("import sys; sys.path.insert(0, %r); import model; "
                "assert model._model is None and model._tokenizer is None; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; print('lazy-ok')" % str(ROOT))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("lazy-ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
