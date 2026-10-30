"""Alternative entry point:  streamlit run frontend/app.py

This file used to be a near-copy of src/app.py (with an unfinished agent view and an eager FLAN-T5 import).
To avoid maintaining two UIs it now simply runs the single implementation in src/app.py, so both launch
commands show the same UniAssist AI application. The experimental FLAN-T5 option from the old file lives in
Privacy & Settings > Experimental.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "src" / "app.py"), run_name="__main__")
