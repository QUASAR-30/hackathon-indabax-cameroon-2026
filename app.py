"""
HuggingFace Spaces entry point — PM2.5 Cameroun Dashboard
IndabaX Cameroun 2026

Usage:
  HF Spaces  : streamlit run app.py  (automatic via sdk: streamlit in README.md)
  Local test : streamlit run app.py  (from repo root)
  Local dev  : streamlit run notebooks/07_dashboard.py

Architecture:
  This file loads notebooks/07_dashboard.py via importlib so that:
    1. ROOT = Path(__file__).resolve().parent.parent  in 07_dashboard.py
       correctly resolves to this repo root (data/ and models/ are here).
    2. The module is cached in sys.modules between Streamlit reruns,
       preserving @st.cache_data and @st.cache_resource registrations.
"""

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Load dashboard module once per process ────────────────────────────────────
# Streamlit re-executes app.py on every widget interaction, but sys.modules
# persists for the lifetime of the process → @st.cache_data stays registered.
if "pm25_dashboard" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "pm25_dashboard",
        ROOT / "notebooks" / "07_dashboard.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["pm25_dashboard"] = _mod
    _spec.loader.exec_module(_mod)

# ── Run ───────────────────────────────────────────────────────────────────────
sys.modules["pm25_dashboard"].main()
