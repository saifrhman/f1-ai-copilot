"""F1 AI Copilot web UI.

Run from the project folder with ``streamlit run ui/streamlit_app.py`` while the API
is running (or start both with ``python scripts/run_app.py``). The UI talks to the
API over HTTP only; ``F1_API_URL`` selects the API (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # `streamlit run` only puts ui/ on sys.path
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from ui.api_client import get_client  # noqa: E402
from ui.components import NAV_SECTIONS, app_page, md_text, render_api_sidebar  # noqa: E402

st.set_page_config(page_title="F1 AI Copilot", page_icon=":material/sports_motorsports:", layout="wide")
navigation = st.navigation({section: [app_page(name) for name in names] for section, names in NAV_SECTIONS.items()})

try:
    get_client()
except ValueError as exc:  # malformed F1_API_URL
    st.error(f"{md_text(exc)}. Set `F1_API_URL` to the API address (e.g. `http://127.0.0.1:8000`) and restart the UI.")
    st.stop()

render_api_sidebar()
navigation.run()
