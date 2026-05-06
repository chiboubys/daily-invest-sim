from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.streamlit_smoke
def test_streamlit_app_smoke() -> None:
    at = AppTest.from_file("streamlit_app/main.py")
    at.run(timeout=20)
    assert not at.exception
