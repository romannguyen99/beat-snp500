import importlib.util
from pathlib import Path


def test_app_module_loads_without_data():
    spec = importlib.util.spec_from_file_location(
        "streamlit_app", Path("app/streamlit_app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise even with no data files
