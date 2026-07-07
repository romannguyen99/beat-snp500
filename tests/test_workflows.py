from pathlib import Path

import yaml


def _load(name):
    return yaml.safe_load(Path(".github/workflows", name).read_text())


def test_ci_workflow_parses():
    wf = _load("ci.yml")
    assert "test" in wf["jobs"]


def test_daily_workflow_schedule_and_permissions():
    wf = _load("daily.yml")
    on = wf.get("on", wf.get(True))  # yaml parses bare `on:` as boolean True
    assert on["schedule"][0]["cron"] == "30 22 * * 1-5"
    assert wf["permissions"]["contents"] == "write"
    steps = wf["jobs"]["pipeline"]["steps"]
    steps_str = " ".join(str(s) for s in steps)
    assert "run_daily.py" in steps_str and "git push" in steps_str
    cache_steps = [s for s in steps if s.get("uses", "").startswith("actions/cache@")]
    assert len(cache_steps) == 1
    assert cache_steps[0]["with"]["path"] == "data/prices.parquet"


def test_streamlit_requirements():
    assert Path("requirements.txt").read_text().strip() == "."
