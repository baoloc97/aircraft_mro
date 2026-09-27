"""Every model feature has a plain-language reason, and families are routed to the right template."""
import pandas as pd
import pytest

from src.features import FEATURES
from src.reasons import EXACT, SUFFIX_RULES, reason


@pytest.fixture(scope="module")
def row():
    df = pd.read_parquet("data/processed/model_table.parquet", columns=None)
    return df[df.label == 1].iloc[0]


def test_every_feature_has_a_template():
    uncovered = [f for f in FEATURES if f not in EXACT and not any(f.endswith(s) for s, _ in SUFFIX_RULES)]
    assert not uncovered, uncovered


def test_reasons_render_without_fallback(row):
    for f in FEATURES:
        text = reason(f, row)
        assert text and not text.startswith(f"{f} ="), (f, text)


def test_trend_is_not_read_as_a_day_window(row):
    # regression: "related_faults_trend" used to match the "related_faults_<N>d" rule ("tren" days)
    assert "days" not in reason("related_faults_trend", row)
