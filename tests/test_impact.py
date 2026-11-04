import pandas as pd
import pytest

from impact import estimate_impact, band_for


def make_table():
    return pd.DataFrame([
        {"band_low": 0, "band_high": 100, "hospitals": 10,
         "error_alone": 0.05, "error_shared": 0.02,
         "share_better_off": 0.8, "share_worse_off": 0.0},
        {"band_low": 100, "band_high": 10_000, "hospitals": 20,
         "error_alone": 0.03, "error_shared": 0.02,
         "share_better_off": 0.6, "share_worse_off": 0.1},
    ])


def test_picks_the_matching_size_band():
    row = band_for(make_table(), 50)
    assert row.error_alone == 0.05
    assert band_for(make_table(), 100).error_alone == 0.03


def test_scales_errors_to_yearly_patients():
    result = estimate_impact(make_table(), 50, 1000)
    assert result["wrong_alone"] == pytest.approx(50)
    assert result["wrong_shared"] == pytest.approx(20)
    assert result["fewer_wrong"] == pytest.approx(30)


def test_rejects_impossible_numbers():
    with pytest.raises(ValueError):
        estimate_impact(make_table(), 0, 1000)
    with pytest.raises(ValueError):
        estimate_impact(make_table(), 50, -5)
