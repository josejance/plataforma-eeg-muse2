"""Testes de estatísticas e extração de eventos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.statistics import SUMMARY_COLUMNS, extract_blink_times, summarize_indices
from db.queries import INDEX_COLUMNS


def _make_indices(n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        't_window': np.arange(n) * 2.5,
        **{c: rng.normal(loc=1.0, scale=0.3, size=n) for c in INDEX_COLUMNS},
    })


def test_summarize_returns_one_row_per_index() -> None:
    df = _make_indices()
    out = summarize_indices(df)
    assert list(out.columns) == SUMMARY_COLUMNS
    assert set(out['índice']) == set(INDEX_COLUMNS)


def test_summarize_values_match_pandas() -> None:
    df = _make_indices(n=50)
    out = summarize_indices(df).set_index('índice')
    for col in INDEX_COLUMNS:
        s = df[col]
        assert out.loc[col, 'mediana'] == pytest.approx(s.median())
        assert out.loc[col, 'média'] == pytest.approx(s.mean())
        assert out.loc[col, 'std'] == pytest.approx(s.std(ddof=1))
        assert out.loc[col, 'min'] == pytest.approx(s.min())
        assert out.loc[col, 'max'] == pytest.approx(s.max())


def test_summarize_pct_above_median_near_50() -> None:
    """Por construção, % acima da mediana intrassujeito é ~50%."""
    df = _make_indices(n=100)
    out = summarize_indices(df)
    pct = out['% acima mediana']
    assert (pct.between(40, 60)).all()


def test_summarize_handles_missing_column() -> None:
    df = pd.DataFrame({'t_window': [0.0], 'atencao': [1.0]})
    out = summarize_indices(df)
    # Só `atencao` está presente entre os índices
    assert list(out['índice']) == ['atencao']


def test_summarize_empty_series() -> None:
    df = pd.DataFrame({'t_window': [], **{c: [] for c in INDEX_COLUMNS}})
    out = summarize_indices(df)
    assert len(out) == len(INDEX_COLUMNS)
    assert out['mediana'].isna().all()


def test_extract_blink_times() -> None:
    df = pd.DataFrame({
        't_sec': [0.0, 1.0, 2.0, 3.0, 4.0],
        'Elements': ['', '/muse/elements/blink', '', '/muse/elements/blink', 'jaw_clench'],
    })
    times = extract_blink_times(df)
    assert times == [1.0, 3.0]


def test_extract_blink_times_no_column() -> None:
    df = pd.DataFrame({'t_sec': [0.0, 1.0]})
    assert extract_blink_times(df) == []
