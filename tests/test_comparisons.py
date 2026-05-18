"""Testes dos boxplots e scatter plots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from visualizations.comparisons import boxplot_by_group, scatter_with_trendline


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'gender': ['feminino'] * 15 + ['masculino'] * 15,
        'atencao_mean': rng.normal(1.0, 0.3, 30),
        'trait_anger': rng.uniform(0, 10, 30),
    })


def test_boxplot_one_trace_per_group() -> None:
    df = _make_df()
    fig = boxplot_by_group(df, 'atencao_mean', 'gender')
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {'feminino', 'masculino'}


def test_boxplot_missing_column_raises() -> None:
    df = _make_df()
    with pytest.raises(ValueError):
        boxplot_by_group(df, 'inexistente', 'gender')


def test_scatter_has_points_and_trendline() -> None:
    df = _make_df()
    fig = scatter_with_trendline(df, 'trait_anger', 'atencao_mean')
    assert isinstance(fig, go.Figure)
    # 1 trace de pontos + 1 trace de tendência
    assert len(fig.data) == 2
    assert fig.data[0].mode == 'markers'
    assert fig.data[1].mode == 'lines'


def test_scatter_color_grouping() -> None:
    df = _make_df()
    fig = scatter_with_trendline(df, 'trait_anger', 'atencao_mean', color_col='gender')
    # 2 grupos de pontos + 1 linha de tendência global
    assert len(fig.data) == 3
    assert {t.name for t in fig.data[:2]} == {'feminino', 'masculino'}


def test_scatter_no_trendline_with_single_x() -> None:
    """Sem variação em X → sem linha de tendência."""
    df = pd.DataFrame({'x': [5.0, 5.0, 5.0], 'y': [1.0, 2.0, 3.0]})
    fig = scatter_with_trendline(df, 'x', 'y')
    assert len(fig.data) == 1
