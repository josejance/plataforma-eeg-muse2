"""Testes da matriz de correlação Spearman e do heatmap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy import stats

from visualizations.correlations import correlation_heatmap, spearman_matrix


def test_spearman_matrix_self_correlations() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'a': rng.normal(size=30),
        'b': rng.normal(size=30),
        'c': rng.normal(size=30),
    })
    rho, p = spearman_matrix(df, cols_x=['a', 'b', 'c'])
    # Diagonal = 1.0
    for col in ['a', 'b', 'c']:
        assert rho.loc[col, col] == pytest.approx(1.0)
        assert p.loc[col, col] == pytest.approx(0.0, abs=1e-9)
    # Simetria
    assert rho.loc['a', 'b'] == pytest.approx(rho.loc['b', 'a'])


def test_spearman_matrix_known_correlation() -> None:
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y_pos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # ρ = 1
        'y_neg': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],   # ρ = -1
    })
    rho, _ = spearman_matrix(df, cols_x=['x'], cols_y=['y_pos', 'y_neg'])
    assert rho.loc['x', 'y_pos'] == pytest.approx(1.0)
    assert rho.loc['x', 'y_neg'] == pytest.approx(-1.0)


def test_spearman_matrix_too_few_observations() -> None:
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 4.0]})
    rho, p = spearman_matrix(df, cols_x=['x'], cols_y=['y'], min_pairs=3)
    assert pd.isna(rho.loc['x', 'y'])
    assert pd.isna(p.loc['x', 'y'])


def test_spearman_matrix_constant_column() -> None:
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'k': [7, 7, 7, 7, 7]})
    rho, _ = spearman_matrix(df, cols_x=['x'], cols_y=['k'])
    assert pd.isna(rho.loc['x', 'k'])


def test_spearman_matrix_missing_column() -> None:
    df = pd.DataFrame({'x': [1, 2, 3]})
    rho, p = spearman_matrix(df, cols_x=['x'], cols_y=['inexistente'])
    assert pd.isna(rho.loc['x', 'inexistente'])


def test_correlation_heatmap_returns_figure() -> None:
    rho = pd.DataFrame({'b': [0.5]}, index=['a'])
    p = pd.DataFrame({'b': [0.01]}, index=['a'])
    fig = correlation_heatmap(rho, p, alpha=0.05)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # 1 célula significativa → asterisco no texto
    assert '0.50*' in fig.data[0].text.flatten().tolist()


def test_correlation_heatmap_no_significance() -> None:
    rho = pd.DataFrame({'b': [0.1]}, index=['a'])
    p = pd.DataFrame({'b': [0.6]}, index=['a'])
    fig = correlation_heatmap(rho, p)
    assert '0.10' in fig.data[0].text.flatten().tolist()
    assert '0.10*' not in fig.data[0].text.flatten().tolist()


def test_correlation_heatmap_handles_nan() -> None:
    rho = pd.DataFrame({'b': [np.nan]}, index=['a'])
    p = pd.DataFrame({'b': [np.nan]}, index=['a'])
    fig = correlation_heatmap(rho, p)
    # NaN cell → texto vazio
    assert fig.data[0].text.flatten().tolist() == ['']
