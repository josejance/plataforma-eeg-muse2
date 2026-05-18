"""Testes das figuras Plotly (estrutura, não aparência)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from db.queries import INDEX_COLUMNS
from visualizations.timeline import (
    INDEX_LABELS,
    PRIMARY_INDICES,
    aggregated_timeline_figure,
    cognitive_vs_affective_figure,
    comparison_timeline_figure,
    faa_timeline_figure,
    timeline_figure,
)


def _make_indices(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        't_window': np.arange(n) * 2.5,
        **{c: rng.normal(1.0, 0.3, n) for c in INDEX_COLUMNS},
    })


def test_timeline_figure_returns_figure() -> None:
    df = _make_indices()
    fig = timeline_figure(df, 'atencao')
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == 'lines'
    assert len(fig.data[0].x) == len(df)
    assert 'Atenção' in fig.layout.title.text


def test_timeline_figure_median_line() -> None:
    df = _make_indices()
    fig = timeline_figure(df, 'atencao', show_median=True)
    # shapes contém hline e (se houver) vlines
    shapes = fig.layout.shapes
    assert any(s.type == 'line' for s in shapes), 'falta linha da mediana'


def test_timeline_figure_no_median_when_disabled() -> None:
    df = _make_indices()
    fig = timeline_figure(df, 'atencao', show_median=False)
    assert len(fig.layout.shapes) == 0


def test_timeline_figure_blink_markers() -> None:
    df = _make_indices()
    fig = timeline_figure(df, 'atencao', show_median=False,
                          blink_times=[5.0, 12.5, 30.0])
    # 3 vlines
    assert len(fig.layout.shapes) == 3


def test_timeline_figure_invalid_index() -> None:
    df = _make_indices()
    with pytest.raises(ValueError, match='ausente'):
        timeline_figure(df, 'indice_inexistente')


def test_primary_indices_have_labels() -> None:
    for idx in PRIMARY_INDICES:
        assert idx in INDEX_LABELS, f'falta label para {idx}'


def test_all_db_indices_have_labels() -> None:
    """Todos os índices salvos no banco devem ter rótulo legível."""
    for idx in INDEX_COLUMNS:
        assert idx in INDEX_LABELS, f'falta label para {idx}'


def test_primary_indices_includes_faa() -> None:
    """FAA precisa estar no rol de gráficos primários (separado do eng_afetivo)."""
    assert 'faa' in PRIMARY_INDICES
    assert 'eng_afetivo' in PRIMARY_INDICES


# ---------- FAA-specific renderer ----------
def test_faa_figure_returns_figure() -> None:
    df = pd.DataFrame({'t_window': [0, 2.5, 5, 7.5], 'faa': [0.5, -0.3, 0.1, -0.2]})
    fig = faa_timeline_figure(df)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)


def test_faa_figure_has_zero_reference_line() -> None:
    df = pd.DataFrame({'t_window': [0, 2.5], 'faa': [0.5, -0.3]})
    fig = faa_timeline_figure(df)
    shapes = fig.layout.shapes
    # Pelo menos uma shape horizontal em y=0
    assert any(s.type == 'line' and s.y0 == 0 and s.y1 == 0 for s in shapes)


def test_faa_figure_has_two_traces_for_pos_and_neg() -> None:
    df = pd.DataFrame({'t_window': [0, 2.5, 5], 'faa': [0.5, -0.3, 0.1]})
    fig = faa_timeline_figure(df)
    # Dois traces: aproximação (positivo) e retração (negativo)
    assert len(fig.data) == 2
    names = [t.name for t in fig.data]
    assert any('proximação' in n for n in names)
    assert any('etração' in n for n in names)


def test_faa_figure_blink_markers() -> None:
    df = pd.DataFrame({'t_window': [0, 2.5, 5], 'faa': [0.5, -0.3, 0.1]})
    fig = faa_timeline_figure(df, blink_times=[1.0, 3.5])
    n_vlines = sum(
        1 for s in fig.layout.shapes
        if s.type == 'line' and s.x0 == s.x1 and s.x0 in (1.0, 3.5)
    )
    assert n_vlines == 2


def test_faa_figure_missing_column_raises() -> None:
    df = pd.DataFrame({'t_window': [0, 1, 2], 'atencao': [1, 2, 3]})
    with pytest.raises(ValueError, match='faa'):
        faa_timeline_figure(df)


# ---------- Cognitive vs Affective combined figure ----------
def test_cog_vs_afet_figure_has_two_traces() -> None:
    df = _make_indices()
    fig = cognitive_vs_affective_figure(df, show_median=False)
    assert len(fig.data) == 2
    names = [t.name for t in fig.data]
    assert any('cognitivo' in n for n in names)
    assert any('afetivo' in n for n in names)


def test_cog_vs_afet_figure_y_values_differ() -> None:
    """As duas curvas no mesmo plot precisam ter Y diferentes para cada x."""
    df = _make_indices()
    # Garante divergência (afet = cog + |faa|, faa random pode dar simetria por acaso)
    df['eng_afetivo'] = df['eng_cognitivo'] + 0.5
    fig = cognitive_vs_affective_figure(df, show_median=False)
    y_cog = list(fig.data[0].y)
    y_afet = list(fig.data[1].y)
    diff = sum(1 for a, b in zip(y_cog, y_afet) if abs(a - b) > 1e-9)
    assert diff == len(y_cog), "Curvas deveriam divergir em todos os pontos"


def test_cog_vs_afet_figure_has_fill_between() -> None:
    df = _make_indices()
    fig = cognitive_vs_affective_figure(df, show_median=False)
    # A linha afetivo (segunda trace) deve ter fill='tonexty' para preencher
    # a área até a linha cognitiva
    assert fig.data[1].fill == 'tonexty'


def test_cog_vs_afet_figure_missing_columns_raises() -> None:
    df = pd.DataFrame({'t_window': [0, 1], 'eng_cognitivo': [1, 2]})
    with pytest.raises(ValueError, match='eng_afetivo'):
        cognitive_vs_affective_figure(df)


# ---------- Aggregated timeline (single group) ----------
def _make_agg_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        't_window': np.arange(n) * 2.5,
        'mean': rng.normal(1.0, 0.2, n),
        'sd': np.full(n, 0.3),
        'sem': np.full(n, 0.1),
        'n': np.full(n, 20),
        'ci_lo': rng.normal(0.8, 0.2, n),
        'ci_hi': rng.normal(1.2, 0.2, n),
    })


def test_aggregated_timeline_figure_with_ci() -> None:
    agg = _make_agg_df()
    fig = aggregated_timeline_figure(agg, label='atencao', band='ci')
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    # 2 traces auxiliares (banda) + 1 trace principal = 3
    assert len(fig.data) == 3


def test_aggregated_timeline_figure_no_band() -> None:
    agg = _make_agg_df()
    fig = aggregated_timeline_figure(agg, label='atencao', band='none')
    # só linha principal
    assert len(fig.data) == 1


def test_aggregated_timeline_figure_empty() -> None:
    fig = aggregated_timeline_figure(pd.DataFrame(), label='x')
    # sem traces, só annotation
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


def test_comparison_timeline_figure_multiple_groups() -> None:
    groups = {
        'esquerda': _make_agg_df(),
        'direita': _make_agg_df(),
    }
    fig = comparison_timeline_figure(groups, label='atencao')
    # Para cada grupo: 2 traces de banda + 1 de média = 3 traces
    # 2 grupos × 3 = 6
    assert len(fig.data) == 6
    names_with_n = [t.name for t in fig.data if t.name]
    assert any('esquerda' in n for n in names_with_n)
    assert any('direita' in n for n in names_with_n)


def test_comparison_timeline_figure_empty_groups() -> None:
    fig = comparison_timeline_figure({}, label='atencao')
    assert len(fig.data) == 0
