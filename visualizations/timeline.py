"""Gráficos de linha do tempo dos índices (Plotly)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


INDEX_LABELS = {
    'atencao': 'Atenção · β / (α + θ) frontal',
    'eng_cognitivo': 'Engajamento cognitivo · (β + γ) / α frontal',
    'eng_afetivo': 'Engajamento afetivo · (β + γ) / α frontal',
    'evocacao': 'Evocação de memórias · θ posterior',
    'aderencia': 'Aderência · (γ_F + γ_P) / θ_P',
    'faa': 'FAA · ln(α_AF8) − ln(α_AF7)',
    'arousal': 'Arousal · β / α total',
    'estresse': 'Estresse · (β/α) + (γ/θ)',
}

# Cores estáveis (Plotly D3 palette) para cada índice
INDEX_COLORS = {
    'atencao': '#1f77b4',
    'eng_cognitivo': '#ff7f0e',
    'eng_afetivo': '#2ca02c',
    'evocacao': '#d62728',
    'aderencia': '#9467bd',
    'faa': '#8c564b',
    'arousal': '#e377c2',
    'estresse': '#7f7f7f',
}

PRIMARY_INDICES = [
    'atencao', 'eng_cognitivo', 'eng_afetivo', 'faa', 'evocacao', 'aderencia',
]


def timeline_figure(
    indices_df: pd.DataFrame,
    index_name: str,
    show_median: bool = True,
    blink_times: Optional[Iterable[float]] = None,
) -> go.Figure:
    """Constrói uma figura Plotly com a série temporal de ``index_name``.

    Args:
        indices_df: DataFrame com ``t_window`` e a coluna ``index_name``.
        index_name: nome da coluna do índice.
        show_median: se True, traça uma linha horizontal tracejada na mediana.
        blink_times: lista de tempos (s) para marcar com linhas verticais.
    """
    if index_name not in indices_df.columns:
        raise ValueError(f"Índice ausente: {index_name}")

    series = indices_df[index_name]
    label = INDEX_LABELS.get(index_name, index_name)
    color = INDEX_COLORS.get(index_name, '#1f77b4')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices_df['t_window'],
        y=series,
        mode='lines',
        name=index_name,
        line=dict(color=color, width=2),
        hovertemplate='t = %{x:.1f}s<br>valor = %{y:.4f}<extra></extra>',
    ))

    if show_median and series.notna().any():
        median = float(series.median())
        fig.add_hline(
            y=median, line_dash='dash', line_color='#666',
            annotation_text=f'mediana = {median:.3f}',
            annotation_position='right',
        )

    if blink_times:
        for t in blink_times:
            fig.add_vline(x=float(t), line_width=0.4,
                          line_color='#d62728', opacity=0.25)

    fig.update_layout(
        title=label,
        xaxis_title='tempo de exposição (s)',
        yaxis_title='valor (suavizado · janela 5 s)',
        height=320,
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode='x unified',
        showlegend=False,
    )
    return fig


def cognitive_vs_affective_figure(
    indices_df: pd.DataFrame,
    show_median: bool = True,
    blink_times: Optional[Iterable[float]] = None,
) -> go.Figure:
    """Engajamento cognitivo + afetivo sobrepostos com área da diferença.

    Mostra eng_cognitivo (laranja) e eng_afetivo (verde) na mesma figura
    e preenche a área entre as curvas em verde claro — essa área é a
    magnitude da valência (|FAA|), que é o que diferencia os dois índices.
    """
    if not {'eng_cognitivo', 'eng_afetivo'}.issubset(indices_df.columns):
        raise ValueError("Colunas 'eng_cognitivo' e 'eng_afetivo' são obrigatórias.")

    t = indices_df['t_window']
    cog = indices_df['eng_cognitivo']
    afet = indices_df['eng_afetivo']

    fig = go.Figure()

    # eng_cognitivo (linha de baixo) — sem fill
    fig.add_trace(go.Scatter(
        x=t, y=cog, mode='lines', name='Eng. cognitivo · (β + γ) / α',
        line=dict(color=INDEX_COLORS['eng_cognitivo'], width=2.5),
        hovertemplate='t = %{x:.1f}s<br>eng_cog = %{y:.4f}<extra></extra>',
    ))

    # eng_afetivo (linha de cima) — preenche até a linha cog com tom esverdeado
    # representa a contribuição da valência |FAA|
    fig.add_trace(go.Scatter(
        x=t, y=afet, mode='lines', name='Eng. afetivo · cog + |FAA|',
        line=dict(color=INDEX_COLORS['eng_afetivo'], width=2.5),
        fill='tonexty', fillcolor='rgba(44, 160, 44, 0.18)',
        hovertemplate='t = %{x:.1f}s<br>eng_afet = %{y:.4f}<extra></extra>',
    ))

    if show_median:
        med_cog = float(cog.median())
        med_afet = float(afet.median())
        fig.add_hline(y=med_cog, line_dash='dot',
                      line_color=INDEX_COLORS['eng_cognitivo'], opacity=0.4,
                      annotation_text=f'mediana cog = {med_cog:.3f}',
                      annotation_position='right')
        fig.add_hline(y=med_afet, line_dash='dot',
                      line_color=INDEX_COLORS['eng_afetivo'], opacity=0.4,
                      annotation_text=f'mediana afet = {med_afet:.3f}',
                      annotation_position='right')

    if blink_times:
        for tb in blink_times:
            fig.add_vline(x=float(tb), line_width=0.4,
                          line_color='#d62728', opacity=0.20)

    fig.update_layout(
        title='Engajamento · cognitivo (laranja) vs afetivo (verde) '
              '— área = magnitude da valência (|FAA|)',
        xaxis_title='tempo de exposição (s)',
        yaxis_title='valor (suavizado · janela 5 s)',
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.20),
    )
    return fig


def aggregated_timeline_figure(
    agg: pd.DataFrame,
    label: str,
    color: str = '#1f77b4',
    band: str = 'ci',
) -> go.Figure:
    """Linha do tempo agregada com banda de incerteza.

    Args:
        agg: DataFrame com colunas ``t_window``, ``mean``, ``sem``,
            ``ci_lo``, ``ci_hi``, ``n``.
        label: rótulo do índice (vai no título e tooltip).
        color: cor hex para a linha principal.
        band: ``'ci'`` para IC 95% (default), ``'sem'`` para ± 1 SE,
            ``'none'`` para sem banda.
    """
    if agg.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sem dados após filtros",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color='#666'))
        fig.update_layout(height=320)
        return fig

    rgba = _hex_to_rgba(color, 0.18)
    t = agg['t_window']
    mean = agg['mean']

    fig = go.Figure()
    if band == 'ci' and 'ci_lo' in agg.columns:
        lo, hi = agg['ci_lo'], agg['ci_hi']
    elif band == 'sem':
        lo, hi = mean - agg['sem'], mean + agg['sem']
    else:
        lo, hi = None, None

    if lo is not None:
        fig.add_trace(go.Scatter(
            x=t, y=hi, mode='lines',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=t, y=lo, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor=rgba,
            showlegend=False, hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=t, y=mean, mode='lines',
        line=dict(color=color, width=2.5), name=label,
        customdata=agg[['n', 'sem']].values,
        hovertemplate=(
            't = %{x:.1f}s<br>M = %{y:.4f}<br>'
            'n = %{customdata[0]:.0f} · SE = %{customdata[1]:.4f}<extra></extra>'
        ),
    ))

    title_suffix = ' · banda IC 95%' if band == 'ci' else (' · banda ± SE' if band == 'sem' else '')
    fig.update_layout(
        title=f'{label}{title_suffix}',
        xaxis_title='tempo de exposição (s)',
        yaxis_title=f'{label} (média entre participantes)',
        height=380,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified',
        showlegend=False,
    )
    return fig


def comparison_timeline_figure(
    groups: dict[str, pd.DataFrame],
    label: str,
    band: str = 'ci',
) -> go.Figure:
    """Várias séries agregadas no mesmo gráfico, uma por grupo.

    ``groups`` é dict ``{nome_do_grupo: DataFrame agregado}``. Cada DataFrame
    deve ter o mesmo formato que ``aggregate_timeseries`` devolve.
    """
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    fig = go.Figure()
    visible_groups = 0
    for i, (name, agg) in enumerate(groups.items()):
        if agg.empty:
            continue
        color = palette[i % len(palette)]
        rgba = _hex_to_rgba(color, 0.13)
        t = agg['t_window']
        mean = agg['mean']

        if band == 'ci' and 'ci_lo' in agg.columns:
            lo, hi = agg['ci_lo'], agg['ci_hi']
        elif band == 'sem':
            lo, hi = mean - agg['sem'], mean + agg['sem']
        else:
            lo = hi = None

        if lo is not None:
            fig.add_trace(go.Scatter(
                x=t, y=hi, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=t, y=lo, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=rgba,
                showlegend=False, hoverinfo='skip',
            ))

        n_max = int(agg['n'].max()) if 'n' in agg.columns and not agg.empty else 0
        fig.add_trace(go.Scatter(
            x=t, y=mean, mode='lines',
            line=dict(color=color, width=2.5),
            name=f"{name} (n até {n_max})",
            customdata=agg[['n', 'sem']].values if 'sem' in agg.columns else None,
            hovertemplate=(
                f'<b>{name}</b><br>t = %{{x:.1f}}s · M = %{{y:.4f}}'
                + ('<br>n = %{customdata[0]:.0f} · SE = %{customdata[1]:.4f}'
                   if 'sem' in agg.columns else '')
                + '<extra></extra>'
            ),
        ))
        visible_groups += 1

    if visible_groups == 0:
        fig.add_annotation(text="Nenhum grupo com dados",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False)

    fig.update_layout(
        title=f'{label} · comparação entre grupos',
        xaxis_title='tempo de exposição (s)',
        yaxis_title=f'{label} (média do grupo)',
        height=440,
        margin=dict(l=60, r=20, t=50, b=80),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.18),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converte '#rrggbb' em 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def faa_timeline_figure(
    indices_df: pd.DataFrame,
    blink_times: Optional[Iterable[float]] = None,
) -> go.Figure:
    """Linha do tempo da FAA com sinal preservado.

    Mostra a assimetria frontal alfa em log natural:
        - Linha tracejada em y=0 como referência de simetria
        - Área positiva em azul (aproximação · α maior à direita / AF8)
        - Área negativa em vermelho (retração · α maior à esquerda / AF7)
    """
    if 'faa' not in indices_df.columns:
        raise ValueError("Coluna 'faa' ausente.")

    t = indices_df['t_window']
    faa = indices_df['faa'].astype(float)

    # Separar em dois segmentos com NaN nas regiões do outro sinal para
    # que `fill='tozeroy'` produza áreas distintas sem ligar os segmentos.
    pos = faa.where(faa >= 0, np.nan)
    neg = faa.where(faa < 0, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=pos, mode='lines',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.30)',
        name='Aproximação (FAA > 0)',
        hovertemplate='t = %{x:.1f}s<br>FAA = %{y:.3f} · aproximação<extra></extra>',
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=t, y=neg, mode='lines',
        line=dict(color='#d62728', width=2),
        fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.30)',
        name='Retração (FAA < 0)',
        hovertemplate='t = %{x:.1f}s<br>FAA = %{y:.3f} · retração<extra></extra>',
        connectgaps=False,
    ))

    # Linha horizontal em zero (referência de simetria)
    fig.add_hline(y=0, line_dash='dash', line_color='#444', line_width=1)

    if blink_times:
        for tb in blink_times:
            fig.add_vline(x=float(tb), line_width=0.4,
                          line_color='#d62728', opacity=0.20)

    fig.update_layout(
        title=INDEX_LABELS['faa'],
        xaxis_title='tempo de exposição (s)',
        yaxis_title='ln(α_AF8) − ln(α_AF7)',
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.20),
    )
    return fig
