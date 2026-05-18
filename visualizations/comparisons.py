"""Boxplots e scatter plots interativos para análise agregada."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def boxplot_by_group(
    df: pd.DataFrame,
    index_col: str,
    group_col: str,
    points: str = 'outliers',
) -> go.Figure:
    """Boxplot de ``index_col`` agrupado por ``group_col``.

    Args:
        points: ``'all'``, ``'outliers'`` ou ``False`` — comportamento do Plotly.
    """
    if index_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Colunas ausentes: {index_col}, {group_col}")

    data = df[[index_col, group_col]].dropna()
    fig = go.Figure()
    for group in sorted(data[group_col].dropna().unique()):
        sub = data[data[group_col] == group][index_col]
        fig.add_trace(go.Box(
            y=sub, name=str(group), boxpoints=points,
            jitter=0.4, pointpos=0,
            hovertemplate=f'{group_col} = {group}<br>valor = %{{y:.4f}}<extra></extra>',
        ))

    fig.update_layout(
        title=f'{index_col} por {group_col}',
        yaxis_title=index_col,
        xaxis_title=group_col,
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
    )
    return fig


def scatter_with_trendline(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
) -> go.Figure:
    """Scatter com regressão linear OLS (numpy.polyfit) sobreposta."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Colunas ausentes: {x_col}, {y_col}")

    cols = [x_col, y_col] + ([color_col] if color_col else [])
    data = df[cols].dropna()
    fig = go.Figure()

    if color_col is None or color_col not in data.columns:
        fig.add_trace(go.Scatter(
            x=data[x_col], y=data[y_col],
            mode='markers',
            marker=dict(size=8, color='#1f77b4', opacity=0.7),
            hovertemplate=f'{x_col} = %{{x:.3f}}<br>{y_col} = %{{y:.3f}}<extra></extra>',
            name='dados',
        ))
    else:
        for group in sorted(data[color_col].dropna().astype(str).unique()):
            sub = data[data[color_col].astype(str) == group]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode='markers', name=str(group),
                marker=dict(size=8, opacity=0.7),
                hovertemplate=(
                    f'{color_col} = {group}<br>'
                    f'{x_col} = %{{x:.3f}}<br>{y_col} = %{{y:.3f}}<extra></extra>'
                ),
            ))

    # Linha de tendência OLS sobre TODOS os pontos
    if len(data) >= 2 and data[x_col].nunique() >= 2:
        coef = np.polyfit(data[x_col], data[y_col], 1)
        xs = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        ys = coef[0] * xs + coef[1]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='black', dash='dash', width=1.5),
            name=f'tendência (a = {coef[0]:.3f}, b = {coef[1]:.3f})',
        ))

    fig.update_layout(
        title=f'{y_col} × {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=480,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig
