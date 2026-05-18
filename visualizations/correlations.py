"""Matriz de correlação Spearman e heatmap."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def spearman_matrix(
    df: pd.DataFrame,
    cols_x: Sequence[str],
    cols_y: Sequence[str] | None = None,
    min_pairs: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula matriz ρ e matriz p de Spearman entre conjuntos de colunas.

    Pares com menos de ``min_pairs`` observações pareadas válidas devolvem NaN.
    """
    cols_y = list(cols_y) if cols_y else list(cols_x)
    cols_x = list(cols_x)

    rho = pd.DataFrame(index=cols_x, columns=cols_y, dtype=float)
    pval = pd.DataFrame(index=cols_x, columns=cols_y, dtype=float)

    for x in cols_x:
        for y in cols_y:
            if x not in df.columns or y not in df.columns:
                rho.loc[x, y] = np.nan
                pval.loc[x, y] = np.nan
                continue
            # Correlação consigo mesmo: trivial, e evita df[[x, x]] (duplica colunas)
            if x == y:
                s = df[x].dropna()
                if len(s) < min_pairs or s.nunique() < 2:
                    rho.loc[x, y] = np.nan
                    pval.loc[x, y] = np.nan
                else:
                    rho.loc[x, y] = 1.0
                    pval.loc[x, y] = 0.0
                continue
            pair = df[[x, y]].dropna()
            if len(pair) < min_pairs or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                rho.loc[x, y] = np.nan
                pval.loc[x, y] = np.nan
                continue
            res = stats.spearmanr(pair[x].to_numpy(), pair[y].to_numpy())
            # scipy >=1.11 expõe `.statistic`; versões antigas usam `.correlation`
            statistic = getattr(res, 'statistic', None)
            if statistic is None:
                statistic = res.correlation
            rho.loc[x, y] = float(statistic)
            pval.loc[x, y] = float(res.pvalue)
    return rho, pval


def correlation_heatmap(
    rho: pd.DataFrame,
    pval: pd.DataFrame,
    alpha: float = 0.05,
    title: str = 'Correlação de Spearman',
) -> go.Figure:
    """Heatmap com ``ρ`` no fundo e células marcadas com ``*`` quando p < alpha."""
    text = pd.DataFrame('', index=rho.index, columns=rho.columns)
    for x in rho.index:
        for y in rho.columns:
            r = rho.loc[x, y]
            p = pval.loc[x, y]
            if pd.isna(r):
                continue
            star = '*' if p < alpha else ''
            text.loc[x, y] = f'{r:.2f}{star}'

    fig = go.Figure(go.Heatmap(
        z=rho.values.astype(float),
        x=list(rho.columns),
        y=list(rho.index),
        colorscale='RdBu_r',
        zmid=0, zmin=-1, zmax=1,
        text=text.values,
        texttemplate='%{text}',
        textfont=dict(size=11),
        hovertemplate='%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>',
        colorbar=dict(title='ρ'),
    ))
    fig.update_layout(
        title=f'{title} (* p < {alpha})',
        height=max(360, 28 * len(rho.index) + 120),
        margin=dict(l=140, r=20, t=60, b=120),
        xaxis=dict(side='top', tickangle=-30),
        yaxis=dict(autorange='reversed'),
    )
    return fig
