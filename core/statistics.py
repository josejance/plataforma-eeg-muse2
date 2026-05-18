"""Estatísticas descritivas dos índices por sessão.

Note:
    "% acima da mediana intrassujeito" é ~50% por construção quando calculado
    contra a mediana da própria série; o valor passa a ser informativo apenas
    quando comparado a uma referência externa (ex: mediana baseline).
"""
from __future__ import annotations

import pandas as pd

from db.queries import INDEX_COLUMNS


SUMMARY_COLUMNS = ['índice', 'mediana', 'média', 'std', 'min', 'max', '% acima mediana']


def summarize_indices(indices_df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por índice: mediana, média, desvio, extremos, % acima da mediana."""
    rows = []
    for col in INDEX_COLUMNS:
        if col not in indices_df.columns:
            continue
        s = indices_df[col].dropna()
        if s.empty:
            rows.append({k: None for k in SUMMARY_COLUMNS} | {'índice': col})
            continue
        median = float(s.median())
        rows.append({
            'índice': col,
            'mediana': median,
            'média': float(s.mean()),
            'std': float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            'min': float(s.min()),
            'max': float(s.max()),
            '% acima mediana': float((s > median).mean() * 100.0),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def extract_blink_times(df_raw: pd.DataFrame) -> list[float]:
    """Devolve os tempos (em segundos) em que ocorreu blink na coluna Elements."""
    if 'Elements' not in df_raw.columns or 't_sec' not in df_raw.columns:
        return []
    mask = df_raw['Elements'].astype(str).str.contains('blink', case=False, na=False)
    return df_raw.loc[mask, 't_sec'].astype(float).tolist()
