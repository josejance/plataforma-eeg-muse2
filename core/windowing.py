"""Janelamento temporal e suavização das séries de índices."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import SMOOTHING_WINDOW, WINDOW_OVERLAP, WINDOW_SIZE


def make_windows(
    t_sec: Sequence[float],
    window_size: float = WINDOW_SIZE,
    overlap: float = WINDOW_OVERLAP,
) -> List[Tuple[float, float]]:
    """Gera janelas (t_start, t_end) com sobreposição.

    Passo entre janelas: ``window_size * (1 - overlap)``. Para overlap=0.5
    e window_size=5s, o passo é 2.5s.
    """
    t_sec = np.asarray(t_sec, dtype=float)
    if t_sec.size == 0:
        return []
    t_min, t_max = float(t_sec[0]), float(t_sec[-1])
    if t_max - t_min < window_size:
        return []
    step = window_size * (1.0 - overlap)
    # +1e-9 evita perder a última janela por arredondamento
    starts = np.arange(t_min, t_max - window_size + 1e-9, step)
    return [(float(s), float(s + window_size)) for s in starts]


def windowed_mean(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[Tuple[float, float]],
    t_col: str = 't_sec',
) -> pd.DataFrame:
    """Calcula a média das colunas dentro de cada janela.

    Returns:
        DataFrame com ``t_window`` (centro da janela) e uma coluna por entrada
        de ``cols``. Janelas vazias são puladas.
    """
    if t_col not in df.columns:
        raise ValueError(f"Coluna de tempo ausente: {t_col}")
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=['t_window'])

    t_arr = df[t_col].to_numpy()
    rows = []
    for t_start, t_end in windows:
        mask = (t_arr >= t_start) & (t_arr < t_end)
        if not mask.any():
            continue
        sub = df.loc[mask, cols]
        row = {'t_window': (t_start + t_end) / 2.0}
        for c in cols:
            row[c] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def moving_average(
    series: pd.Series,
    window_sec: float = SMOOTHING_WINDOW,
    step_sec: float = WINDOW_SIZE * (1.0 - WINDOW_OVERLAP),
) -> pd.Series:
    """Suaviza a série por média móvel centrada.

    O tamanho da janela em amostras é ``round(window_sec / step_sec)``.
    Para window=5s e step=2.5s, isso dá 2 amostras.
    """
    n = max(1, int(round(window_sec / step_sec)))
    return series.rolling(window=n, min_periods=1, center=True).mean()
