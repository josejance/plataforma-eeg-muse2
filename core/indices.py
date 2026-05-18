"""Cálculo dos 8 índices neurocognitivos a partir das bandas Mind Monitor.

Pipeline: log10 → linear → janelamento (5 s, overlap 50%) → fórmulas →
suavização (média móvel de 5 s sobre a série já janelada).

Todas as razões entre bandas operam em potência linear (após ``10^valor``).
A única exceção é a FAA (Frontal Alpha Asymmetry), que opera em log natural:
o Mind Monitor entrega log10, então convertemos via fator ``ln(10)``.

Referências:
- Pope, A. T., Bogart, E. H., & Bartolome, D. S. (1995). Biocybernetic system
  evaluates indices of operator engagement in automated task. Biological
  Psychology, 40(1-2), 187–195. (Índice de atenção β/(α+θ).)
- Arsalan, A., Majid, M., Butt, A. R., & Anwar, S. M. (2019). Classification
  of perceived mental stress using a commercially available EEG headband.
  IEEE Journal of Biomedical and Health Informatics, 23(6), 2257–2264.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from config import ALL_CHANNELS, BANDS, FRONTAL_CHANNELS, POSTERIOR_CHANNELS
from core.preprocessing import log_to_linear
from core.windowing import make_windows, moving_average, windowed_mean


def _band_lin(df: pd.DataFrame, band: str, channels: Sequence[str]) -> pd.Series:
    """Média da banda (escala linear) entre os canais informados."""
    cols = [f"{band}_{ch}_lin" for ch in channels]
    return df[cols].mean(axis=1)


def compute_attention(df_lin: pd.DataFrame) -> pd.Series:
    """β / (α + θ) frontal — Pope et al. (1995)."""
    beta = _band_lin(df_lin, 'Beta', FRONTAL_CHANNELS)
    alpha = _band_lin(df_lin, 'Alpha', FRONTAL_CHANNELS)
    theta = _band_lin(df_lin, 'Theta', FRONTAL_CHANNELS)
    return beta / (alpha + theta)


def compute_cognitive_engagement(df_lin: pd.DataFrame) -> pd.Series:
    """(β + γ) / α frontal."""
    beta = _band_lin(df_lin, 'Beta', FRONTAL_CHANNELS)
    gamma = _band_lin(df_lin, 'Gamma', FRONTAL_CHANNELS)
    alpha = _band_lin(df_lin, 'Alpha', FRONTAL_CHANNELS)
    return (beta + gamma) / alpha


def compute_affective_engagement(df_lin: pd.DataFrame) -> pd.Series:
    """|FAA| + (β + γ) / α frontal — combina intensidade e magnitude da valência.

    Diferentemente do engajamento cognitivo, soma a magnitude da assimetria
    frontal alfa (FAA em log natural, sempre não-negativa via valor absoluto)
    à razão (β + γ) / α. O SINAL da FAA — que indica aproximação (positivo)
    vs retração (negativo) — é reportado separadamente como o índice ``faa``.

    Requer que o DataFrame contenha tanto as colunas lineares (``*_lin``)
    quanto as colunas de Alfa em log10 (``Alpha_AF7``, ``Alpha_AF8``) — é o
    caso quando vindo de :func:`compute_all_indices` ou de
    :func:`core.preprocessing.log_to_linear`.
    """
    eng_cog = compute_cognitive_engagement(df_lin)
    faa_signed = compute_faa(df_lin)
    return faa_signed.abs() + eng_cog


def compute_faa(df_log: pd.DataFrame) -> pd.Series:
    """Frontal Alpha Asymmetry: ln(α_AF8) − ln(α_AF7).

    Mind Monitor entrega α em log10; conversão para ln via fator ``ln(10)``.
    """
    return (df_log['Alpha_AF8'] - df_log['Alpha_AF7']) * np.log(10.0)


def compute_memory_evocation(df_lin: pd.DataFrame) -> pd.Series:
    """θ posterior absoluto (média TP9, TP10) em escala linear."""
    return _band_lin(df_lin, 'Theta', POSTERIOR_CHANNELS)


def compute_adherence(df_lin: pd.DataFrame) -> pd.Series:
    """(γ_frontal + γ_posterior) / θ_posterior — proxy operacional."""
    g_front = _band_lin(df_lin, 'Gamma', FRONTAL_CHANNELS)
    g_post = _band_lin(df_lin, 'Gamma', POSTERIOR_CHANNELS)
    t_post = _band_lin(df_lin, 'Theta', POSTERIOR_CHANNELS)
    return (g_front + g_post) / t_post


def compute_arousal(df_lin: pd.DataFrame) -> pd.Series:
    """β_total / α_total (média sobre os 4 canais)."""
    beta = _band_lin(df_lin, 'Beta', ALL_CHANNELS)
    alpha = _band_lin(df_lin, 'Alpha', ALL_CHANNELS)
    return beta / alpha


def compute_stress(df_lin: pd.DataFrame) -> pd.Series:
    """(β/α) + (γ/θ), médias sobre os 4 canais — Arsalan et al. (2019)."""
    beta = _band_lin(df_lin, 'Beta', ALL_CHANNELS)
    alpha = _band_lin(df_lin, 'Alpha', ALL_CHANNELS)
    gamma = _band_lin(df_lin, 'Gamma', ALL_CHANNELS)
    theta = _band_lin(df_lin, 'Theta', ALL_CHANNELS)
    return (beta / alpha) + (gamma / theta)


# Ordem usada no DataFrame de saída de compute_all_indices()
INDEX_FUNCTIONS_LIN = {
    'atencao': compute_attention,
    'eng_cognitivo': compute_cognitive_engagement,
    'eng_afetivo': compute_affective_engagement,
    'evocacao': compute_memory_evocation,
    'aderencia': compute_adherence,
    'arousal': compute_arousal,
    'estresse': compute_stress,
}


def compute_all_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo: aplica conversão log→linear, janela, fórmulas e suavização.

    Args:
        df: DataFrame retornado por :func:`core.parser.load_csv` (deve conter as
            colunas de banda em log10, mais ``t_sec``).

    Returns:
        DataFrame com ``t_window`` (centro da janela, em segundos) e 8 colunas
        de índice: atencao, eng_cognitivo, eng_afetivo, evocacao, aderencia,
        arousal, estresse, faa.
    """
    if 't_sec' not in df.columns:
        raise ValueError("DataFrame precisa da coluna 't_sec'.")

    df_lin = log_to_linear(df)

    band_cols_lin = [f"{b}_{ch}_lin" for b in BANDS for ch in ALL_CHANNELS]
    alpha_log_cols = [f"Alpha_{ch}" for ch in ALL_CHANNELS]
    cols_to_window = [c for c in band_cols_lin + alpha_log_cols if c in df_lin.columns]

    windows = make_windows(df['t_sec'].to_numpy())
    out_cols = ['t_window'] + list(INDEX_FUNCTIONS_LIN.keys()) + ['faa']
    if not windows:
        return pd.DataFrame(columns=out_cols)

    win_df = windowed_mean(df_lin, cols_to_window, windows, t_col='t_sec')
    if win_df.empty:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame({'t_window': win_df['t_window']})
    for name, fn in INDEX_FUNCTIONS_LIN.items():
        out[name] = fn(win_df).to_numpy()
    out['faa'] = compute_faa(win_df).to_numpy()

    for col in out.columns:
        if col == 't_window':
            continue
        out[col] = moving_average(out[col])

    return out
