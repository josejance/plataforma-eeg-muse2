"""Filtros de qualidade e conversão log10 → linear das bandas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    ALL_CHANNELS,
    BANDS,
    BLINKS_PER_MIN_WARN,
    HEADBAND_OFF_MAX_RATIO,
    HSI_BAD_THRESHOLD,
    HSI_MEAN_WARN,
    MIN_SESSION_DURATION_SEC,
)


@dataclass
class QualityReport:
    """Relatório de qualidade do sinal de uma sessão."""
    n_samples_total: int
    n_samples_valid: int
    pct_discarded: float
    hsi_mean_per_channel: Dict[str, float]
    blink_rate_per_min: float
    headband_off_ratio: float
    duration_sec: float
    alerts: List[str] = field(default_factory=list)


def quality_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
    """Descarta amostras com sinal ruim e gera relatório.

    Critério de descarte: linhas em que pelo menos 3 dos 4 canais têm
    ``HSI >= HSI_BAD_THRESHOLD`` (valor 4 = sem contato no Mind Monitor).

    Args:
        df: DataFrame retornado por :func:`core.parser.load_csv`, contendo as
            colunas HSI_*, HeadBandOn, t_sec e (opcionalmente) Elements.

    Returns:
        Tupla ``(df_filtrado, relatório)``. O relatório usa o DataFrame
        original (pré-filtro) para HSI médio, taxa de piscadas e duração.
    """
    n_total = len(df)
    hsi_cols = [f"HSI_{ch}" for ch in ALL_CHANNELS]
    hsi_values = df[hsi_cols].to_numpy()
    bad_per_row = (hsi_values >= HSI_BAD_THRESHOLD).sum(axis=1)
    keep_mask = bad_per_row < 3
    df_valid = df.loc[keep_mask].copy().reset_index(drop=True)

    n_valid = len(df_valid)
    pct_disc = 100.0 * (n_total - n_valid) / n_total if n_total > 0 else 0.0

    hsi_means: Dict[str, float] = {
        ch: float(df[f"HSI_{ch}"].mean()) for ch in ALL_CHANNELS
    }
    blink_rate = _estimate_blink_rate(df)
    headband_off_ratio = (
        1.0 - float(df['HeadBandOn'].mean()) if 'HeadBandOn' in df.columns else 0.0
    )
    duration_sec = (
        float(df['t_sec'].iloc[-1]) if 't_sec' in df.columns and n_total > 0 else 0.0
    )

    alerts: List[str] = []
    for ch, m in hsi_means.items():
        if m > HSI_MEAN_WARN:
            alerts.append(f"HSI médio alto em {ch}: {m:.2f} (> {HSI_MEAN_WARN})")
    if blink_rate > BLINKS_PER_MIN_WARN:
        alerts.append(
            f"Taxa de piscadas alta: {blink_rate:.1f}/min (> {BLINKS_PER_MIN_WARN})"
        )
    if headband_off_ratio > HEADBAND_OFF_MAX_RATIO:
        alerts.append(
            f"Headband fora da cabeça em {100*headband_off_ratio:.1f}% das amostras "
            f"(limiar: {100*HEADBAND_OFF_MAX_RATIO:.0f}%)"
        )
    if duration_sec < MIN_SESSION_DURATION_SEC:
        alerts.append(
            f"Sessão muito curta: {duration_sec:.1f}s (mínimo: {MIN_SESSION_DURATION_SEC}s)"
        )

    return df_valid, QualityReport(
        n_samples_total=n_total,
        n_samples_valid=n_valid,
        pct_discarded=pct_disc,
        hsi_mean_per_channel=hsi_means,
        blink_rate_per_min=blink_rate,
        headband_off_ratio=float(headband_off_ratio),
        duration_sec=duration_sec,
        alerts=alerts,
    )


def _estimate_blink_rate(df: pd.DataFrame) -> float:
    """Conta ocorrências de 'blink' na coluna Elements e divide por duração."""
    if 'Elements' not in df.columns or len(df) == 0:
        return 0.0
    if 't_sec' not in df.columns:
        return 0.0
    duration_min = (df['t_sec'].iloc[-1] - df['t_sec'].iloc[0]) / 60.0
    if duration_min <= 0:
        return 0.0
    blinks = df['Elements'].astype(str).str.contains('blink', case=False, na=False).sum()
    return float(blinks) / duration_min


def quality_score(report: QualityReport) -> float:
    """0.0 = sinal péssimo, 1.0 = sinal perfeito.

    Baseia-se no HSI médio dos 4 canais (1 = ótimo, 4 = sem contato).
    """
    if not report.hsi_mean_per_channel:
        return 0.0
    mean_hsi = sum(report.hsi_mean_per_channel.values()) / len(report.hsi_mean_per_channel)
    return max(0.0, min(1.0, 1.0 - (mean_hsi - 1.0) / 3.0))


def log_to_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as 20 colunas de banda (log10 de potência) para escala linear.

    Cria colunas novas com sufixo ``_lin`` preservando os originais. A conversão
    precisa acontecer ANTES de qualquer razão entre bandas, porque a soma de
    valores em escala log não corresponde à soma das potências.
    """
    out = df.copy()
    for band in BANDS:
        for ch in ALL_CHANNELS:
            col = f"{band}_{ch}"
            if col in out.columns:
                out[f"{col}_lin"] = np.power(10.0, out[col].astype(float))
    return out
