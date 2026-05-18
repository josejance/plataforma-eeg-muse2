"""Gera um CSV sintético em formato Mind Monitor para uso nos testes.

Os valores são escolhidos para serem fisiológicamente plausíveis e para que
cálculos manuais sejam reproduzíveis: log10(potência) entre -1 e 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def make_sample_df(
    n_samples: int = 320,
    rate_hz: float = 10.0,
    start: str = '2025-01-01T12:00:00',
    hsi_value: int = 1,
    headband_on: int = 1,
    inject_bad_rows: int = 0,
    bands_log10: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Constrói um DataFrame com a estrutura do Mind Monitor.

    Args:
        n_samples: número de linhas (a 10 Hz, 320 linhas = 32 s).
        rate_hz: taxa de amostragem das bandas.
        hsi_value: valor de HSI uniforme para todos os canais (1 = ótimo).
        headband_on: 0 ou 1.
        inject_bad_rows: número de linhas no final em que HSI vira 4 (sem contato)
            em 3+ canais — usado para testar o filtro de qualidade.
        bands_log10: dict opcional ``{band: log10}`` para definir valores
            uniformes; o ruído é desativado.
        seed: semente para reprodutibilidade do ruído.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_samples, freq=f"{1000/rate_hz:.3f}ms")

    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    data: dict = {'TimeStamp': timestamps.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]}

    # Valores médios típicos (log10 de potência) para Muse 2 em sinal limpo
    default_log = {'Delta': 0.5, 'Theta': 0.3, 'Alpha': 0.6, 'Beta': 0.2, 'Gamma': -0.2}
    targets = bands_log10 or default_log

    for band in bands:
        for ch in channels:
            base = targets.get(band, default_log[band])
            if bands_log10 is None:
                noise = rng.normal(0, 0.05, n_samples)
            else:
                noise = np.zeros(n_samples)
            data[f"{band}_{ch}"] = base + noise

    for ch in channels:
        data[f"HSI_{ch}"] = np.full(n_samples, hsi_value, dtype=float)

    data['HeadBandOn'] = np.full(n_samples, headband_on, dtype=int)
    data['Elements'] = np.full(n_samples, '', dtype=object)

    # Colunas RAW (mantidas em branco — taxa diferente das bandas)
    for ch in channels:
        data[f'RAW_{ch}'] = np.nan

    df = pd.DataFrame(data)

    if inject_bad_rows > 0:
        bad_start = max(0, n_samples - inject_bad_rows)
        for ch in channels[:3]:  # 3 canais ruins = descartado pelo filtro
            df.loc[bad_start:, f"HSI_{ch}"] = 4.0

    return df


def write_sample_csv(path: Path, **kwargs) -> Path:
    """Gera o DataFrame e grava em CSV."""
    df = make_sample_df(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
