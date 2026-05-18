"""Leitura e validação de CSVs exportados pelo Mind Monitor."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from config import ALL_CHANNELS, BANDS


class MindMonitorParseError(Exception):
    """Erro ao ler/validar um CSV do Mind Monitor."""


REQUIRED_NON_BAND_COLUMNS = [
    'TimeStamp',
    *[f'HSI_{ch}' for ch in ALL_CHANNELS],
    'HeadBandOn',
]

REQUIRED_BAND_COLUMNS = [
    f'{band}_{ch}' for band in BANDS for ch in ALL_CHANNELS
]


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Lê um CSV do Mind Monitor, valida estrutura e devolve DataFrame.

    Adiciona a coluna ``t_sec`` com o tempo em segundos desde a primeira amostra.

    Args:
        path: Caminho do CSV.

    Returns:
        DataFrame ordenado por TimeStamp com ``t_sec`` adicionada.

    Raises:
        MindMonitorParseError: se o arquivo não existir, faltarem colunas
            obrigatórias, o TimeStamp não for parseável, ou não houver nenhuma
            linha com valores de banda (CSV "RAW-only" da configuração antiga).
    """
    path = Path(path)
    if not path.exists():
        raise MindMonitorParseError(f"Arquivo não encontrado: {path}")

    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback para acentos brasileiros mal codificados
        df = pd.read_csv(path, encoding='latin-1')
    except Exception as exc:
        raise MindMonitorParseError(f"Erro ao ler o CSV: {exc}") from exc

    missing = [c for c in REQUIRED_NON_BAND_COLUMNS + REQUIRED_BAND_COLUMNS
               if c not in df.columns]
    if missing:
        raise MindMonitorParseError(
            f"Colunas obrigatórias ausentes ({len(missing)}): {missing[:5]}..."
        )

    try:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    except Exception as exc:
        raise MindMonitorParseError(f"TimeStamp inválido: {exc}") from exc

    if df['TimeStamp'].isna().all():
        raise MindMonitorParseError("Todas as linhas de TimeStamp são inválidas.")

    df = df.sort_values('TimeStamp').reset_index(drop=True)
    df['t_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()

    valid_band_rows = df[REQUIRED_BAND_COLUMNS].notna().any(axis=1).sum()
    if valid_band_rows == 0:
        raise MindMonitorParseError(
            "Nenhuma linha com valores de banda — provável CSV de configuração "
            "antiga do Mind Monitor (apenas sinal RAW a 1 Hz)."
        )

    return df
