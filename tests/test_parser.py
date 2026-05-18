"""Testes do parser de CSV do Mind Monitor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.parser import MindMonitorParseError, load_csv
from tests.data.sample_generator import make_sample_df


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = make_sample_df(n_samples=320)
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p


def test_load_csv_valid(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    assert len(df) == 320
    assert 't_sec' in df.columns
    assert df['t_sec'].iloc[0] == 0.0
    assert df['t_sec'].iloc[-1] > 30.0
    assert pd.api.types.is_datetime64_any_dtype(df['TimeStamp'])


def test_load_csv_missing_column(tmp_path: Path) -> None:
    df = make_sample_df(n_samples=320)
    df = df.drop(columns=['Alpha_AF7'])
    p = tmp_path / "missing.csv"
    df.to_csv(p, index=False)

    with pytest.raises(MindMonitorParseError, match="Colunas obrigatórias ausentes"):
        load_csv(p)


def test_load_csv_invalid_timestamp(tmp_path: Path) -> None:
    df = make_sample_df(n_samples=320)
    df['TimeStamp'] = 'isto_não_é_uma_data'
    p = tmp_path / "bad_ts.csv"
    df.to_csv(p, index=False)

    with pytest.raises(MindMonitorParseError):
        load_csv(p)


def test_load_csv_all_bands_nan(tmp_path: Path) -> None:
    """CSV "RAW-only" da configuração antiga do Mind Monitor."""
    df = make_sample_df(n_samples=320)
    band_cols = [c for c in df.columns
                 if any(c.startswith(b + '_') for b in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])]
    df[band_cols] = np.nan
    p = tmp_path / "raw_only.csv"
    df.to_csv(p, index=False)

    with pytest.raises(MindMonitorParseError, match="RAW"):
        load_csv(p)


def test_load_csv_file_not_found() -> None:
    with pytest.raises(MindMonitorParseError, match="não encontrado"):
        load_csv("C:/temp/inexistente_abc123.csv")


def test_load_csv_path_with_accents(tmp_path: Path) -> None:
    """Caminho de arquivo com acentos brasileiros."""
    df = make_sample_df(n_samples=320)
    pasta = tmp_path / "Participação_001"
    pasta.mkdir()
    p = pasta / "sessão_vídeo.csv"
    df.to_csv(p, index=False, encoding='utf-8')

    result = load_csv(p)
    assert len(result) == 320
