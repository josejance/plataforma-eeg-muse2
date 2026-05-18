"""Testes de pré-processamento: filtro de qualidade e conversão log→linear."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import ALL_CHANNELS, BANDS
from core.parser import load_csv
from core.preprocessing import QualityReport, log_to_linear, quality_filter
from tests.data.sample_generator import make_sample_df


def _to_df_with_t_sec(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona t_sec da mesma forma que o parser faria."""
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['t_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()
    return df


def test_log_to_linear_basic() -> None:
    df = pd.DataFrame({
        'Alpha_AF7': [0.0, 1.0, 2.0, -1.0],
        'Alpha_AF8': [0.0, 1.0, 2.0, -1.0],
        'Alpha_TP9': [0.0, 1.0, 2.0, -1.0],
        'Alpha_TP10': [0.0, 1.0, 2.0, -1.0],
    })
    out = log_to_linear(df)
    expected = [1.0, 10.0, 100.0, 0.1]
    for ch in ['AF7', 'AF8', 'TP9', 'TP10']:
        np.testing.assert_allclose(out[f'Alpha_{ch}_lin'].values, expected)


def test_log_to_linear_all_combinations() -> None:
    """Cria as 20 colunas _lin para todas as combinações banda×canal."""
    df = make_sample_df(n_samples=10)
    out = log_to_linear(df)
    for band in BANDS:
        for ch in ALL_CHANNELS:
            assert f'{band}_{ch}_lin' in out.columns, f'falta {band}_{ch}_lin'
            assert f'{band}_{ch}' in out.columns, 'coluna original deve ser preservada'


def test_quality_filter_drops_bad_rows() -> None:
    df = make_sample_df(n_samples=100, inject_bad_rows=20)
    df = _to_df_with_t_sec(df)
    df_valid, report = quality_filter(df)

    assert isinstance(report, QualityReport)
    assert report.n_samples_total == 100
    assert report.n_samples_valid == 80
    assert 19.0 < report.pct_discarded < 21.0


def test_quality_filter_keeps_clean_data() -> None:
    df = make_sample_df(n_samples=320, hsi_value=1)
    df = _to_df_with_t_sec(df)
    df_valid, report = quality_filter(df)

    assert report.n_samples_valid == 320
    assert report.pct_discarded == 0.0


def test_quality_alert_short_duration() -> None:
    df = make_sample_df(n_samples=50)  # 5 segundos a 10 Hz
    df = _to_df_with_t_sec(df)
    _, report = quality_filter(df)

    assert any('curta' in a.lower() for a in report.alerts)


def test_quality_alert_bad_hsi_mean() -> None:
    df = make_sample_df(n_samples=320, hsi_value=3)
    df = _to_df_with_t_sec(df)
    _, report = quality_filter(df)

    assert any('HSI médio alto' in a for a in report.alerts)


def test_quality_alert_headband_off() -> None:
    df = make_sample_df(n_samples=320, headband_on=0)
    df = _to_df_with_t_sec(df)
    _, report = quality_filter(df)

    assert any('Headband' in a for a in report.alerts)
    assert report.headband_off_ratio == pytest.approx(1.0)


def test_quality_blink_rate() -> None:
    df = make_sample_df(n_samples=600)  # 60 s a 10 Hz
    df = _to_df_with_t_sec(df)
    # 50 blinks em 60 s = 50/min — acima do limiar de 40
    df.loc[0:49, 'Elements'] = '/muse/elements/blink'
    _, report = quality_filter(df)

    assert report.blink_rate_per_min == pytest.approx(50.0, abs=0.1)
    assert any('piscadas' in a.lower() for a in report.alerts)


def test_quality_filter_empty_df() -> None:
    """Não deve quebrar em DataFrame vazio."""
    cols = ['TimeStamp', 'HeadBandOn', 't_sec']
    cols += [f'HSI_{ch}' for ch in ALL_CHANNELS]
    df = pd.DataFrame(columns=cols)
    df_valid, report = quality_filter(df)
    assert report.n_samples_total == 0
    assert report.pct_discarded == 0.0
