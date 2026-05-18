"""Testes das 8 fórmulas de índices neurocognitivos.

Cada teste usa valores log10 cuidadosamente escolhidos para que a potência
linear seja inteira e o resultado seja calculável à mão. As fórmulas são o
coração da plataforma — se um teste aqui falhar, todos os números das
análises estão errados.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config import ALL_CHANNELS, BANDS
from core.indices import (
    compute_adherence,
    compute_affective_engagement,
    compute_all_indices,
    compute_arousal,
    compute_attention,
    compute_cognitive_engagement,
    compute_faa,
    compute_memory_evocation,
    compute_stress,
)
from core.preprocessing import log_to_linear


def _make_log_df(values: dict, n_rows: int = 1) -> pd.DataFrame:
    """Constrói DataFrame com colunas de banda × canal em escala log10.

    ``values`` é dict ``{band: log10_value}`` aplicado uniformemente a todos
    os canais. Colunas ausentes recebem 0 (potência linear = 1).
    """
    data: dict = {}
    for band in BANDS:
        v = values.get(band, 0.0)
        for ch in ALL_CHANNELS:
            data[f'{band}_{ch}'] = np.full(n_rows, v, dtype=float)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Atenção: β / (α + θ) frontal
# --------------------------------------------------------------------------
def test_attention_formula() -> None:
    # log10: α=1 → 10, β=2 → 100, θ=0 → 1.  Esperado: 100 / (10+1) = 9.0909...
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0, 'Theta': 0.0})
    df_lin = log_to_linear(df)
    result = compute_attention(df_lin)

    assert result.iloc[0] == pytest.approx(100 / 11, rel=1e-9)


def test_attention_uses_only_frontal_channels() -> None:
    """Mudanças em canais posteriores não devem afetar atenção."""
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0, 'Theta': 0.0})
    df.loc[:, 'Beta_TP9'] = 99.0  # valor absurdo só nos posteriores
    df.loc[:, 'Beta_TP10'] = 99.0
    df_lin = log_to_linear(df)

    assert compute_attention(df_lin).iloc[0] == pytest.approx(100 / 11, rel=1e-9)


# --------------------------------------------------------------------------
# Engajamento cognitivo e afetivo: (β + γ) / α
# --------------------------------------------------------------------------
def test_cognitive_engagement_formula() -> None:
    # α=10, β=100, γ=10 → (100+10)/10 = 11
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0, 'Gamma': 1.0})
    df_lin = log_to_linear(df)

    assert compute_cognitive_engagement(df_lin).iloc[0] == pytest.approx(11.0, rel=1e-9)


def test_affective_engagement_symmetric_alpha_equals_cognitive() -> None:
    """Caso degenerado: AF7 == AF8 → FAA = 0 → eng_afet = eng_cog."""
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0, 'Gamma': 1.0})
    df_lin = log_to_linear(df)

    cog = compute_cognitive_engagement(df_lin).iloc[0]
    afet = compute_affective_engagement(df_lin).iloc[0]
    assert afet == pytest.approx(cog, rel=1e-9)


def test_affective_engagement_asymmetric_alpha_adds_faa_magnitude() -> None:
    """AF7 != AF8 → eng_afet = |FAA| + eng_cog, distinto do eng_cog."""
    df = pd.DataFrame({
        'Alpha_AF7': [1.0], 'Alpha_AF8': [2.0],   # log10 → lin 10, 100 → média 55
        'Alpha_TP9': [1.0], 'Alpha_TP10': [1.0],
        'Beta_AF7':  [2.0], 'Beta_AF8':  [2.0],   # lin 100 cada → média 100
        'Beta_TP9':  [0.0], 'Beta_TP10': [0.0],
        'Gamma_AF7': [1.0], 'Gamma_AF8': [1.0],   # lin 10 cada → média 10
        'Gamma_TP9': [0.0], 'Gamma_TP10':[0.0],
        'Theta_AF7': [0.0], 'Theta_AF8': [0.0],
        'Theta_TP9': [0.0], 'Theta_TP10':[0.0],
        'Delta_AF7': [0.0], 'Delta_AF8': [0.0],
        'Delta_TP9': [0.0], 'Delta_TP10':[0.0],
    })
    df_lin = log_to_linear(df)

    cog = compute_cognitive_engagement(df_lin).iloc[0]
    faa = compute_faa(df_lin).iloc[0]
    afet = compute_affective_engagement(df_lin).iloc[0]

    # eng_cog = (100 + 10) / 55 = 2.0
    assert cog == pytest.approx(2.0, rel=1e-9)
    # FAA = (2 - 1) * ln(10) ≈ 2.3026
    assert faa == pytest.approx(math.log(10), rel=1e-9)
    # eng_afet = |FAA| + eng_cog ≈ 4.3026, distinto do eng_cog
    assert afet == pytest.approx(math.log(10) + 2.0, rel=1e-9)
    assert afet != pytest.approx(cog, rel=1e-3)


def test_affective_engagement_nonnegative_with_realistic_data() -> None:
    """|FAA| ≥ 0 e eng_cog > 0 (potências lineares) → eng_afet sempre ≥ 0."""
    rng = np.random.default_rng(0)
    cols: dict = {}
    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            cols[f'{band}_{ch}'] = rng.normal(0.5, 0.3, 50)
    df = pd.DataFrame(cols)
    df_lin = log_to_linear(df)
    assert (compute_affective_engagement(df_lin) >= 0).all()


# --------------------------------------------------------------------------
# FAA: ln(α_AF8) - ln(α_AF7), em log NATURAL
# --------------------------------------------------------------------------
def test_faa_formula() -> None:
    df = pd.DataFrame({
        'Alpha_AF7': [1.0],
        'Alpha_AF8': [2.0],
    })
    # log10: AF7=1 → α=10; AF8=2 → α=100
    # ln(100) - ln(10) = ln(10) ≈ 2.302585
    result = compute_faa(df)

    assert result.iloc[0] == pytest.approx(math.log(10), rel=1e-9)


def test_faa_symmetric_when_equal() -> None:
    df = pd.DataFrame({'Alpha_AF7': [1.5], 'Alpha_AF8': [1.5]})
    assert compute_faa(df).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_faa_sign_convention() -> None:
    """Atividade alfa maior à direita (AF8) → FAA positivo (mais aproximação)."""
    df = pd.DataFrame({'Alpha_AF7': [0.0], 'Alpha_AF8': [1.0]})
    assert compute_faa(df).iloc[0] > 0


# --------------------------------------------------------------------------
# Evocação de memória: θ posterior
# --------------------------------------------------------------------------
def test_memory_evocation_formula() -> None:
    # θ_TP9 = θ_TP10 → log10=1 → 10
    df = _make_log_df({'Theta': 1.0})
    df_lin = log_to_linear(df)

    assert compute_memory_evocation(df_lin).iloc[0] == pytest.approx(10.0, rel=1e-9)


def test_memory_evocation_ignores_frontal() -> None:
    df = _make_log_df({'Theta': 1.0})
    df.loc[:, 'Theta_AF7'] = 99.0
    df.loc[:, 'Theta_AF8'] = 99.0
    df_lin = log_to_linear(df)

    assert compute_memory_evocation(df_lin).iloc[0] == pytest.approx(10.0, rel=1e-9)


# --------------------------------------------------------------------------
# Aderência: (γ_front + γ_post) / θ_post
# --------------------------------------------------------------------------
def test_adherence_formula() -> None:
    # γ uniforme=10 → front=10, post=10. θ_post=1.
    # (10 + 10) / 1 = 20
    df = _make_log_df({'Gamma': 1.0, 'Theta': 0.0})
    df_lin = log_to_linear(df)

    assert compute_adherence(df_lin).iloc[0] == pytest.approx(20.0, rel=1e-9)


# --------------------------------------------------------------------------
# Arousal: β_total / α_total
# --------------------------------------------------------------------------
def test_arousal_formula() -> None:
    # β_all=100, α_all=10 → 10
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0})
    df_lin = log_to_linear(df)

    assert compute_arousal(df_lin).iloc[0] == pytest.approx(10.0, rel=1e-9)


# --------------------------------------------------------------------------
# Estresse: (β/α) + (γ/θ) — Arsalan et al. 2019
# --------------------------------------------------------------------------
def test_stress_formula() -> None:
    # β=100, α=10 → β/α=10; γ=10, θ=1 → γ/θ=10. Total = 20.
    df = _make_log_df({'Alpha': 1.0, 'Beta': 2.0, 'Gamma': 1.0, 'Theta': 0.0})
    df_lin = log_to_linear(df)

    assert compute_stress(df_lin).iloc[0] == pytest.approx(20.0, rel=1e-9)


# --------------------------------------------------------------------------
# Integração: pipeline completo compute_all_indices
# --------------------------------------------------------------------------
def test_compute_all_indices_constant_signal() -> None:
    """Sinal log10 constante → todos os índices constantes e iguais aos calculados à mão."""
    n_samples = 600  # 60 s a 10 Hz
    rate_hz = 10.0
    timestamps = pd.date_range('2025-01-01', periods=n_samples, freq=f'{1000/rate_hz:.3f}ms')

    df = _make_log_df(
        {'Alpha': 1.0, 'Beta': 2.0, 'Gamma': 1.0, 'Theta': 0.0, 'Delta': 0.0},
        n_rows=n_samples,
    )
    df['TimeStamp'] = timestamps
    df['t_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()

    indices = compute_all_indices(df)

    assert len(indices) > 10  # janelas suficientes para 60s/2.5s ≈ 22

    # Sinal constante → cada índice deve ter variância zero (ou desprezível)
    for col in ['atencao', 'eng_cognitivo', 'eng_afetivo', 'evocacao',
                'aderencia', 'arousal', 'estresse', 'faa']:
        assert indices[col].std() < 1e-9, f'{col} não é constante'

    # Valores esperados (mesmas contas dos testes unitários)
    assert indices['atencao'].iloc[0] == pytest.approx(100 / 11, rel=1e-9)
    assert indices['eng_cognitivo'].iloc[0] == pytest.approx(11.0, rel=1e-9)
    assert indices['eng_afetivo'].iloc[0] == pytest.approx(11.0, rel=1e-9)
    assert indices['evocacao'].iloc[0] == pytest.approx(1.0, rel=1e-9)  # θ_post lin = 10^0 = 1
    assert indices['aderencia'].iloc[0] == pytest.approx(20.0, rel=1e-9)
    assert indices['arousal'].iloc[0] == pytest.approx(10.0, rel=1e-9)
    assert indices['estresse'].iloc[0] == pytest.approx(20.0, rel=1e-9)
    assert indices['faa'].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_compute_all_indices_short_session_returns_empty() -> None:
    """Sessão menor que uma janela retorna DataFrame vazio com colunas certas."""
    n_samples = 30  # 3 s a 10 Hz (< 5 s)
    timestamps = pd.date_range('2025-01-01', periods=n_samples, freq='100ms')
    df = _make_log_df({'Alpha': 1.0}, n_rows=n_samples)
    df['TimeStamp'] = timestamps
    df['t_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()

    result = compute_all_indices(df)
    assert result.empty
    for col in ['atencao', 'eng_cognitivo', 'faa']:
        assert col in result.columns


def test_compute_all_indices_raises_without_t_sec() -> None:
    df = _make_log_df({'Alpha': 1.0})
    with pytest.raises(ValueError, match='t_sec'):
        compute_all_indices(df)


def test_log_to_linear_then_mean_differs_from_mean_then_log_to_linear() -> None:
    """Documenta a razão de fazer log→linear ANTES do janelamento.

    Se média dos log10s = 1.5 mas valores são 1.0 e 2.0:
    - mean(10^x) = (10 + 100)/2 = 55  (correto, médias aritméticas das potências)
    - 10^mean(x) = 10^1.5 ≈ 31.62      (média geométrica — viés pra baixo)
    """
    vals_log = np.array([1.0, 2.0])
    correct = np.mean(np.power(10.0, vals_log))
    wrong = np.power(10.0, np.mean(vals_log))

    assert correct == pytest.approx(55.0)
    assert wrong == pytest.approx(31.6227766, rel=1e-6)
    assert correct != pytest.approx(wrong, rel=1e-3)
