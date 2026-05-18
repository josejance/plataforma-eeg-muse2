"""Testes de estatística inferencial."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.inferential import (
    anova_one_way,
    apa_anova,
    apa_correlation,
    apa_t_test,
    chi_squared_test,
    correlation_full,
    correlation_table,
    cross_tab,
    describe,
    descriptives_by_group,
    format_p,
    linear_regression,
    t_test_independent,
)


# ---------- Descritivas ----------
def test_describe_basic() -> None:
    d = describe([1, 2, 3, 4, 5])
    assert d.n == 5
    assert d.mean == pytest.approx(3.0)
    assert d.median == pytest.approx(3.0)
    assert d.sd == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))


def test_describe_empty() -> None:
    d = describe([])
    assert d.n == 0
    assert np.isnan(d.mean)


def test_descriptives_by_group() -> None:
    df = pd.DataFrame({
        'val': [1.0, 2.0, 3.0, 10.0, 20.0],
        'group': ['A', 'A', 'A', 'B', 'B'],
    })
    out = descriptives_by_group(df, 'val', 'group').set_index('grupo')
    assert out.loc['A', 'mean'] == pytest.approx(2.0)
    assert out.loc['B', 'mean'] == pytest.approx(15.0)
    assert out.loc['A', 'n'] == 3
    assert out.loc['B', 'n'] == 2


# ---------- t-test ----------
def test_t_test_two_groups() -> None:
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 10, 11, 12, 13],
        'g': ['A'] * 4 + ['B'] * 4,
    })
    res = t_test_independent(df, 'x', 'g')
    # Cross-check com scipy direto
    a = np.array([1, 2, 3, 4]); b = np.array([10, 11, 12, 13])
    t_expected, p_expected = stats.ttest_ind(a, b, equal_var=False)
    assert res['t'] == pytest.approx(t_expected)
    assert res['p_value'] == pytest.approx(p_expected)
    assert res['n_1'] == 4 and res['n_2'] == 4
    assert res['cohen_d'] < 0  # B > A


def test_t_test_three_groups_raises() -> None:
    df = pd.DataFrame({'x': [1, 2, 3], 'g': ['A', 'B', 'C']})
    with pytest.raises(ValueError, match='2 grupos'):
        t_test_independent(df, 'x', 'g')


# ---------- ANOVA ----------
def test_anova_one_way() -> None:
    df = pd.DataFrame({
        'x': [1, 2, 3, 10, 11, 12, 20, 21, 22],
        'g': ['A'] * 3 + ['B'] * 3 + ['C'] * 3,
    })
    res = anova_one_way(df, 'x', 'g')
    f_expected, p_expected = stats.f_oneway([1, 2, 3], [10, 11, 12], [20, 21, 22])
    assert res['F'] == pytest.approx(f_expected)
    assert res['p_value'] == pytest.approx(p_expected)
    assert res['df_between'] == 2
    assert res['df_within'] == 6
    assert 0 <= res['eta_squared'] <= 1


def test_anova_eta_squared_close_to_one_for_perfect_separation() -> None:
    df = pd.DataFrame({
        'x': [0, 0, 0, 100, 100, 100],
        'g': ['A'] * 3 + ['B'] * 3,
    })
    res = anova_one_way(df, 'x', 'g')
    assert res['eta_squared'] == pytest.approx(1.0)


# ---------- Correlação ----------
def test_correlation_full_perfect() -> None:
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
    res = correlation_full(df, 'x', 'y')
    assert res['pearson_r'] == pytest.approx(1.0)
    assert res['spearman_r'] == pytest.approx(1.0)
    assert res['n'] == 5


def test_correlation_full_returns_ci() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = x + rng.normal(0, 0.5, size=50)
    df = pd.DataFrame({'x': x, 'y': y})
    res = correlation_full(df, 'x', 'y')
    assert res['pearson_ci_lo'] < res['pearson_r'] < res['pearson_ci_hi']
    assert res['spearman_ci_lo'] < res['spearman_r'] < res['spearman_ci_hi']


def test_correlation_full_too_few_data() -> None:
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    res = correlation_full(df, 'x', 'y')
    assert np.isnan(res['pearson_r'])


def test_correlation_table_pairs() -> None:
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'c': [5, 4, 3, 2, 1],
    })
    out = correlation_table(df, ['a', 'b', 'c'])
    assert len(out) == 3  # 3 pares
    pair_ab = out[(out['X'] == 'a') & (out['Y'] == 'b')].iloc[0]
    assert pair_ab['pearson_r'] == pytest.approx(1.0)


# ---------- Regressão ----------
def test_linear_regression_simple_perfect_fit() -> None:
    df = pd.DataFrame({'x': np.arange(10, dtype=float),
                       'y': 2.0 * np.arange(10, dtype=float) + 1.0})
    res = linear_regression(df, ['x'], 'y')
    assert res['r_squared'] == pytest.approx(1.0)
    coefs = res['coefficients'].set_index('variable')
    assert coefs.loc['(intercept)', 'estimate'] == pytest.approx(1.0)
    assert coefs.loc['x', 'estimate'] == pytest.approx(2.0)


def test_linear_regression_multiple() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    y = 2 * x1 + 3 * x2 + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    res = linear_regression(df, ['x1', 'x2'], 'y')
    assert res['r_squared'] > 0.9
    assert res['F'] is not None
    coefs = res['coefficients'].set_index('variable')
    assert coefs.loc['x1', 'estimate'] == pytest.approx(2.0, abs=0.3)
    assert coefs.loc['x2', 'estimate'] == pytest.approx(3.0, abs=0.3)


def test_linear_regression_insufficient_data_raises() -> None:
    df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
    with pytest.raises(ValueError):
        linear_regression(df, ['x'], 'y')


# ---------- Qui-quadrado ----------
def test_chi_squared_independence_no_effect() -> None:
    """Tabela uniforme: chi² perto de 0, p alto."""
    df = pd.DataFrame({
        'a': ['x', 'x', 'y', 'y'] * 25,
        'b': ['p', 'q', 'p', 'q'] * 25,
    })
    res = chi_squared_test(df, 'a', 'b')
    assert res['p_value'] > 0.9
    assert res['chi_squared'] == pytest.approx(0.0, abs=1e-9)


def test_chi_squared_independence_perfect_assoc() -> None:
    df = pd.DataFrame({
        'a': ['x'] * 50 + ['y'] * 50,
        'b': ['p'] * 50 + ['q'] * 50,
    })
    res = chi_squared_test(df, 'a', 'b')
    assert res['p_value'] < 0.001
    assert res['cramer_v'] == pytest.approx(1.0)


# ---------- Cross-tab ----------
def test_cross_tab_counts() -> None:
    df = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['p', 'q', 'p']})
    ct = cross_tab(df, 'a', 'b')
    assert ct.loc['x', 'p'] == 1
    assert ct.loc['All', 'All'] == 3


def test_cross_tab_normalize_index() -> None:
    df = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['p', 'q', 'p']})
    ct = cross_tab(df, 'a', 'b', normalize='index')
    assert ct.loc['x', 'p'] == pytest.approx(0.5)


# ---------- Formatação APA ----------
def test_format_p() -> None:
    assert format_p(0.0001) == '<.001'
    assert format_p(0.04) == '.040'
    assert format_p(float('nan')) == '—'


def test_apa_t_test_string() -> None:
    res = {'df': 30, 't': 2.5, 'p_value': 0.018, 'cohen_d': 0.45}
    out = apa_t_test(res)
    assert 't(30)' in out and 'd = 0.45' in out


def test_apa_anova_string() -> None:
    res = {'df_between': 2, 'df_within': 30, 'F': 5.5, 'p_value': 0.009, 'eta_squared': 0.27}
    out = apa_anova(res)
    assert 'F(2, 30)' in out and 'η²' in out


def test_apa_correlation_string() -> None:
    res = {'n': 50, 'pearson_r': 0.35, 'pearson_p': 0.013}
    out = apa_correlation(res, 'pearson')
    assert 'r(48)' in out
