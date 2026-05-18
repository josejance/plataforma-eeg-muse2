"""Estatística inferencial estilo Jamovi: descritivas, t-test, ANOVA,
correlações (Pearson + Spearman com IC 95%), regressão OLS e qui-quadrado.

Todas as funções devolvem ``dict`` ou ``pandas.DataFrame`` em formato pronto
para renderizar em tabela e exportar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Descritivas
# ---------------------------------------------------------------------------
@dataclass
class DescriptiveStats:
    n: int
    mean: float
    sd: float
    se: float
    median: float
    minimum: float
    maximum: float


def describe(values) -> DescriptiveStats:
    s = pd.Series(values).dropna()
    n = len(s)
    if n == 0:
        return DescriptiveStats(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    return DescriptiveStats(
        n=n,
        mean=float(s.mean()),
        sd=sd,
        se=sd / np.sqrt(n) if n > 0 else 0.0,
        median=float(s.median()),
        minimum=float(s.min()),
        maximum=float(s.max()),
    )


def descriptives_by_group(
    df: pd.DataFrame, value_col: str, group_col: str
) -> pd.DataFrame:
    """Descritivas (M, SD, SE, n, mediana, min, max) por nível de ``group_col``."""
    rows = []
    sub = df[[value_col, group_col]].dropna(subset=[group_col])
    for g, grp in sub.groupby(group_col, dropna=False):
        d = describe(grp[value_col])
        rows.append({'grupo': g, **asdict(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# t-test independente (Welch) + d de Cohen
# ---------------------------------------------------------------------------
def t_test_independent(
    df: pd.DataFrame, value_col: str, group_col: str
) -> dict:
    """t-test independente de Welch (não assume variâncias iguais)."""
    sub = df[[value_col, group_col]].dropna()
    groups = sorted(sub[group_col].unique())
    if len(groups) != 2:
        raise ValueError(
            f"t-test exige exatamente 2 grupos; encontrei {len(groups)}: {groups}"
        )
    a = sub[sub[group_col] == groups[0]][value_col].to_numpy()
    b = sub[sub[group_col] == groups[1]][value_col].to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Cada grupo precisa de pelo menos 2 observações.")

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # d de Cohen (variâncias agrupadas como em Jamovi default)
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_sd = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    cohen_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    return {
        'test': "Independent Samples T-Test (Welch)",
        'group_1': str(groups[0]), 'group_2': str(groups[1]),
        'n_1': int(len(a)), 'n_2': int(len(b)),
        'mean_1': float(a.mean()), 'mean_2': float(b.mean()),
        'sd_1': float(np.sqrt(var_a)), 'sd_2': float(np.sqrt(var_b)),
        't': float(t_stat),
        'df': float(len(a) + len(b) - 2),
        'p_value': float(p_value),
        'cohen_d': float(cohen_d),
    }


# ---------------------------------------------------------------------------
# One-way ANOVA + η²
# ---------------------------------------------------------------------------
def anova_one_way(
    df: pd.DataFrame, value_col: str, group_col: str
) -> dict:
    """One-way ANOVA com eta² (effect size)."""
    sub = df[[value_col, group_col]].dropna()
    groups = sorted(sub[group_col].unique())
    if len(groups) < 2:
        raise ValueError(f"ANOVA exige 2+ grupos; encontrei {len(groups)}.")
    samples = [sub[sub[group_col] == g][value_col].to_numpy() for g in groups]
    if any(len(s) < 2 for s in samples):
        raise ValueError("Cada grupo precisa de 2+ observações.")

    f_stat, p_value = stats.f_oneway(*samples)

    all_vals = np.concatenate(samples)
    grand_mean = all_vals.mean()
    ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
    ss_within = sum(((s - s.mean()) ** 2).sum() for s in samples)
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)

    return {
        'test': 'One-Way ANOVA',
        'groups': [str(g) for g in groups],
        'n_total': int(len(all_vals)),
        'F': float(f_stat),
        'df_between': int(df_between),
        'df_within': int(df_within),
        'p_value': float(p_value),
        'eta_squared': float(eta_squared),
        'ss_between': float(ss_between),
        'ss_within': float(ss_within),
        'ss_total': float(ss_total),
    }


# ---------------------------------------------------------------------------
# Correlação detalhada
# ---------------------------------------------------------------------------
def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confiança para correlação via transformação z de Fisher."""
    if n < 4 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


def correlation_full(
    df: pd.DataFrame, x_col: str, y_col: str, alpha: float = 0.05
) -> dict:
    """Pearson + Spearman com IC 95% (Fisher) e p-valores."""
    sub = df[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {
            'n': n, 'pearson_r': np.nan, 'pearson_p': np.nan,
            'pearson_ci_lo': np.nan, 'pearson_ci_hi': np.nan,
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'spearman_ci_lo': np.nan, 'spearman_ci_hi': np.nan,
            'note': 'n < 3 ou variância nula em uma das variáveis',
        }
    pr_r, pr_p = stats.pearsonr(sub[x_col], sub[y_col])
    sp = stats.spearmanr(sub[x_col].to_numpy(), sub[y_col].to_numpy())
    sp_r = float(getattr(sp, 'statistic', getattr(sp, 'correlation', np.nan)))
    sp_p = float(sp.pvalue)
    pr_lo, pr_hi = _fisher_ci(pr_r, n, alpha)
    sp_lo, sp_hi = _fisher_ci(sp_r, n, alpha)
    return {
        'n': int(n),
        'pearson_r': float(pr_r), 'pearson_p': float(pr_p),
        'pearson_ci_lo': pr_lo, 'pearson_ci_hi': pr_hi,
        'spearman_r': sp_r, 'spearman_p': sp_p,
        'spearman_ci_lo': sp_lo, 'spearman_ci_hi': sp_hi,
    }


def correlation_table(
    df: pd.DataFrame, cols: Sequence[str],
) -> pd.DataFrame:
    """Tabela com correlação par-a-par (Pearson + Spearman + n + p)."""
    cols = list(cols)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            res = correlation_full(df, a, b)
            rows.append({
                'X': a, 'Y': b, 'n': res['n'],
                'pearson_r': res['pearson_r'], 'pearson_p': res['pearson_p'],
                'spearman_r': res['spearman_r'], 'spearman_p': res['spearman_p'],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regressão linear OLS
# ---------------------------------------------------------------------------
def linear_regression(
    df: pd.DataFrame, x_cols: Sequence[str], y_col: str,
) -> dict:
    """OLS simples ou múltipla, com coeficientes, IC95%, t e p por coeficiente.

    Devolve também R², R² ajustado e teste F do modelo completo.
    """
    cols = list(x_cols) + [y_col]
    sub = df[cols].dropna()
    n = len(sub)
    k = len(x_cols)
    if n < k + 2:
        raise ValueError(f"Necessário n >= {k + 2}, recebido n={n}")

    X = sub[list(x_cols)].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    X_int = np.column_stack([np.ones(n), X])

    beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ beta
    resid = y - y_pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    df_resid = n - k - 1
    mse = ss_res / df_resid if df_resid > 0 else np.nan
    try:
        cov = mse * np.linalg.inv(X_int.T @ X_int)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k + 1, np.nan)

    t_stats = beta / se
    t_crit = stats.t.ppf(0.975, df_resid) if df_resid > 0 else np.nan
    p_values = [2 * (1 - stats.t.cdf(abs(t), df_resid)) if df_resid > 0 else np.nan
                for t in t_stats]

    if df_resid > 0 and r_squared < 1 and k > 0:
        f_stat = (r_squared / k) / ((1 - r_squared) / df_resid)
        f_p = float(stats.f.sf(f_stat, k, df_resid))
    else:
        f_stat = np.nan
        f_p = np.nan

    coefs = pd.DataFrame({
        'variable': ['(intercept)'] + list(x_cols),
        'estimate': beta,
        'se': se,
        'ci_95_lo': beta - t_crit * se,
        'ci_95_hi': beta + t_crit * se,
        't': t_stats,
        'p_value': p_values,
    })

    return {
        'n': n,
        'r_squared': float(r_squared),
        'adj_r_squared': float(1 - (1 - r_squared) * (n - 1) / df_resid) if df_resid > 0 else np.nan,
        'F': float(f_stat) if not np.isnan(f_stat) else None,
        'F_p_value': f_p if not np.isnan(f_p) else None,
        'df_model': k,
        'df_resid': df_resid,
        'coefficients': coefs,
    }


# ---------------------------------------------------------------------------
# Qui-quadrado de independência + V de Cramér
# ---------------------------------------------------------------------------
def chi_squared_test(
    df: pd.DataFrame, row_col: str, col_col: str,
    correction: bool = False,
) -> dict:
    """χ² de independência entre duas variáveis categóricas + V de Cramér.

    ``correction=False`` (default) desativa a correção de continuidade de Yates,
    seguindo a convenção mais comum em pacotes estatísticos (incluindo Jamovi)
    para cálculo do V de Cramér.
    """
    ct = pd.crosstab(df[row_col], df[col_col])
    if ct.empty or ct.values.sum() == 0:
        raise ValueError("Tabela de contingência vazia.")
    result = stats.chi2_contingency(ct, correction=correction)
    # scipy >= 1.11 retorna ContingencyResult; antes retornava tupla
    if hasattr(result, 'statistic'):
        chi2 = float(result.statistic)
        p_value = float(result.pvalue)
        dof = int(result.dof)
        expected = result.expected_freq
    else:
        chi2, p_value, dof, expected = result
        chi2, p_value, dof = float(chi2), float(p_value), int(dof)

    n = int(ct.values.sum())
    r, c = ct.shape
    min_dim = min(r - 1, c - 1)
    cramer_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

    return {
        'test': "Chi-squared Test of Independence",
        'chi_squared': chi2,
        'p_value': p_value,
        'df': dof,
        'n': n,
        'cramer_v': cramer_v,
        'observed': ct,
        'expected': pd.DataFrame(expected, index=ct.index, columns=ct.columns),
    }


def cross_tab(
    df: pd.DataFrame, row_col: str, col_col: str,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Tabela de contingência com totais; opcionalmente normalizada.

    Args:
        normalize: ``None`` (contagens), ``'index'`` (% da linha),
            ``'columns'`` (% da coluna), ou ``'all'`` (% do total).
    """
    # pandas exige False (não None) para "sem normalização"
    norm_arg = normalize if normalize else False
    return pd.crosstab(df[row_col], df[col_col], normalize=norm_arg, margins=True)


# ---------------------------------------------------------------------------
# Formatadores para estilo Jamovi/APA
# ---------------------------------------------------------------------------
def format_p(p: float) -> str:
    if pd.isna(p):
        return '—'
    if p < .001:
        return '<.001'
    return f"{p:.3f}".lstrip('0')


def apa_t_test(res: dict) -> str:
    return (
        f"t({res['df']:.0f}) = {res['t']:.2f}, p = {format_p(res['p_value'])}, "
        f"d = {res['cohen_d']:.2f}"
    )


def apa_anova(res: dict) -> str:
    return (
        f"F({res['df_between']}, {res['df_within']}) = {res['F']:.2f}, "
        f"p = {format_p(res['p_value'])}, η² = {res['eta_squared']:.3f}"
    )


def apa_correlation(res: dict, kind: str = 'pearson') -> str:
    r = res[f'{kind}_r']
    p = res[f'{kind}_p']
    n = res['n']
    return f"r({n - 2}) = {r:.2f}, p = {format_p(p)}"
