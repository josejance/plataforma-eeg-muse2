"""Painel avançado · Análise inferencial estilo Jamovi.

Cinco seções (abas):
  1. Descritivas por grupo  · M, SD, SE, mediana, min, max
  2. Comparação de médias   · t-test (2 grupos) ou ANOVA (3+ grupos) + tamanho de efeito
  3. Correlação detalhada   · Pearson + Spearman + IC 95% + p-valores
  4. Regressão linear       · simples ou múltipla (OLS) com IC dos coeficientes
  5. Tabela de contingência · cross-tab + χ² independência + V de Cramér

Filtros da sidebar são compartilhados com a página 4 (Análise Agregada) e
acumulam-se com AND. Cada resultado tem botão de download em CSV.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import DB_PATH  # noqa: E402
from core.aggregations import (  # noqa: E402
    INDEX_MEAN_COLUMNS,
    SELF_REPORT_CATEGORICAL,
    SELF_REPORT_NUMERIC,
    TRAIT_COLUMNS,
    apply_filters,
    build_master_table,
)
from core.inferential import (  # noqa: E402
    anova_one_way,
    apa_anova,
    apa_correlation,
    apa_t_test,
    chi_squared_test,
    correlation_full,
    correlation_table,
    cross_tab,
    descriptives_by_group,
    format_p,
    linear_regression,
    t_test_independent,
)
from db.schema import get_connection, init_db  # noqa: E402

st.set_page_config(page_title="Painel Avançado — EEG", page_icon="🔬", layout="wide")
st.title("🔬 Painel Avançado · Análise Inferencial")

init_db(DB_PATH)
conn = get_connection(DB_PATH)
master = build_master_table(conn)

if master.empty:
    st.info("Sem dados. Importe sessões primeiro.")
    st.stop()


# --- Sidebar: filtros completos ----------------------------------------------
with st.sidebar:
    st.header("Filtros")
    genders = st.multiselect("Gênero", sorted(master['gender'].dropna().unique()))
    age_groups = st.multiselect("Faixa etária", sorted(master['age_group'].dropna().unique()))
    political = st.multiselect("Posição política", sorted(master['political_position'].dropna().unique()))
    videos = st.multiselect("Vídeo", sorted(master['video_id'].dropna().unique()))
    concordances = st.multiselect("Concordância", sorted(master['concordance'].dropna().unique()))
    veracities = st.multiselect("Veracidade", sorted(master['veracity'].dropna().unique()))
    sharing = st.multiselect("Compartilhamento", sorted(master['sharing_intent'].dropna().unique()))

filtered = apply_filters(
    master, genders, age_groups, political, videos,
    concordances, veracities, sharing,
)

st.caption(
    f"**{len(filtered)}** sessões selecionadas · "
    f"**{filtered['participant_id'].nunique()}** participantes únicos"
)
if filtered.empty:
    st.warning("Filtros zeraram a amostra. Afrouxe ou limpe as seleções.")
    st.stop()


# --- Conjuntos de variáveis disponíveis --------------------------------------
NUMERIC_VARS = (
    ['age'] + TRAIT_COLUMNS + INDEX_MEAN_COLUMNS
    + [c for c in SELF_REPORT_NUMERIC if c in filtered.columns]
    + ['quality_score', 'n_blinks_per_min']
)
NUMERIC_VARS = [c for c in NUMERIC_VARS if c in filtered.columns]

CATEGORICAL_VARS = [
    c for c in ['gender', 'age_group', 'political_position', 'video_id',
                'concordance', 'veracity', 'sharing_intent']
    if c in filtered.columns
]


def _csv_button(df: pd.DataFrame, filename_prefix: str, label: str) -> None:
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    st.download_button(
        f"📥 {label}",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"{filename_prefix}_{stamp}.csv",
        mime="text/csv",
    )


def _format_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Arredonda colunas float para exibição."""
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].round(decimals)
    return fmt


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descritivas por grupo",
    "🆎 Comparação de médias",
    "📈 Correlação detalhada",
    "🧮 Regressão linear",
    "🧊 Tabela de contingência",
])


# ============================================================================
# Tab 1: Descritivas por grupo
# ============================================================================
with tab1:
    st.markdown("Resumo estatístico de uma variável numérica por níveis de uma categórica.")
    c1, c2 = st.columns(2)
    val_col = c1.selectbox("Variável numérica", NUMERIC_VARS, key='d_val')
    grp_col = c2.selectbox("Agrupar por", CATEGORICAL_VARS, key='d_grp')

    if val_col and grp_col:
        out = descriptives_by_group(filtered, val_col, grp_col)
        st.dataframe(_format_df(out), use_container_width=True, hide_index=True)
        _csv_button(out, f"descritivas_{val_col}_por_{grp_col}",
                    "Baixar descritivas (CSV)")


# ============================================================================
# Tab 2: t-test / ANOVA
# ============================================================================
with tab2:
    st.markdown(
        "**t-test independente** (Welch) para 2 grupos ou "
        "**ANOVA one-way** para 3+ grupos. Detecta automaticamente pelo nº de níveis."
    )
    c1, c2 = st.columns(2)
    val_col = c1.selectbox("Variável dependente (numérica)", NUMERIC_VARS, key='tt_val')
    grp_col = c2.selectbox("Variável independente (categórica)", CATEGORICAL_VARS, key='tt_grp')

    if val_col and grp_col:
        n_groups = filtered[grp_col].dropna().nunique()
        st.caption(f"{n_groups} grupos detectados em `{grp_col}`")

        # Descritivas
        desc = descriptives_by_group(filtered, val_col, grp_col)
        st.markdown("##### Descritivas por grupo")
        st.dataframe(_format_df(desc), use_container_width=True, hide_index=True)

        st.markdown("##### Teste")
        try:
            if n_groups == 2:
                res = t_test_independent(filtered, val_col, grp_col)
                tbl = pd.DataFrame([{
                    'Teste': res['test'],
                    'M (grupo 1)': res['mean_1'],
                    'M (grupo 2)': res['mean_2'],
                    't': res['t'],
                    'df': res['df'],
                    'p': res['p_value'],
                    "d (Cohen)": res['cohen_d'],
                }])
                st.dataframe(_format_df(tbl), use_container_width=True, hide_index=True)
                st.info(f"**Resultado APA**: {apa_t_test(res)}")
                _csv_button(tbl, f"ttest_{val_col}_x_{grp_col}",
                            "Baixar resultado do teste (CSV)")
            elif n_groups >= 3:
                res = anova_one_way(filtered, val_col, grp_col)
                tbl = pd.DataFrame([{
                    'Teste': res['test'],
                    'F': res['F'],
                    'df (entre)': res['df_between'],
                    'df (dentro)': res['df_within'],
                    'p': res['p_value'],
                    "η²": res['eta_squared'],
                    'SS (entre)': res['ss_between'],
                    'SS (dentro)': res['ss_within'],
                }])
                st.dataframe(_format_df(tbl), use_container_width=True, hide_index=True)
                st.info(f"**Resultado APA**: {apa_anova(res)}")
                _csv_button(tbl, f"anova_{val_col}_x_{grp_col}",
                            "Baixar resultado do teste (CSV)")
            else:
                st.warning(f"Apenas {n_groups} grupo — não é possível comparar.")
        except ValueError as exc:
            st.error(f"Erro: {exc}")


# ============================================================================
# Tab 3: Correlação detalhada
# ============================================================================
with tab3:
    st.markdown(
        "**Pearson** (linear) + **Spearman** (rank) com IC 95% (transformação z de Fisher) "
        "e p-valor para uma única dupla, OU **matriz de correlações** para um conjunto."
    )

    mode = st.radio("Modo", ['Par único', 'Matriz de correlações'], horizontal=True, key='corr_mode')

    if mode == 'Par único':
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X", NUMERIC_VARS, key='corr_x')
        y_col = c2.selectbox("Y", NUMERIC_VARS, key='corr_y',
                              index=min(1, len(NUMERIC_VARS) - 1))
        if x_col and y_col and x_col != y_col:
            res = correlation_full(filtered, x_col, y_col)
            tbl = pd.DataFrame([{
                'n': res['n'],
                'Pearson r': res['pearson_r'],
                'IC 95% (Pearson)': f"[{res['pearson_ci_lo']:.3f}, {res['pearson_ci_hi']:.3f}]"
                    if not pd.isna(res['pearson_ci_lo']) else '—',
                'p (Pearson)': format_p(res['pearson_p']),
                'Spearman ρ': res['spearman_r'],
                'IC 95% (Spearman)': f"[{res['spearman_ci_lo']:.3f}, {res['spearman_ci_hi']:.3f}]"
                    if not pd.isna(res['spearman_ci_lo']) else '—',
                'p (Spearman)': format_p(res['spearman_p']),
            }])
            st.dataframe(_format_df(tbl), use_container_width=True, hide_index=True)
            if not pd.isna(res['pearson_r']):
                st.info(
                    f"**APA · Pearson**: {apa_correlation(res, 'pearson')}\n\n"
                    f"**APA · Spearman**: {apa_correlation(res, 'spearman')}"
                )
            _csv_button(tbl, f"corr_{x_col}_x_{y_col}",
                        "Baixar correlação (CSV)")
        elif x_col == y_col:
            st.info("Escolha duas variáveis diferentes.")

    else:  # Matriz
        cols = st.multiselect(
            "Variáveis (3 ou mais)", NUMERIC_VARS,
            default=TRAIT_COLUMNS[:3] + [INDEX_MEAN_COLUMNS[0]] if len(INDEX_MEAN_COLUMNS) > 0 else [],
        )
        if len(cols) >= 2:
            tbl = correlation_table(filtered, cols)
            st.dataframe(_format_df(tbl), use_container_width=True, hide_index=True)
            _csv_button(tbl, "matriz_correlacoes",
                        "Baixar matriz (CSV)")
        else:
            st.info("Selecione 2+ variáveis.")


# ============================================================================
# Tab 4: Regressão linear
# ============================================================================
with tab4:
    st.markdown(
        "Regressão linear OLS — simples (1 preditor) ou múltipla. Mostra coeficientes "
        "com IC 95%, R², R² ajustado e teste F do modelo completo."
    )

    c1, c2 = st.columns(2)
    y_col = c1.selectbox("Variável dependente (Y)", NUMERIC_VARS, key='reg_y')
    x_cols = c2.multiselect(
        "Preditores (X)",
        [v for v in NUMERIC_VARS if v != y_col],
        default=[TRAIT_COLUMNS[0]] if TRAIT_COLUMNS else [],
        key='reg_x',
    )

    if y_col and x_cols:
        try:
            res = linear_regression(filtered, x_cols, y_col)
            st.markdown("##### Resumo do modelo")
            summary = pd.DataFrame([{
                'n': res['n'],
                'R²': res['r_squared'],
                'R² ajustado': res['adj_r_squared'],
                'F': res['F'] if res['F'] is not None else '—',
                'df (modelo, resíduo)': f"{res['df_model']}, {res['df_resid']}",
                'p (F)': format_p(res['F_p_value']) if res['F_p_value'] is not None else '—',
            }])
            st.dataframe(_format_df(summary), use_container_width=True, hide_index=True)

            st.markdown("##### Coeficientes")
            coefs = res['coefficients'].copy()
            coefs['p_value'] = coefs['p_value'].apply(format_p)
            st.dataframe(_format_df(coefs), use_container_width=True, hide_index=True)
            _csv_button(coefs, f"regressao_{y_col}",
                        "Baixar coeficientes (CSV)")
        except ValueError as exc:
            st.error(f"Erro: {exc}")


# ============================================================================
# Tab 5: Tabela de contingência + χ²
# ============================================================================
with tab5:
    st.markdown(
        "Tabela de contingência entre duas variáveis categóricas + "
        "teste χ² de independência + V de Cramér (tamanho de efeito)."
    )

    c1, c2, c3 = st.columns(3)
    row_col = c1.selectbox("Linhas", CATEGORICAL_VARS, key='ct_row')
    col_col = c2.selectbox("Colunas",
                            [c for c in CATEGORICAL_VARS if c != row_col],
                            key='ct_col')
    norm = c3.selectbox(
        "Normalização",
        ['Contagens', '% do total', '% da linha', '% da coluna'],
        key='ct_norm',
    )
    norm_map = {'Contagens': None, '% do total': 'all',
                '% da linha': 'index', '% da coluna': 'columns'}

    if row_col and col_col:
        try:
            ct = cross_tab(filtered, row_col, col_col, normalize=norm_map[norm])
            st.markdown(f"##### Tabela ({norm.lower()})")
            display = ct if norm == 'Contagens' else (ct * 100).round(2)
            st.dataframe(display, use_container_width=True)

            res = chi_squared_test(filtered, row_col, col_col)
            tbl = pd.DataFrame([{
                'χ²': res['chi_squared'],
                'df': res['df'],
                'n': res['n'],
                'p': res['p_value'],
                'V de Cramér': res['cramer_v'],
            }])
            st.markdown("##### Teste de independência")
            st.dataframe(_format_df(tbl), use_container_width=True, hide_index=True)
            st.info(
                f"**APA**: χ²({res['df']}, N={res['n']}) = {res['chi_squared']:.2f}, "
                f"p = {format_p(res['p_value'])}, V = {res['cramer_v']:.3f}"
            )

            with st.expander("Frequências esperadas (sob hipótese de independência)"):
                st.dataframe(_format_df(res['expected']), use_container_width=True)

            _csv_button(ct.reset_index(), f"crosstab_{row_col}_x_{col_col}",
                        "Baixar tabela de contingência (CSV)")
        except ValueError as exc:
            st.error(f"Erro: {exc}")
