"""Página de análise agregada: tabela mestra, boxplots, correlações, scatter."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import DB_PATH  # noqa: E402
from core.aggregations import (  # noqa: E402
    INDEX_MEAN_COLUMNS,
    SELF_REPORT_NUMERIC,
    TRAIT_COLUMNS,
    apply_filters,
    build_master_table,
)
from db.schema import get_connection, init_db  # noqa: E402
from visualizations.comparisons import boxplot_by_group, scatter_with_trendline  # noqa: E402
from visualizations.correlations import correlation_heatmap, spearman_matrix  # noqa: E402

st.set_page_config(page_title="Análise agregada — EEG", page_icon="📊", layout="wide")
st.title("📊 Análise agregada")

init_db(DB_PATH)
conn = get_connection(DB_PATH)
master = build_master_table(conn)

if master.empty:
    st.info("Nenhuma sessão registrada. Use a página 📥 Importar primeiro.")
    st.stop()


# --- Sidebar: filtros --------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    st.caption("Demografia")
    genders = st.multiselect(
        "Gênero", sorted(master['gender'].dropna().unique()),
    )
    age_groups = st.multiselect(
        "Faixa etária", sorted(master['age_group'].dropna().unique()),
    )
    political = st.multiselect(
        "Posição política", sorted(master['political_position'].dropna().unique()),
    )

    st.caption("Estímulo")
    videos = st.multiselect(
        "Vídeo", sorted(master['video_id'].dropna().unique()),
    )

    st.caption("Avaliação do conteúdo")
    concordances = st.multiselect(
        "Concordância", sorted(master['concordance'].dropna().unique()),
    )
    veracities = st.multiselect(
        "Veracidade", sorted(master['veracity'].dropna().unique()),
    )
    sharing = st.multiselect(
        "Compartilhamento", sorted(master['sharing_intent'].dropna().unique()),
    )

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


# --- Tabela mestra -----------------------------------------------------------
st.subheader("Tabela mestra")
st.caption(
    f"**{len(filtered)}** linhas × **{len(filtered.columns)}** colunas "
    "(role lateralmente na tabela para ver todas)."
)
st.dataframe(filtered, use_container_width=True, height=320, hide_index=True)


def _csv_bytes(df: 'pd.DataFrame') -> bytes:
    """CSV com BOM UTF-8 para abrir corretamente no Excel/Jamovi (Windows)."""
    return df.to_csv(index=False).encode('utf-8-sig')


from datetime import datetime  # noqa: E402

stamp = datetime.now().strftime('%Y%m%d-%H%M')

col_dl1, col_dl2 = st.columns(2)
col_dl1.download_button(
    "📥 Baixar **tabela filtrada** (CSV, todas as colunas)",
    data=_csv_bytes(filtered),
    file_name=f"tabela_mestra_filtrada_{len(filtered)}linhas_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
    help=f"{len(filtered)} linhas × {len(filtered.columns)} colunas",
)
col_dl2.download_button(
    "📥 Baixar **tabela completa** (CSV, ignora filtros)",
    data=_csv_bytes(master),
    file_name=f"tabela_mestra_completa_{len(master)}linhas_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
    help=f"{len(master)} linhas × {len(master.columns)} colunas",
)


# --- Comparações entre grupos -----------------------------------------------
st.divider()
st.subheader("Comparações entre grupos")

c1, c2 = st.columns(2)
group_col = c1.selectbox(
    "Agrupar por", ['gender', 'age_group', 'political_position', 'video_id'],
    index=0,
)
index_col = c2.selectbox(
    "Índice EEG", INDEX_MEAN_COLUMNS,
    index=INDEX_MEAN_COLUMNS.index('atencao_mean'),
)

if filtered[group_col].dropna().nunique() < 2:
    st.info(f"Só há um grupo em {group_col} após filtros — boxplot não faz sentido.")
else:
    st.plotly_chart(
        boxplot_by_group(filtered, index_col, group_col, points='all'),
        use_container_width=True,
    )


# --- Correlações Spearman ----------------------------------------------------
st.divider()
st.subheader("Correlações entre traços e índices EEG médios")
st.caption(
    "Spearman ρ entre os 6 traços e as 8 médias de índice por sessão. "
    "Asterisco (`*`) marca p < 0,05 (cuidado com inflação por múltiplos testes)."
)

# Traços são por participante; usar média por participante para evitar repetições
participant_level = (
    filtered.groupby('participant_id')[TRAIT_COLUMNS + INDEX_MEAN_COLUMNS]
    .mean(numeric_only=True)
    .reset_index()
)
rho_te, pval_te = spearman_matrix(
    participant_level, cols_x=TRAIT_COLUMNS, cols_y=INDEX_MEAN_COLUMNS,
)
st.plotly_chart(
    correlation_heatmap(rho_te, pval_te, title='Traços × índices EEG (média por participante)'),
    use_container_width=True,
)

# Bônus: autorrelato pós-vídeo × índices (no nível de sessão)
self_report_cols = [c for c in SELF_REPORT_NUMERIC if filtered[c].notna().sum() >= 3]
if self_report_cols:
    rho_sr, pval_sr = spearman_matrix(
        filtered, cols_x=self_report_cols, cols_y=INDEX_MEAN_COLUMNS,
    )
    st.plotly_chart(
        correlation_heatmap(
            rho_sr, pval_sr,
            title='Autorrelato pós-vídeo × índices EEG (por sessão)',
        ),
        use_container_width=True,
    )


# --- Scatter exploratório ----------------------------------------------------
st.divider()
st.subheader("Scatter exploratório")

numeric_cols = filtered.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ('participant_id', 'session_id')]
categorical_cols = ['', 'gender', 'age_group', 'political_position', 'video_id']

c1, c2, c3 = st.columns(3)
x_col = c1.selectbox("Eixo X", numeric_cols,
                     index=numeric_cols.index('trait_anger') if 'trait_anger' in numeric_cols else 0)
y_col = c2.selectbox("Eixo Y", numeric_cols,
                     index=numeric_cols.index('atencao_mean') if 'atencao_mean' in numeric_cols else 0)
color_col = c3.selectbox("Colorir por (opcional)", categorical_cols)

st.plotly_chart(
    scatter_with_trendline(filtered, x_col, y_col, color_col=color_col or None),
    use_container_width=True,
)
