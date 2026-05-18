"""Linhas do tempo agregadas por grupo · médias entre participantes.

Para uma combinação (vídeo × índice), agrega as séries temporais de todos
os participantes que passam pelos filtros e mostra a média ao longo do tempo
com banda de incerteza (IC 95% ou ± SE). Permite comparar 2+ grupos no mesmo
gráfico (ex.: esquerda vs direita, concordou vs não concordou).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import DB_PATH, VIDEO_DURATIONS, VIDEO_IDS  # noqa: E402
from core.aggregations import (  # noqa: E402
    aggregate_timeseries,
    apply_filters,
    build_master_table,
)
from db.queries import INDEX_COLUMNS  # noqa: E402
from db.schema import get_connection, init_db  # noqa: E402
from visualizations.timeline import (  # noqa: E402
    INDEX_COLORS,
    INDEX_LABELS,
    aggregated_timeline_figure,
    comparison_timeline_figure,
)

st.set_page_config(page_title="Linhas do tempo agregadas — EEG",
                   page_icon="📈", layout="wide")
st.title("📈 Linhas do tempo agregadas por grupo")

init_db(DB_PATH)
conn = get_connection(DB_PATH)
master = build_master_table(conn)

if master.empty:
    st.info("Sem dados.")
    st.stop()


# --- Sidebar: filtros (mesmos da análise agregada) ---------------------------
with st.sidebar:
    st.header("Filtros")
    st.caption("Demografia")
    genders = st.multiselect("Gênero", sorted(master['gender'].dropna().unique()))
    age_groups = st.multiselect("Faixa etária",
                                  sorted(master['age_group'].dropna().unique()))
    political = st.multiselect("Posição política",
                                sorted(master['political_position'].dropna().unique()))

    st.caption("Avaliação do conteúdo")
    concordances = st.multiselect("Concordância",
                                    sorted(master['concordance'].dropna().unique()))
    veracities = st.multiselect("Veracidade",
                                  sorted(master['veracity'].dropna().unique()))
    sharing = st.multiselect("Compartilhamento",
                              sorted(master['sharing_intent'].dropna().unique()))


# --- Controles principais ----------------------------------------------------
c1, c2, c3 = st.columns(3)
video_id = c1.selectbox(
    "Vídeo", VIDEO_IDS,
    help=f"Durações esperadas: {VIDEO_DURATIONS}",
)
index_col = c2.selectbox(
    "Índice EEG", INDEX_COLUMNS,
    format_func=lambda c: INDEX_LABELS.get(c, c).split('·')[0].strip(),
)
band_choice = c3.selectbox(
    "Banda de incerteza",
    options=['IC 95%', '± SE', 'sem banda'],
    index=0,
)
band_map = {'IC 95%': 'ci', '± SE': 'sem', 'sem banda': 'none'}


# Aplica filtros para selecionar o conjunto de participantes do vídeo
# (filtrando o master pelo vídeo + outros critérios)
filtered_master = apply_filters(
    master, genders, age_groups, political, [video_id],
    concordances, veracities, sharing,
)
n_subjects = filtered_master['participant_code'].nunique()

mode = st.radio(
    "Modo",
    ['Único grupo agregado', 'Comparar grupos'],
    horizontal=True,
)


# ----------------------------------------------------------------------------
# Modo 1: agregação única
# ----------------------------------------------------------------------------
if mode == 'Único grupo agregado':
    st.caption(
        f"Vídeo `{video_id}` · {n_subjects} participante(s) após filtros · "
        f"banda = {band_choice}"
    )
    if filtered_master.empty:
        st.warning("Nenhum participante após filtros.")
        st.stop()

    codes = filtered_master['participant_code'].unique().tolist()
    agg = aggregate_timeseries(conn, video_id, codes, index_col)
    if agg.empty:
        st.warning("Sem séries temporais para essa combinação.")
        st.stop()

    label = INDEX_LABELS.get(index_col, index_col).split('·')[0].strip()
    color = INDEX_COLORS.get(index_col, '#1f77b4')
    fig = aggregated_timeline_figure(agg, label=label, color=color,
                                       band=band_map[band_choice])
    st.plotly_chart(fig, use_container_width=True, key='agg_single')

    with st.expander("Tabela de agregação (1 linha por janela)"):
        st.dataframe(agg.round(5), use_container_width=True, hide_index=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    st.download_button(
        f"📥 CSV da série agregada · `{index_col}` em `{video_id}` (n={n_subjects})",
        data=agg.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"agregada_{video_id}_{index_col}_n{n_subjects}_{stamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ----------------------------------------------------------------------------
# Modo 2: comparação entre grupos
# ----------------------------------------------------------------------------
else:
    split_options = ['gender', 'age_group', 'political_position',
                     'concordance', 'veracity', 'sharing_intent']
    split_by = st.selectbox(
        "Comparar por (uma linha por nível desta variável)",
        split_options,
        index=split_options.index('political_position'),
    )

    if filtered_master.empty:
        st.warning("Nenhum participante após filtros.")
        st.stop()

    levels = sorted(filtered_master[split_by].dropna().unique())
    if len(levels) < 2:
        st.warning(
            f"Apenas {len(levels)} nível em `{split_by}` após filtros — "
            "não é possível comparar."
        )
        st.stop()

    # Quais níveis incluir
    selected_levels = st.multiselect(
        f"Níveis de `{split_by}` a incluir",
        levels, default=levels,
    )
    if not selected_levels:
        st.info("Selecione pelo menos um nível.")
        st.stop()

    label = INDEX_LABELS.get(index_col, index_col).split('·')[0].strip()
    groups: dict[str, 'pd.DataFrame'] = {}
    n_per_level: dict[str, int] = {}
    for lvl in selected_levels:
        sub = filtered_master[filtered_master[split_by] == lvl]
        codes = sub['participant_code'].unique().tolist()
        n_per_level[lvl] = len(codes)
        if not codes:
            continue
        groups[str(lvl)] = aggregate_timeseries(conn, video_id, codes, index_col)

    st.caption(
        f"Vídeo `{video_id}` · `{split_by}`: "
        + " · ".join(f"**{l}**: n={n_per_level[l]}" for l in selected_levels)
    )

    fig = comparison_timeline_figure(groups, label=label,
                                      band=band_map[band_choice])
    st.plotly_chart(fig, use_container_width=True, key='agg_compare')

    # Tabela longa: t_window × grupo
    import pandas as pd  # noqa: E402
    combined_rows = []
    for name, agg in groups.items():
        if agg.empty:
            continue
        tmp = agg.copy()
        tmp.insert(0, 'grupo', name)
        combined_rows.append(tmp)
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        with st.expander("Tabela agregada (todas as janelas × todos os grupos)"):
            st.dataframe(combined.round(5), use_container_width=True, hide_index=True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M')
        st.download_button(
            f"📥 CSV comparativo · `{index_col}` em `{video_id}` por `{split_by}`",
            data=combined.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"comparativo_{video_id}_{index_col}_por_{split_by}_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --- Notas metodológicas -----------------------------------------------------
st.divider()
with st.expander("ℹ️ Notas metodológicas"):
    st.markdown(
        "- **Resolução temporal**: o pipeline opera em **janelas de 5 s** com "
        "**passo de 2,5 s** (50% de sobreposição). Os pontos no gráfico são "
        "espaçados a cada 2,5 s.\n"
        "- **Alinhamento**: todas as sessões usam exatamente o mesmo grid de "
        "janelas (mesmo `t_window`), então a agregação simples por `t_window` é "
        "válida sem necessidade de interpolação.\n"
        "- **Banda de incerteza**:\n"
        "  - **IC 95%** = média ± 1,96 × SE (default)\n"
        "  - **± SE** = média ± erro padrão\n"
        "  - **Sem banda** = só a linha da média\n"
        "- **n por janela** pode variar dentro da mesma sessão (artefatos "
        "filtrados em t específicos); o hover mostra `n` exato.\n"
        "- **FAA**: usa log natural com sinal preservado — valores positivos "
        "indicam aproximação (α maior à direita), negativos retração.\n"
    )
