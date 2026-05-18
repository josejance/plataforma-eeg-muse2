"""Página de visualização de uma sessão individual."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import DB_PATH  # noqa: E402
from core.parser import MindMonitorParseError, load_csv  # noqa: E402
from core.statistics import extract_blink_times, summarize_indices  # noqa: E402
from db.queries import (  # noqa: E402
    get_indices,
    get_participant,
    get_self_report,
    list_participants,
    list_sessions,
)
from db.schema import get_connection, init_db  # noqa: E402
from visualizations.timeline import (  # noqa: E402
    PRIMARY_INDICES,
    cognitive_vs_affective_figure,
    faa_timeline_figure,
    timeline_figure,
)

st.set_page_config(page_title="Sessão individual — EEG", page_icon="👤", layout="wide")
st.title("👤 Sessão individual")

init_db(DB_PATH)
conn = get_connection(DB_PATH)

participants = list_participants(conn)
if not participants:
    st.info("Nenhum participante cadastrado. Use a página 📥 Importar primeiro.")
    st.stop()


# --- Sidebar: seletor de sessão ----------------------------------------------
with st.sidebar:
    st.header("Selecionar sessão")
    p_codes = [p['code'] for p in participants]
    p_choice = st.selectbox("Participante", p_codes)
    pid = next(p['id'] for p in participants if p['code'] == p_choice)

    sessions = list_sessions(conn, participant_id=pid)
    if not sessions:
        st.warning("Participante sem sessões. Importe uma.")
        st.stop()

    s_choice = st.selectbox(
        "Sessão", sessions,
        format_func=lambda s: f"{s['video_id']} — {s.get('video_type') or 'sem tipo'}",
    )
    sid = s_choice['id']

    st.divider()
    show_median = st.checkbox("Linha da mediana", value=True)
    show_blinks = st.checkbox("Marcadores de piscadas", value=False)


# --- Cabeçalho ---------------------------------------------------------------
participant = get_participant(conn, pid)
session = s_choice

st.subheader(f"Participante {participant['code']} · Sessão `{session['video_id']}`")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Participante**")
    st.markdown(
        f"- Gênero: {participant.get('gender') or '—'}\n"
        f"- Idade: {participant.get('age') or '—'}\n"
        f"- Posição política: {participant.get('political_position') or '—'}"
    )
with c2:
    st.markdown("**Sessão**")
    st.markdown(
        f"- Tipo: {session.get('video_type') or '—'}\n"
        f"- Duração esperada: {session.get('video_duration_expected') or '—'} s\n"
        f"- Arquivo: {session.get('csv_filename') or '—'}"
    )
with c3:
    st.markdown("**Qualidade do sinal**")
    qs = session.get('quality_score')
    st.markdown(
        f"- Score: {qs:.2f}\n" if qs is not None else "- Score: —\n"
    )
    st.markdown(
        f"- Piscadas/min: {session.get('n_blinks_per_min', 0):.1f}\n"
        f"- Amostras válidas: "
        f"{session.get('n_samples_valid') or 0}/{session.get('n_samples_total') or 0}"
    )

# Autorrelato (se houver)
self_report = get_self_report(conn, sid)
if self_report:
    with st.expander("📝 Autorrelato pós-vídeo"):
        cols = st.columns(4)
        for col, key, label in zip(
            cols,
            ['alegria_intensity', 'medo_raiva_intensity',
             'tristeza_intensity', 'serenidade_intensity'],
            ['Alegria', 'Medo/Raiva', 'Tristeza', 'Serenidade'],
        ):
            v = self_report.get(key)
            col.metric(label, f"{v:.1f}" if v is not None else '—')
        st.caption(
            f"Concordância: {self_report.get('concordance') or '—'} · "
            f"Veracidade: {self_report.get('veracity') or '—'} · "
            f"Compartilhamento: {self_report.get('sharing_intent') or '—'}"
        )


# --- Carregar índices --------------------------------------------------------
from db.queries import INDEX_COLUMNS  # noqa: E402

from visualizations.timeline import INDEX_LABELS  # noqa: E402

indices_df = get_indices(conn, sid)
has_time_series = len(indices_df) >= 2

if not has_time_series:
    st.info(
        "🟦 **Esta sessão só tem valores médios agregados** (sem série temporal). "
        "Os 8 índices abaixo vieram do CSV de resultados consolidado. "
        "Para ver linhas do tempo, importe o CSV bruto do Mind Monitor desta "
        "sessão pela página **📥 Importar**."
    )
    st.subheader("📋 Valores médios da sessão")
    cols = st.columns(4)
    for i, idx in enumerate(INDEX_COLUMNS):
        val = session.get(f'{idx}_mean')
        label = INDEX_LABELS.get(idx, idx).split('·')[0].strip()
        cols[i % 4].metric(label, f"{val:.4f}" if val is not None else '—')
    st.stop()

blink_times: list[float] = []
if show_blinks:
    file_path = session.get('file_path')
    if file_path and Path(file_path).exists():
        try:
            raw = load_csv(file_path)
            blink_times = extract_blink_times(raw)
            st.caption(f"{len(blink_times)} piscadas detectadas no CSV original.")
        except MindMonitorParseError as e:
            st.warning(f"Não foi possível recarregar o CSV para piscadas: {e}")
    else:
        st.info("CSV original não encontrado — não há como marcar piscadas.")


# --- Gráficos ----------------------------------------------------------------
st.divider()
st.subheader("Linhas do tempo · 5 painéis principais")

# 1. Atenção (sozinho)
st.plotly_chart(
    timeline_figure(indices_df, 'atencao', show_median=show_median,
                    blink_times=blink_times if show_blinks else None),
    use_container_width=True, key='plot_atencao',
)

# 2. Engajamento cognitivo + afetivo (combinados, com área da diferença = |FAA|)
st.plotly_chart(
    cognitive_vs_affective_figure(
        indices_df, show_median=show_median,
        blink_times=blink_times if show_blinks else None,
    ),
    use_container_width=True, key='plot_eng_combinado',
)
st.caption(
    "💡 Cognitivo e afetivo dividem o mesmo termo `(β+γ)/α frontal`; o afetivo "
    "soma `|FAA|`. A área verde clara entre as curvas mostra exatamente essa "
    "contribuição da valência — quanto maior a área, mais mobilizada a "
    "emoção no momento."
)

# 3. FAA (renderer especial com sinal preservado)
st.plotly_chart(
    faa_timeline_figure(
        indices_df, blink_times=blink_times if show_blinks else None,
    ),
    use_container_width=True, key='plot_faa',
)

# 4. Evocação de memórias
st.plotly_chart(
    timeline_figure(indices_df, 'evocacao', show_median=show_median,
                    blink_times=blink_times if show_blinks else None),
    use_container_width=True, key='plot_evocacao',
)

# 5. Aderência
st.plotly_chart(
    timeline_figure(indices_df, 'aderencia', show_median=show_median,
                    blink_times=blink_times if show_blinks else None),
    use_container_width=True, key='plot_aderencia',
)


# --- Tabela resumo -----------------------------------------------------------
st.subheader("📋 Estatísticas por índice")
summary = summarize_indices(indices_df)
st.dataframe(
    summary.style.format({
        'mediana': '{:.4f}',
        'média': '{:.4f}',
        'std': '{:.4f}',
        'min': '{:.4f}',
        'max': '{:.4f}',
        '% acima mediana': '{:.1f}%',
    }),
    use_container_width=True, hide_index=True,
)
st.caption(
    "Linha da mediana intrassujeito tem `% acima` ≈ 50% por construção; "
    "o valor passa a ser informativo só quando comparado a uma referência externa."
)


# --- Downloads das séries temporais ------------------------------------------
import io  # noqa: E402
import zipfile  # noqa: E402
from datetime import datetime  # noqa: E402

st.divider()
st.subheader("📥 Exportar séries temporais")

stamp = datetime.now().strftime('%Y%m%d_%H%M')


def _series_with_metadata(df: 'pd.DataFrame', code: str, vid: str) -> 'pd.DataFrame':
    """Adiciona colunas de identificação no início do DataFrame da série."""
    out = df.copy()
    out.insert(0, 'video_id', vid)
    out.insert(0, 'participant_code', code)
    return out


def _csv_bytes(df: 'pd.DataFrame') -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')


# 1. Esta sessão (1 CSV)
serie_atual = _series_with_metadata(indices_df, p_choice, session['video_id'])
c1, c2 = st.columns(2)
c1.download_button(
    f"📥 CSV desta sessão · `{p_choice}/{session['video_id']}`",
    data=_csv_bytes(serie_atual),
    file_name=f"serie_{p_choice}_{session['video_id']}_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
    help=f"{len(serie_atual)} linhas × {len(serie_atual.columns)} colunas",
)

# 2. ZIP com todas as sessões deste participante
all_sessions_p = list_sessions(conn, participant_id=pid)
buf_p = io.BytesIO()
n_files = 0
with zipfile.ZipFile(buf_p, 'w', zipfile.ZIP_DEFLATED) as zf:
    for s in all_sessions_p:
        idx_df = get_indices(conn, s['id'])
        if idx_df.empty:
            continue
        ser = _series_with_metadata(idx_df, p_choice, s['video_id'])
        zf.writestr(f"{p_choice}/{s['video_id']}.csv", _csv_bytes(ser))
        n_files += 1
c2.download_button(
    f"📦 ZIP de todas as sessões de `{p_choice}` ({n_files} arquivo{'s' if n_files != 1 else ''})",
    data=buf_p.getvalue(),
    file_name=f"sessoes_{p_choice}_{stamp}.zip",
    mime="application/zip",
    use_container_width=True,
    disabled=n_files == 0,
)

# 3. Por índice individual (uma série por arquivo)
with st.expander("📊 Baixar série de um índice individual"):
    st.caption(
        "Útil quando quer abrir só uma das séries (ex.: só atenção) em "
        "ferramenta externa."
    )
    single_cols = st.columns(4)
    for i, idx_name in enumerate(
        ['atencao', 'eng_cognitivo', 'eng_afetivo', 'evocacao',
         'aderencia', 'faa', 'arousal', 'estresse']
    ):
        if idx_name not in indices_df.columns:
            continue
        single_df = serie_atual[['participant_code', 'video_id', 't_window', idx_name]]
        single_cols[i % 4].download_button(
            f"{idx_name}.csv",
            data=_csv_bytes(single_df),
            file_name=f"{p_choice}_{session['video_id']}_{idx_name}_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_single_{idx_name}",
        )
