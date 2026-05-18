"""Página de importação: 1 slot por vídeo (V1–V4) por participante.

Cada slot aceita CSV ou ZIP (com CSV dentro) do Mind Monitor. Avaliação
do conteúdo (concordância/veracidade/compartilhamento) já vem pré-cadastrada
do questionário externo; o pesquisador só preenche as emoções e sobe o EEG.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Bootstrap sys.path para imports do projeto independente do cwd
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import CSV_STORAGE_DIR, DB_PATH, VIDEO_DURATIONS, VIDEO_IDS  # noqa: E402
from core.aggregations import TRAIT_COLUMNS, TRAIT_LABELS  # noqa: E402
from core.import_pipeline import (  # noqa: E402
    ParticipantData,
    PreparedSession,
    SelfReportData,
    SessionData,
    extract_csv_bytes,
    persist_session,
    prepare_session,
    save_csv_copy,
)
from core.parser import MindMonitorParseError  # noqa: E402
from db.queries import (  # noqa: E402
    get_self_report,
    list_participants,
    list_sessions,
)
from db.schema import get_connection, init_db  # noqa: E402

st.set_page_config(page_title="Importar — EEG", page_icon="📥", layout="wide")
st.title("📥 Importar sessões EEG")

init_db(DB_PATH)
conn = get_connection(DB_PATH)


@dataclass
class VideoSlotState:
    """Estado processado de um slot (após upload + prepare_session)."""
    upload_id: tuple
    tmp_path: Path
    prep: PreparedSession


def _slot_state_key(video_id: str) -> str:
    return f"slot_state_{video_id}"


def _reset_slot(video_id: str) -> None:
    state = st.session_state.pop(_slot_state_key(video_id), None)
    if state and state.tmp_path.exists():
        state.tmp_path.unlink(missing_ok=True)


def _process_upload(upload, video_id: str) -> Optional[VideoSlotState]:
    """Processa o arquivo subido (CSV ou ZIP) com cache por upload_id."""
    upload_id = (upload.name, len(upload.getbuffer()))
    state: Optional[VideoSlotState] = st.session_state.get(_slot_state_key(video_id))
    if state and state.upload_id == upload_id:
        return state

    if state:
        _reset_slot(video_id)

    try:
        csv_bytes = extract_csv_bytes(upload, filename=upload.name)
    except ValueError as exc:
        st.error(f"Erro ao extrair CSV de {upload.name}: {exc}")
        return None

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = Path(tmp.name)

    try:
        prep = prepare_session(tmp_path)
    except MindMonitorParseError as exc:
        st.error(f"CSV inválido em {upload.name}: {exc}")
        tmp_path.unlink(missing_ok=True)
        return None

    state = VideoSlotState(upload_id=upload_id, tmp_path=tmp_path, prep=prep)
    st.session_state[_slot_state_key(video_id)] = state
    return state


def _render_quality(prep: PreparedSession) -> None:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Score", f"{prep.quality_score:.2f}")
    q2.metric(
        "Amostras válidas",
        f"{prep.quality_report.n_samples_valid}/{prep.quality_report.n_samples_total}",
    )
    q3.metric("Piscadas/min", f"{prep.quality_report.blink_rate_per_min:.1f}")
    q4.metric("Duração", f"{prep.quality_report.duration_sec:.1f} s")
    for alert in prep.quality_report.alerts:
        st.warning(alert)


def _render_pre_response(sess: Optional[dict]) -> None:
    """Mostra concordância/veracidade/compartilhamento pré-cadastrados."""
    if not sess:
        return
    sr = get_self_report(conn, sess['id'])
    if not sr:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Concordância", sr.get('concordance') or '—')
    c2.metric("Veracidade", sr.get('veracity') or '—')
    c3.metric("Compartilhamento", sr.get('sharing_intent') or '—')


def _emotion_form(video_id: str) -> dict:
    """Inputs de intensidade + tempo das 4 emoções. Retorna dict com Nones."""
    st.markdown("**Intensidade percebida (0–10)**")
    c1, c2, c3, c4 = st.columns(4)
    int_alegria = c1.slider("Alegria", 0.0, 10.0, 0.0, 0.5, key=f"int_aleg_{video_id}")
    int_medo = c2.slider("Medo/Raiva", 0.0, 10.0, 0.0, 0.5, key=f"int_medo_{video_id}")
    int_tristeza = c3.slider("Tristeza", 0.0, 10.0, 0.0, 0.5, key=f"int_trist_{video_id}")
    int_seren = c4.slider("Serenidade", 0.0, 10.0, 0.0, 0.5, key=f"int_ser_{video_id}")

    st.markdown("**Tempo sentido (s)**")
    c1, c2, c3, c4 = st.columns(4)
    sec_alegria = c1.number_input("Alegria (s)", 0.0, value=0.0, step=1.0, key=f"sec_aleg_{video_id}")
    sec_medo = c2.number_input("Medo/Raiva (s)", 0.0, value=0.0, step=1.0, key=f"sec_medo_{video_id}")
    sec_tristeza = c3.number_input("Tristeza (s)", 0.0, value=0.0, step=1.0, key=f"sec_trist_{video_id}")
    sec_seren = c4.number_input("Serenidade (s)", 0.0, value=0.0, step=1.0, key=f"sec_ser_{video_id}")

    return {
        'alegria_intensity': int_alegria if int_alegria > 0 else None,
        'medo_raiva_intensity': int_medo if int_medo > 0 else None,
        'tristeza_intensity': int_tristeza if int_tristeza > 0 else None,
        'serenidade_intensity': int_seren if int_seren > 0 else None,
        'alegria_seconds': sec_alegria if sec_alegria > 0 else None,
        'medo_raiva_seconds': sec_medo if sec_medo > 0 else None,
        'tristeza_seconds': sec_tristeza if sec_tristeza > 0 else None,
        'serenidade_seconds': sec_seren if sec_seren > 0 else None,
    }


def _persist_slot(
    code: str, video_id: str, state: VideoSlotState,
    upload_name: str, emotions: dict, pre_sess: Optional[dict],
) -> Any:
    """Executa a gravação no banco para um slot. Preserva concord/verac/sharing."""
    pre_sr = get_self_report(conn, pre_sess['id']) if pre_sess else None

    csv_dest = save_csv_copy(state.tmp_path, CSV_STORAGE_DIR, code, video_id)

    session_data = SessionData(
        video_id=video_id,
        video_duration_expected=VIDEO_DURATIONS.get(video_id),
        file_path=str(csv_dest),
        csv_filename=upload_name,
    )
    self_report = SelfReportData(
        **emotions,
        concordance=pre_sr.get('concordance') if pre_sr else None,
        veracity=pre_sr.get('veracity') if pre_sr else None,
        sharing_intent=pre_sr.get('sharing_intent') if pre_sr else None,
    )
    # 'replace' substitui sessão esqueleto (sem EEG) ou re-importação
    return persist_session(
        conn, state.prep, ParticipantData(code=code), session_data, self_report,
        on_duplicate_video='replace',
    )


# --- Sidebar: participante ----------------------------------------------------
with st.sidebar:
    st.header("Participante")
    participants = list_participants(conn)
    if not participants:
        st.warning("Nenhum participante cadastrado. Vá em 📋 Cadastrar Participante.")
        st.stop()

    codes = [p['code'] for p in participants]
    code = st.selectbox("Selecionar *", codes, key='participant_select')
    selected = next(p for p in participants if p['code'] == code)
    st.caption(
        f"id={selected['id']} · "
        f"{selected.get('gender') or '—'} · "
        f"{selected.get('age') or '—'} anos · "
        f"{selected.get('political_position') or '—'}"
    )
    with st.expander("Traços do participante"):
        for tc in TRAIT_COLUMNS:
            v = selected.get(tc)
            st.text(f"{TRAIT_LABELS[tc]:<22}: {v if v is not None else '—'}")


# --- Status row + abas dos 4 vídeos -------------------------------------------
sessions_for_p = list_sessions(conn, participant_id=selected['id'])
sessions_by_video: dict[str, dict] = {}
for s in sessions_for_p:
    if s['video_id'] in VIDEO_IDS:
        sessions_by_video[s['video_id']] = s

st.subheader(f"Slots de vídeo de `{code}`")
status_cols = st.columns(len(VIDEO_IDS))
for col, vid in zip(status_cols, VIDEO_IDS):
    sess = sessions_by_video.get(vid)
    label_dur = f"{VIDEO_DURATIONS[vid]:.0f}s"
    if sess and sess.get('file_path'):
        col.success(f"✅ **{vid}** · EEG importado · esperado {label_dur}")
    elif sess:
        col.info(f"⏳ **{vid}** · só pré-cadastro · esperado {label_dur}")
    else:
        col.warning(f"❌ **{vid}** · sem dados · esperado {label_dur}")

tabs = st.tabs([f"📹 {v}  ·  {VIDEO_DURATIONS[v]:.0f}s" for v in VIDEO_IDS])

for tab, vid in zip(tabs, VIDEO_IDS):
    with tab:
        sess = sessions_by_video.get(vid)

        # Pré-cadastro (questionário externo)
        st.markdown("##### Avaliação pré-cadastrada (questionário)")
        _render_pre_response(sess)

        st.divider()
        st.markdown("##### Upload do EEG (CSV ou ZIP contendo CSV)")
        if sess and sess.get('file_path'):
            st.info(
                f"Já existe EEG para `{vid}` desse participante. "
                "Subir um novo arquivo **substitui** o anterior."
            )

        upload = st.file_uploader(
            f"Arquivo EEG de {vid}",
            type=['csv', 'zip'],
            key=f"upload_{vid}",
            label_visibility='collapsed',
        )

        slot_state = None
        if upload is not None:
            slot_state = _process_upload(upload, vid)

        if slot_state is not None:
            st.markdown("##### Qualidade do sinal")
            _render_quality(slot_state.prep)
        elif upload is None:
            st.caption("Sem arquivo carregado neste slot.")

        st.divider()
        st.markdown("##### Autorrelato pós-vídeo")
        emotions = _emotion_form(vid)

        # Botão individual
        disabled = slot_state is None
        if st.button(
            f"💾 Importar {vid}", type='primary',
            use_container_width=True, key=f"btn_{vid}", disabled=disabled,
        ):
            try:
                result = _persist_slot(
                    code, vid, slot_state, upload.name, emotions, sess,
                )
                st.success(
                    f"✅ `{vid}` importado · sessão #{result.session_id} · "
                    f"{result.n_indices_rows} janelas"
                )
                _reset_slot(vid)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro: {exc}")


# --- Botão de importação em bloco --------------------------------------------
st.divider()
pending_slots = [
    vid for vid in VIDEO_IDS
    if _slot_state_key(vid) in st.session_state
    and st.session_state[_slot_state_key(vid)] is not None
]

if pending_slots:
    st.markdown(
        f"**{len(pending_slots)} slot(s) prontos**: "
        + ", ".join(f"`{v}`" for v in pending_slots)
    )
    if st.button(
        f"📦 Importar todos os {len(pending_slots)} slots prontos",
        type='primary', use_container_width=True,
    ):
        results = []
        for vid in pending_slots:
            state = st.session_state[_slot_state_key(vid)]
            sess = sessions_by_video.get(vid)
            # Recuperar emoções do session_state (chaves dos sliders/inputs)
            emotions = {
                'alegria_intensity': st.session_state.get(f"int_aleg_{vid}", 0.0) or None,
                'medo_raiva_intensity': st.session_state.get(f"int_medo_{vid}", 0.0) or None,
                'tristeza_intensity': st.session_state.get(f"int_trist_{vid}", 0.0) or None,
                'serenidade_intensity': st.session_state.get(f"int_ser_{vid}", 0.0) or None,
                'alegria_seconds': st.session_state.get(f"sec_aleg_{vid}", 0.0) or None,
                'medo_raiva_seconds': st.session_state.get(f"sec_medo_{vid}", 0.0) or None,
                'tristeza_seconds': st.session_state.get(f"sec_trist_{vid}", 0.0) or None,
                'serenidade_seconds': st.session_state.get(f"sec_ser_{vid}", 0.0) or None,
            }
            emotions = {k: (v if v and v > 0 else None) for k, v in emotions.items()}
            upload_obj = st.session_state.get(f"upload_{vid}")
            upload_name = upload_obj.name if upload_obj else f"{vid}.csv"
            try:
                r = _persist_slot(code, vid, state, upload_name, emotions, sess)
                results.append((vid, True, f"sessão #{r.session_id}"))
                _reset_slot(vid)
            except Exception as exc:  # noqa: BLE001
                results.append((vid, False, str(exc)))

        for vid, ok, msg in results:
            (st.success if ok else st.error)(f"{vid}: {msg}")
        if all(ok for _, ok, _ in results):
            st.balloons()
            st.rerun()
else:
    st.caption(
        "Suba arquivos nos slots acima para habilitar a importação em bloco."
    )
