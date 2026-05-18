"""Cadastro de participante: dados demográficos + escores dos 7 traços.

Traços são valores numéricos decimais já calculados a partir de instrumentos
externos (Likert). Cada participante tem um valor por traço; esse valor é
replicado para todas as sessões dele.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import DB_PATH, GENDER_OPTIONS, POLITICAL_OPTIONS  # noqa: E402
from core.aggregations import TRAIT_COLUMNS, TRAIT_LABELS  # noqa: E402
from db.queries import (  # noqa: E402
    create_participant,
    get_participant_by_code,
    list_participants,
    update_participant,
)
from db.schema import get_connection, init_db  # noqa: E402

st.set_page_config(page_title="Cadastrar participante — EEG", page_icon="📋", layout="wide")
st.title("📋 Cadastrar participante")

init_db(DB_PATH)
conn = get_connection(DB_PATH)


def _select_index(options: list[str], value) -> int:
    """Devolve o índice de ``value`` em ``['—'] + options`` (0 = sem seleção)."""
    if value is None or value not in options:
        return 0
    return options.index(value) + 1


# --- Formulário ---------------------------------------------------------------
code = st.text_input("Código do participante *", placeholder="ex: P001")
existing = get_participant_by_code(conn, code) if code else None
is_editing = existing is not None

if is_editing:
    st.info(f"Editando participante existente · id = {existing['id']}")
elif code:
    st.success("Cadastro novo")

# Sufixo nas chaves garante que os widgets resetam ao trocar de código
key_suffix = code or "__novo"

st.subheader("Dados demográficos")
c1, c2, c3 = st.columns(3)
gender = c1.selectbox(
    "Gênero", ['—'] + GENDER_OPTIONS,
    index=_select_index(GENDER_OPTIONS, existing['gender'] if existing else None),
    key=f"gender_{key_suffix}",
)
age = c2.number_input(
    "Idade", min_value=0, max_value=120, step=1,
    value=int(existing['age']) if existing and existing['age'] is not None else 25,
    key=f"age_{key_suffix}",
)
political = c3.selectbox(
    "Posição política", ['—'] + POLITICAL_OPTIONS,
    index=_select_index(POLITICAL_OPTIONS, existing['political_position'] if existing else None),
    key=f"pol_{key_suffix}",
)

st.subheader("Traços (escore decimal do instrumento)")
st.caption(
    "Cada traço é a pontuação numérica final calculada a partir do questionário "
    "Likert; aceita valores decimais (ex.: 7.40). Pode deixar em branco se algum "
    "instrumento não foi aplicado."
)

trait_values: dict[str, float | None] = {}
trait_cols = st.columns(2)
for i, col_key in enumerate(TRAIT_COLUMNS):
    label = TRAIT_LABELS[col_key]
    default = existing[col_key] if existing and existing[col_key] is not None else None
    trait_values[col_key] = trait_cols[i % 2].number_input(
        label,
        value=float(default) if default is not None else None,
        step=0.01,
        format="%.2f",
        key=f"trait_{col_key}_{key_suffix}",
        placeholder="vazio",
    )

empty_count = sum(1 for v in trait_values.values() if v is None)
if empty_count > 0:
    st.warning(
        f"{empty_count} de {len(TRAIT_COLUMNS)} traços em branco. "
        "Você pode salvar mesmo assim."
    )

st.divider()
action_label = "Atualizar participante" if is_editing else "Cadastrar participante"
disabled = not code.strip()
if st.button(action_label, type="primary", use_container_width=True, disabled=disabled):
    fields = {
        'gender': None if gender == '—' else gender,
        'age': int(age) if age else None,
        'political_position': None if political == '—' else political,
        **trait_values,
    }
    try:
        if is_editing:
            update_participant(conn, existing['id'], **fields)
            st.success(f"Participante `{code}` atualizado.")
        else:
            create_participant(conn, code=code.strip(), **fields)
            st.success(f"Participante `{code}` cadastrado.")
    except Exception as exc:  # noqa: BLE001 — UI captura para feedback
        st.error(f"Erro ao salvar: {exc}")

if disabled:
    st.caption("Preencha o código para habilitar o botão.")


# --- Listagem -----------------------------------------------------------------
st.divider()
st.subheader("Participantes já cadastrados")

participants = list_participants(conn)
if not participants:
    st.info("Nenhum participante ainda. Use o formulário acima.")
else:
    df = pd.DataFrame(participants)
    base_cols = ['code', 'gender', 'age', 'political_position']
    show_cols = [c for c in base_cols + TRAIT_COLUMNS if c in df.columns]
    rename_map = {
        'code': 'código', 'gender': 'gênero', 'age': 'idade',
        'political_position': 'posição política',
        **TRAIT_LABELS,
    }
    st.dataframe(
        df[show_cols].rename(columns=rename_map),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Total: **{len(participants)}** participantes.")
