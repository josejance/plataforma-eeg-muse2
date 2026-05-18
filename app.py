"""Plataforma EEG Muse 2 — ponto de entrada Streamlit."""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que `config`, `core.*`, `db.*` sejam importáveis mesmo quando o
# Streamlit é iniciado de outro diretório de trabalho.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import DB_PATH, LOG_DIR, LOG_FILE  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402
from db.queries import list_participants, list_sessions  # noqa: E402
from db.schema import get_connection, init_db  # noqa: E402

# Inicializa logging em logs/app.log (rotação 10MB × 3)
setup_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Plataforma EEG", page_icon="🧠", layout="wide")

init_db(DB_PATH)
conn = get_connection(DB_PATH)

st.title("🧠 Plataforma EEG · Muse 2 / Mind Monitor")
st.markdown(
    "Análise de resposta neural a vídeos de desinformação política. "
    "Use o menu lateral para navegar entre as páginas."
)

participants = list_participants(conn)
sessions = list_sessions(conn)
sessions_with_eeg = [s for s in sessions if s.get('file_path')]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Participantes", len(participants))
c2.metric("Sessões registradas", len(sessions))
c3.metric("Com EEG importado", len(sessions_with_eeg))
c4.metric("Banco", DB_PATH.name)

st.divider()
st.markdown(
    "**Fluxo recomendado**\n"
    "1. 📋 Cadastrar Participante — registre demografia e os 7 traços.\n"
    "2. 📥 Importar sessões — envie o CSV/ZIP do Mind Monitor por slot de vídeo.\n"
    "3. 👤 Sessão Individual — explore os 6 gráficos da sessão.\n"
    "4. 📊 Análise Agregada — comparações entre grupos e correlações Spearman.\n"
    "5. 💾 Exportar Jamovi — gere wide/long/metadata em XLSX para análise estatística."
)


# --- Administração -----------------------------------------------------------
st.divider()
with st.expander("🔧 Administração"):
    st.markdown(
        f"- **Banco**: `{DB_PATH.resolve()}`\n"
        f"- **Log**: `{(LOG_DIR / LOG_FILE).resolve()}`"
    )

    st.markdown("**Recalcular índices**")
    st.caption(
        "Roda novamente o pipeline (filtros + janelamento + fórmulas) sobre os "
        "CSVs originais já importados, substituindo a série salva no banco. "
        "Use depois de mudanças de fórmula."
    )
    if st.button("🔁 Recalcular índices de todas as sessões com EEG"):
        from scripts.recompute_indices import recompute_all
        with st.spinner("Recalculando..."):
            stats = recompute_all(conn)
        logger.info("Recalculo concluído: %s", stats)
        st.success(
            f"✅ {stats['recalculadas']} recalculadas · "
            f"{stats['puladas']} puladas · "
            f"{len(stats['erros'])} erros"
        )
        for label, msg in stats['erros']:
            st.error(f"{label}: {msg}")

    st.markdown("**Últimas linhas do log**")
    log_path = LOG_DIR / LOG_FILE
    if log_path.exists():
        try:
            with log_path.open('r', encoding='utf-8') as fh:
                lines = fh.readlines()
            st.code(''.join(lines[-30:]) or '(log vazio)', language='text')
        except OSError as exc:
            st.warning(f"Não foi possível ler o log: {exc}")
    else:
        st.info("Log ainda não foi criado.")

logger.info(
    "app.py carregado · participantes=%d · sessoes=%d · com_eeg=%d",
    len(participants), len(sessions), len(sessions_with_eeg),
)
