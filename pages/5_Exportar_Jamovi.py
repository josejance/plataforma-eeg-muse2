"""Página de exportação para Jamovi (wide, long, metadata)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import DB_PATH  # noqa: E402
from core.aggregations import build_master_table  # noqa: E402
from db.schema import get_connection, init_db  # noqa: E402
from exports.jamovi import (  # noqa: E402
    build_long,
    build_metadata,
    build_wide,
    to_csv_zip_bytes,
    to_xlsx_bytes,
)

st.set_page_config(page_title="Exportar Jamovi — EEG", page_icon="💾", layout="wide")
st.title("💾 Exportar para Jamovi")

init_db(DB_PATH)
conn = get_connection(DB_PATH)
master = build_master_table(conn)

if master.empty:
    st.info("Sem dados para exportar. Importe sessões primeiro.")
    st.stop()

wide = build_wide(master)
long_df = build_long(master)
metadata = build_metadata()

# --- Cabeçalho com estatísticas ----------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Participantes", wide.shape[0])
c2.metric("Sessões totais", long_df.shape[0])
c3.metric("Colunas wide", wide.shape[1])
c4.metric("Colunas long", long_df.shape[1])


# --- Preview ------------------------------------------------------------------
st.divider()
st.subheader("Preview")
tab_wide, tab_long, tab_meta = st.tabs(
    [f"📊 Wide ({wide.shape[0]}×{wide.shape[1]})",
     f"📈 Long ({long_df.shape[0]}×{long_df.shape[1]})",
     f"📋 Metadados ({len(metadata)})"]
)

with tab_wide:
    st.caption(
        "Uma linha por participante. Cada métrica por vídeo vira coluna "
        "`<metrica>_V1`, `<metrica>_V2`, etc. Médias entre os 4 vídeos terminam em `_mean`."
    )
    st.dataframe(wide, use_container_width=True, height=380, hide_index=True)

with tab_long:
    st.caption(
        "Uma linha por (participante × vídeo). Traços do participante "
        "aparecem replicados em cada linha — formato ideal para modelos mistos."
    )
    st.dataframe(long_df, use_container_width=True, height=380, hide_index=True)

with tab_meta:
    st.caption(
        "Descrição de cada variável: tipo (continuous/integer/nominal) e escala "
        "ou valores possíveis. Útil para configurar o Data Editor do Jamovi."
    )
    st.dataframe(metadata, use_container_width=True, height=380, hide_index=True)


# --- Downloads ----------------------------------------------------------------
st.divider()
st.subheader("Baixar")

stamp = datetime.now().strftime('%Y%m%d_%H%M')

# Pacote XLSX (3 abas) — recomendado
try:
    xlsx_bytes = to_xlsx_bytes({
        'wide': wide,
        'long': long_df,
        'metadata': metadata,
    })
    st.download_button(
        "⭐ **Baixar pacote XLSX** (3 abas: wide, long, metadata) — recomendado",
        data=xlsx_bytes,
        file_name=f"jamovi_pacote_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type='primary',
        use_container_width=True,
    )
except ModuleNotFoundError:
    st.warning(
        "`openpyxl` não está instalado — apenas CSV e ZIP disponíveis. "
        "Para habilitar XLSX, rode: `pip install openpyxl`"
    )

# ZIP (3 CSVs)
zip_bytes = to_csv_zip_bytes({
    'wide': wide,
    'long': long_df,
    'metadata': metadata,
})
st.download_button(
    "📦 **Baixar pacote ZIP** (3 CSVs com BOM UTF-8)",
    data=zip_bytes,
    file_name=f"jamovi_pacote_{stamp}.zip",
    mime="application/zip",
    use_container_width=True,
)

st.markdown("**CSVs individuais:**")
c1, c2, c3 = st.columns(3)
c1.download_button(
    "📥 wide.csv",
    data=wide.to_csv(index=False).encode('utf-8-sig'),
    file_name=f"jamovi_wide_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
)
c2.download_button(
    "📥 long.csv",
    data=long_df.to_csv(index=False).encode('utf-8-sig'),
    file_name=f"jamovi_long_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
)
c3.download_button(
    "📥 metadata.csv",
    data=metadata.to_csv(index=False).encode('utf-8-sig'),
    file_name=f"jamovi_metadata_{stamp}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()


# --- Bulk: TODAS as séries temporais por sessão ------------------------------
st.subheader("📦 Séries temporais (todas as sessões)")
st.caption(
    "ZIP contendo um CSV por sessão (`<código>/<vídeo>.csv`) com a série "
    "temporal completa dos 8 índices por janela de 5 s. **Não confunda com o "
    "wide/long acima** — esses contêm apenas as médias por sessão. Este ZIP é "
    "para análise em série temporal externa (ex.: filtro adicional, análise de "
    "wavelets, modelagem dinâmica)."
)

from db.queries import get_indices, list_sessions  # noqa: E402

all_sessions = list_sessions(conn)
n_sessions_with_data = sum(
    1 for s in all_sessions
    if not get_indices(conn, s['id']).empty
)

if st.button(
    f"🔧 Preparar ZIP com séries de {n_sessions_with_data} sessões",
    help="Gera o ZIP sob demanda — pode levar alguns segundos para 384 sessões."
):
    import io  # noqa: E402
    import zipfile  # noqa: E402

    progress = st.progress(0.0)
    buf = io.BytesIO()
    n_written = 0
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Recarrega lista com codes via JOIN
        sessions_with_code = conn.execute(
            "SELECT s.id, s.video_id, p.code AS code "
            "FROM sessions s JOIN participants p ON s.participant_id = p.id "
            "ORDER BY p.code, s.video_id"
        ).fetchall()
        total = len(sessions_with_code)
        for i, row in enumerate(sessions_with_code):
            idx_df = get_indices(conn, row['id'])
            if idx_df.empty:
                continue
            idx_df = idx_df.copy()
            idx_df.insert(0, 'video_id', row['video_id'])
            idx_df.insert(0, 'participant_code', row['code'])
            zf.writestr(
                f"{row['code']}/{row['video_id']}.csv",
                idx_df.to_csv(index=False).encode('utf-8-sig'),
            )
            n_written += 1
            progress.progress((i + 1) / total)
    progress.empty()
    st.session_state['bulk_zip'] = buf.getvalue()
    st.session_state['bulk_zip_n'] = n_written
    st.success(f"ZIP pronto · {n_written} CSVs.")

if 'bulk_zip' in st.session_state:
    st.download_button(
        f"📥 Baixar ZIP de séries ({st.session_state['bulk_zip_n']} sessões)",
        data=st.session_state['bulk_zip'],
        file_name=f"series_temporais_{stamp}.zip",
        mime="application/zip",
        type='primary',
        use_container_width=True,
    )


st.divider()
st.markdown(
    "##### Como abrir no Jamovi\n"
    "1. **XLSX**: `File → Open → Browse → Computer → Excel files (*.xlsx)` → "
    "selecionar a aba `wide` ou `long`.\n"
    "2. **CSV individual**: `File → Open → Computer → Text files (*.csv)`.\n"
    "3. A aba `metadata` documenta cada variável (tipo, escala) — referência "
    "para configurar o Data Editor.\n"
    "4. Para correlações entre traços e índices EEG: use o **wide** com módulo "
    "`Correlation Matrix`; para modelos mistos: use o **long** com `GAMLj`."
)
