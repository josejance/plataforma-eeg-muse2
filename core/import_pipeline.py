"""Pipeline de importação de uma sessão (CSV + metadados) para o banco.

Separado em duas etapas para que a UI possa exibir o painel de qualidade
ANTES de gravar nada:

  1. :func:`prepare_session` — leitura, filtro de qualidade e índices.
  2. :func:`persist_session`  — gravação de participante, sessão, autorrelato
     e série temporal de índices, conforme regras de duplicidade.

Aceita também ZIP contendo CSV via :func:`extract_csv_bytes`.
"""
from __future__ import annotations

import io
import sqlite3
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from core.indices import compute_all_indices
from core.logging_setup import get_logger
from core.parser import load_csv
from core.preprocessing import QualityReport, quality_filter, quality_score

_log = get_logger(__name__)
from db.queries import (
    SELF_REPORT_FIELDS,
    create_participant,
    create_session,
    delete_session,
    get_participant_by_code,
    list_sessions,
    resolve_video_id,
    save_indices,
    update_participant,
    upsert_self_report,
)


# ---------------------------------------------------------------------------
# Dataclasses de entrada
# ---------------------------------------------------------------------------
@dataclass
class ParticipantData:
    code: str
    gender: Optional[str] = None
    age: Optional[int] = None
    political_position: Optional[str] = None
    trait_anger: Optional[float] = None
    trait_fear: Optional[float] = None
    trait_stress: Optional[float] = None
    trait_narcissism: Optional[float] = None
    trait_humility: Optional[float] = None
    trait_mysticism: Optional[float] = None
    trait_habits: Optional[float] = None


@dataclass
class SessionData:
    video_id: str
    video_type: Optional[str] = None
    video_duration_expected: Optional[float] = None
    file_path: Optional[str] = None
    csv_filename: Optional[str] = None


@dataclass
class SelfReportData:
    alegria_intensity: Optional[float] = None
    medo_raiva_intensity: Optional[float] = None
    tristeza_intensity: Optional[float] = None
    serenidade_intensity: Optional[float] = None
    alegria_seconds: Optional[float] = None
    medo_raiva_seconds: Optional[float] = None
    tristeza_seconds: Optional[float] = None
    serenidade_seconds: Optional[float] = None
    concordance: Optional[str] = None
    veracity: Optional[str] = None
    sharing_intent: Optional[str] = None


@dataclass
class PreparedSession:
    df_raw: pd.DataFrame
    df_filtered: pd.DataFrame
    quality_report: QualityReport
    quality_score: float
    indices_df: pd.DataFrame


@dataclass
class PersistResult:
    participant_id: int
    session_id: int
    final_video_id: str
    participant_was_new: bool
    n_indices_rows: int


# ---------------------------------------------------------------------------
# Etapa 1: preparar (sem tocar no banco)
# ---------------------------------------------------------------------------
def prepare_session(csv_path: Union[str, Path]) -> PreparedSession:
    """Lê CSV, filtra qualidade e computa índices.

    Retorna :class:`PreparedSession` com dados intermediários para a UI exibir
    antes da gravação. Não escreve no banco.

    Raises:
        MindMonitorParseError: se o CSV for inválido.
    """
    df_raw = load_csv(csv_path)
    df_filtered, report = quality_filter(df_raw)
    score = quality_score(report)
    indices = (
        compute_all_indices(df_filtered)
        if not df_filtered.empty
        else pd.DataFrame(columns=['t_window'])
    )
    return PreparedSession(
        df_raw=df_raw,
        df_filtered=df_filtered,
        quality_report=report,
        quality_score=score,
        indices_df=indices,
    )


# ---------------------------------------------------------------------------
# Etapa 2: persistir
# ---------------------------------------------------------------------------
ON_DUPLICATE_MODES = ('suffix', 'replace', 'fail')


def persist_session(
    conn: sqlite3.Connection,
    prepared: PreparedSession,
    participant: ParticipantData,
    session: SessionData,
    self_report: Optional[SelfReportData] = None,
    on_duplicate_video: str = 'suffix',
) -> PersistResult:
    """Grava prepared + metadados no banco em uma transação por etapa.

    Args:
        conn: conexão aberta (FK habilitada).
        prepared: resultado de :func:`prepare_session`.
        participant: metadados do participante. Se ``code`` já existir, traços
            não-None são atualizados; demais campos do participante existente
            permanecem.
        session: metadados do estímulo. ``video_id`` pode ser ajustado se
            ``on_duplicate_video='suffix'``.
        self_report: dados pós-vídeo (opcional).
        on_duplicate_video: ``'suffix'`` (cria _v2/_v3), ``'replace'`` (apaga
            sessão antiga primeiro) ou ``'fail'`` (deixa IntegrityError subir).
    """
    if on_duplicate_video not in ON_DUPLICATE_MODES:
        raise ValueError(
            f"on_duplicate_video deve ser um de {ON_DUPLICATE_MODES}, "
            f"recebeu: {on_duplicate_video!r}"
        )

    pid, was_new = _upsert_participant(conn, participant)
    final_video_id = _resolve_video_conflict(
        conn, pid, session.video_id, on_duplicate_video
    )

    sid = create_session(
        conn,
        participant_id=pid,
        video_id=final_video_id,
        video_type=session.video_type,
        video_duration_expected=session.video_duration_expected,
        file_path=session.file_path,
        csv_filename=session.csv_filename,
        quality_score=prepared.quality_score,
        n_blinks_per_min=prepared.quality_report.blink_rate_per_min,
        n_samples_valid=prepared.quality_report.n_samples_valid,
        n_samples_total=prepared.quality_report.n_samples_total,
    )

    if self_report is not None and _has_any_value(self_report):
        upsert_self_report(conn, sid, **asdict(self_report))

    n_rows = 0
    if not prepared.indices_df.empty:
        n_rows = save_indices(conn, sid, prepared.indices_df)

    result = PersistResult(
        participant_id=pid,
        session_id=sid,
        final_video_id=final_video_id,
        participant_was_new=was_new,
        n_indices_rows=n_rows,
    )
    _log.info(
        "Sessao persistida: code=%s video=%s sid=%d novo_p=%s n_janelas=%d",
        participant.code, final_video_id, sid, was_new, n_rows,
    )
    return result


def _upsert_participant(
    conn: sqlite3.Connection, participant: ParticipantData
) -> tuple[int, bool]:
    """Insere o participante se não existir; senão atualiza traços não-None."""
    existing = get_participant_by_code(conn, participant.code)
    if existing is None:
        return create_participant(conn, **asdict(participant)), True

    # Já existe — atualizar somente os campos preenchidos (não-None)
    updates = {
        k: v for k, v in asdict(participant).items()
        if v is not None and k != 'code'
    }
    if updates:
        update_participant(conn, existing['id'], **updates)
    return int(existing['id']), False


def _resolve_video_conflict(
    conn: sqlite3.Connection,
    participant_id: int,
    base_video_id: str,
    mode: str,
) -> str:
    if mode == 'suffix':
        return resolve_video_id(conn, participant_id, base_video_id)
    if mode == 'replace':
        existing = list_sessions(
            conn, participant_id=participant_id, video_id=base_video_id
        )
        for s in existing:
            delete_session(conn, s['id'])
        return base_video_id
    return base_video_id  # 'fail' — IntegrityError sobe no create_session


def _has_any_value(sr: SelfReportData) -> bool:
    return any(getattr(sr, f) is not None for f in SELF_REPORT_FIELDS)


# ---------------------------------------------------------------------------
# Persistência de arquivos
# ---------------------------------------------------------------------------
def save_csv_copy(
    src: Union[str, Path, bytes],
    storage_dir: Union[str, Path],
    participant_code: str,
    video_id: str,
) -> Path:
    """Copia (ou grava bytes) o CSV original numa pasta organizada por participante.

    Retorna o caminho final como :class:`Path`. Sobrescreve em caso de mesmo nome.
    """
    out_dir = Path(storage_dir) / _safe_name(participant_code)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{_safe_name(video_id)}.csv"

    if isinstance(src, (bytes, bytearray)):
        dest.write_bytes(src)
    else:
        src_path = Path(src)
        dest.write_bytes(src_path.read_bytes())
    return dest


def _safe_name(name: str) -> str:
    """Remove caracteres problemáticos para sistema de arquivos."""
    bad = '<>:"/\\|?*'
    out = ''.join('_' if c in bad else c for c in name).strip()
    return out or 'sem_nome'


def extract_csv_bytes(source: Any, filename: Optional[str] = None) -> bytes:
    """Devolve os bytes do CSV a partir de um arquivo CSV puro ou de um ZIP.

    Aceita: ``bytes``, ``Path``, ``str`` (caminho), ou objeto de upload com
    ``.getbuffer()`` / ``.read()`` (o que o ``st.file_uploader`` do Streamlit
    devolve). Quando o nome termina em ``.zip``, extrai o primeiro CSV de dentro.

    Raises:
        ValueError: ZIP sem CSV dentro, ou ZIP corrompido.
    """
    # Determina os bytes brutos e o nome (para detectar zip)
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        name = (filename or '').lower()
    elif isinstance(source, (str, Path)):
        path = Path(source)
        raw = path.read_bytes()
        name = (filename or path.name).lower()
    elif hasattr(source, 'getbuffer'):
        raw = bytes(source.getbuffer())
        name = (filename or getattr(source, 'name', '') or '').lower()
    elif hasattr(source, 'read'):
        raw = source.read()
        name = (filename or getattr(source, 'name', '') or '').lower()
    else:
        raise TypeError(f"Fonte não suportada: {type(source)}")

    is_zip = name.endswith('.zip') or (raw[:4] == b'PK\x03\x04')
    if not is_zip:
        return raw

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_names = [
                n for n in zf.namelist()
                if n.lower().endswith('.csv') and not n.startswith('__MACOSX')
            ]
            if not csv_names:
                raise ValueError("ZIP não contém nenhum arquivo .csv")
            return zf.read(csv_names[0])
    except zipfile.BadZipFile as exc:
        raise ValueError(f"ZIP corrompido: {exc}") from exc
