"""Testes do pipeline de importação (CSV + metadados → banco)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core.import_pipeline import (
    ParticipantData,
    SelfReportData,
    SessionData,
    persist_session,
    prepare_session,
    save_csv_copy,
)
from core.parser import MindMonitorParseError
from db.queries import (
    get_indices,
    get_participant,
    get_participant_by_code,
    get_self_report,
    get_session,
    list_sessions,
)
from db.schema import init_db
from tests.data.sample_generator import make_sample_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path: Path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = make_sample_df(n_samples=600)  # 60 s a 10 Hz
    p = tmp_path / "session.csv"
    df.to_csv(p, index=False)
    return p


def _make_participant(code: str = "P001", **kw) -> ParticipantData:
    return ParticipantData(code=code, gender="feminino", age=24,
                           political_position="centro", trait_anger=3.0, **kw)


def _make_session(video_id: str = "V1") -> SessionData:
    return SessionData(video_id=video_id, video_type="desinformação",
                       video_duration_expected=60.0, csv_filename="session.csv")


def _make_self_report() -> SelfReportData:
    return SelfReportData(
        alegria_intensity=2.0, medo_raiva_intensity=7.0,
        tristeza_intensity=3.0, serenidade_intensity=1.0,
        alegria_seconds=5.0, medo_raiva_seconds=30.0,
        tristeza_seconds=10.0, serenidade_seconds=2.0,
        concordance="Não concordo", veracity="Mentiroso",
        sharing_intent="Não compartilharia",
    )


# ---------------------------------------------------------------------------
# prepare_session
# ---------------------------------------------------------------------------
def test_prepare_session_full_pipeline(sample_csv: Path) -> None:
    prep = prepare_session(sample_csv)

    assert prep.df_raw.shape[0] == 600
    assert prep.df_filtered.shape[0] == 600  # CSV limpo, nada descartado
    assert prep.quality_report.n_samples_valid == 600
    assert prep.quality_score == pytest.approx(1.0)  # HSI=1 em todos
    assert len(prep.indices_df) > 10
    assert 'atencao' in prep.indices_df.columns


def test_prepare_session_with_bad_signal(tmp_path: Path) -> None:
    df = make_sample_df(n_samples=600, inject_bad_rows=120)
    p = tmp_path / "noisy.csv"
    df.to_csv(p, index=False)

    prep = prepare_session(p)
    assert prep.quality_report.n_samples_valid == 480
    assert prep.quality_report.pct_discarded == pytest.approx(20.0)


def test_prepare_session_invalid_csv_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("não,é,um,csv,válido", encoding='utf-8')
    with pytest.raises(MindMonitorParseError):
        prepare_session(p)


def test_prepare_session_short_signal_empty_indices(tmp_path: Path) -> None:
    df = make_sample_df(n_samples=30)  # 3 s
    p = tmp_path / "short.csv"
    df.to_csv(p, index=False)
    prep = prepare_session(p)

    assert prep.indices_df.empty
    assert any('curta' in a.lower() for a in prep.quality_report.alerts)


# ---------------------------------------------------------------------------
# persist_session — participante novo
# ---------------------------------------------------------------------------
def test_persist_session_new_participant(conn: sqlite3.Connection, sample_csv: Path) -> None:
    prep = prepare_session(sample_csv)
    result = persist_session(
        conn, prep, _make_participant(), _make_session(), _make_self_report()
    )

    assert result.participant_was_new is True
    assert result.final_video_id == "V1"
    assert result.n_indices_rows > 10

    p = get_participant(conn, result.participant_id)
    assert p['code'] == "P001"
    assert p['age'] == 24
    s = get_session(conn, result.session_id)
    assert s['video_id'] == "V1"
    assert s['quality_score'] == pytest.approx(1.0)
    assert s['n_samples_valid'] == 600

    sr = get_self_report(conn, result.session_id)
    assert sr['medo_raiva_intensity'] == 7.0

    idx_df = get_indices(conn, result.session_id)
    assert len(idx_df) == result.n_indices_rows
    assert 'atencao' in idx_df.columns


def test_persist_session_without_self_report(conn: sqlite3.Connection, sample_csv: Path) -> None:
    prep = prepare_session(sample_csv)
    result = persist_session(
        conn, prep, _make_participant(), _make_session(), self_report=None,
    )
    assert get_self_report(conn, result.session_id) is None


def test_persist_session_empty_self_report_not_stored(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    result = persist_session(
        conn, prep, _make_participant(), _make_session(), SelfReportData(),
    )
    assert get_self_report(conn, result.session_id) is None


# ---------------------------------------------------------------------------
# Participante já existente
# ---------------------------------------------------------------------------
def test_persist_session_existing_participant_reuses_id(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    r1 = persist_session(conn, prep, _make_participant(), _make_session("V1"))
    r2 = persist_session(conn, prep, _make_participant(), _make_session("V2"))

    assert r1.participant_id == r2.participant_id
    assert r2.participant_was_new is False


def test_persist_session_updates_non_null_traits(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)

    # Primeira gravação: define raiva e medo
    persist_session(
        conn, prep,
        ParticipantData(code="P001", trait_anger=3.0, trait_fear=4.0),
        _make_session("V1"),
    )
    # Segunda: muda raiva, deixa medo em None (não deve apagar)
    persist_session(
        conn, prep,
        ParticipantData(code="P001", trait_anger=5.0),
        _make_session("V2"),
    )

    p = get_participant_by_code(conn, "P001")
    assert p['trait_anger'] == 5.0
    assert p['trait_fear'] == 4.0  # preservado


# ---------------------------------------------------------------------------
# Vídeo duplicado
# ---------------------------------------------------------------------------
def test_persist_session_duplicate_video_suffix(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    r1 = persist_session(conn, prep, _make_participant(), _make_session("V1"),
                         on_duplicate_video='suffix')
    r2 = persist_session(conn, prep, _make_participant(), _make_session("V1"),
                         on_duplicate_video='suffix')

    assert r1.final_video_id == "V1"
    assert r2.final_video_id == "V1_v2"
    assert len(list_sessions(conn, participant_id=r1.participant_id)) == 2


def test_persist_session_duplicate_video_replace(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    r1 = persist_session(conn, prep, _make_participant(), _make_session("V1"),
                         on_duplicate_video='replace')
    r2 = persist_session(conn, prep, _make_participant(), _make_session("V1"),
                         on_duplicate_video='replace')

    assert r2.final_video_id == "V1"
    assert r2.session_id != r1.session_id
    assert get_session(conn, r1.session_id) is None  # sessão antiga deletada
    assert len(list_sessions(conn, participant_id=r1.participant_id)) == 1


def test_persist_session_duplicate_video_fail(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    persist_session(conn, prep, _make_participant(), _make_session("V1"),
                    on_duplicate_video='fail')
    with pytest.raises(sqlite3.IntegrityError):
        persist_session(conn, prep, _make_participant(), _make_session("V1"),
                        on_duplicate_video='fail')


def test_persist_session_invalid_on_duplicate_mode(
    conn: sqlite3.Connection, sample_csv: Path
) -> None:
    prep = prepare_session(sample_csv)
    with pytest.raises(ValueError, match="on_duplicate_video"):
        persist_session(conn, prep, _make_participant(), _make_session("V1"),
                        on_duplicate_video='delete_everything')


# ---------------------------------------------------------------------------
# save_csv_copy
# ---------------------------------------------------------------------------
def test_save_csv_copy_from_path(tmp_path: Path, sample_csv: Path) -> None:
    storage = tmp_path / "storage"
    dest = save_csv_copy(sample_csv, storage, "P001", "V1")

    assert dest.exists()
    assert dest.parent.name == "P001"
    assert dest.name == "V1.csv"
    assert dest.read_bytes() == sample_csv.read_bytes()


def test_save_csv_copy_from_bytes(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    dest = save_csv_copy(b"col1,col2\n1,2\n", storage, "P001", "V1")
    assert dest.read_text() == "col1,col2\n1,2\n"


def test_save_csv_copy_sanitizes_name(tmp_path: Path) -> None:
    """Caracteres inválidos para sistema de arquivos viram '_'."""
    dest = save_csv_copy(b"x", tmp_path, "P/001", "V*1?")
    assert dest.parent.name == "P_001"
    assert dest.name == "V_1_.csv"


def test_save_csv_copy_accent_handling(tmp_path: Path) -> None:
    """Acentos brasileiros são preservados."""
    dest = save_csv_copy(b"x", tmp_path, "Participação", "vídeo")
    assert "Participação" in str(dest)
    assert "vídeo" in dest.name
