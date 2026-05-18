"""Testes do schema SQLite e CRUD."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from db.queries import (
    INDEX_COLUMNS,
    create_participant,
    create_session,
    delete_indices,
    delete_participant,
    delete_session,
    get_indices,
    get_participant,
    get_participant_by_code,
    get_self_report,
    get_session,
    list_participants,
    list_sessions,
    resolve_video_id,
    save_indices,
    update_participant,
    upsert_self_report,
)
from db.schema import init_db


@pytest.fixture
def conn(tmp_path: Path):
    db_path = tmp_path / "test.db"
    c = init_db(db_path)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_init_db_creates_tables(tmp_path: Path) -> None:
    c = init_db(tmp_path / "x.db")
    tables = {r['name'] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {'participants', 'sessions', 'self_reports', 'eeg_indices'} <= tables


def test_init_db_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "x.db"
    init_db(p).close()
    # Segunda chamada não deve quebrar
    c = init_db(p)
    assert c is not None
    c.close()


def test_foreign_keys_enabled(conn: sqlite3.Connection) -> None:
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


# ---------------------------------------------------------------------------
# Participantes
# ---------------------------------------------------------------------------
def test_create_and_get_participant(conn: sqlite3.Connection) -> None:
    pid = create_participant(
        conn, code="P001", gender="feminino", age=24,
        political_position="centro-esquerda",
        trait_anger=3.5, trait_fear=4.0, trait_stress=5.0,
        trait_narcissism=2.0, trait_humility=7.0, trait_mysticism=3.0,
        trait_habits=6.5,
    )
    assert pid >= 1
    p = get_participant(conn, pid)
    assert p is not None
    assert p['code'] == "P001"
    assert p['age'] == 24
    assert p['trait_anger'] == 3.5
    assert p['trait_habits'] == 6.5
    assert p['created_at']  # ISO string presente


def test_migration_adds_trait_habits(tmp_path: Path) -> None:
    """Banco criado em versão anterior (sem trait_habits) deve ganhar a coluna."""
    # Cria schema legado (sem trait_habits)
    legacy = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(legacy))
    raw.execute("""
        CREATE TABLE participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            gender TEXT, age INTEGER, political_position TEXT,
            trait_anger REAL, trait_fear REAL, trait_stress REAL,
            trait_narcissism REAL, trait_humility REAL, trait_mysticism REAL,
            created_at TEXT NOT NULL
        )
    """)
    raw.execute(
        "INSERT INTO participants (code, age, trait_anger, created_at) "
        "VALUES ('P001', 30, 5.0, '2025-01-01T00:00:00')"
    )
    raw.commit()
    raw.close()

    # init_db deve migrar sem perder dados
    c = init_db(legacy)
    cols = {r['name'] for r in c.execute("PRAGMA table_info(participants)")}
    assert 'trait_habits' in cols

    p = c.execute("SELECT * FROM participants WHERE code='P001'").fetchone()
    assert p['age'] == 30
    assert p['trait_anger'] == 5.0
    assert p['trait_habits'] is None  # default
    c.close()


def test_create_participant_duplicate_code(conn: sqlite3.Connection) -> None:
    create_participant(conn, code="P001")
    with pytest.raises(sqlite3.IntegrityError):
        create_participant(conn, code="P001")


def test_get_participant_by_code(conn: sqlite3.Connection) -> None:
    create_participant(conn, code="P042", age=30)
    p = get_participant_by_code(conn, "P042")
    assert p is not None and p['age'] == 30
    assert get_participant_by_code(conn, "inexistente") is None


def test_list_participants_ordered_by_code(conn: sqlite3.Connection) -> None:
    create_participant(conn, code="P003")
    create_participant(conn, code="P001")
    create_participant(conn, code="P002")
    codes = [p['code'] for p in list_participants(conn)]
    assert codes == ["P001", "P002", "P003"]


def test_update_participant(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001", age=20)
    assert update_participant(conn, pid, age=21, trait_fear=4.0) is True
    p = get_participant(conn, pid)
    assert p['age'] == 21 and p['trait_fear'] == 4.0


def test_update_participant_rejects_invalid_field(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    with pytest.raises(ValueError):
        update_participant(conn, pid, campo_inexistente=1)


def test_delete_participant_cascades(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, participant_id=pid, video_id="V1")

    df = pd.DataFrame({'t_window': [2.5], **{c: [1.0] for c in INDEX_COLUMNS}})
    save_indices(conn, sid, df)

    assert delete_participant(conn, pid) is True
    assert get_session(conn, sid) is None
    assert len(get_indices(conn, sid)) == 0


# ---------------------------------------------------------------------------
# Sessões
# ---------------------------------------------------------------------------
def test_create_session_and_list_by_participant(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid1 = create_session(conn, pid, "V1", csv_filename="v1.csv")
    sid2 = create_session(conn, pid, "V2", csv_filename="v2.csv")
    sessions = list_sessions(conn, participant_id=pid)
    assert len(sessions) == 2
    assert {s['id'] for s in sessions} == {sid1, sid2}


def test_create_session_invalid_participant_fk(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        create_session(conn, participant_id=999, video_id="V1")


def test_create_session_duplicate_video_id_raises(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    create_session(conn, pid, "V1")
    with pytest.raises(sqlite3.IntegrityError):
        create_session(conn, pid, "V1")


def test_resolve_video_id_generates_suffix(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    assert resolve_video_id(conn, pid, "V1") == "V1"
    create_session(conn, pid, "V1")
    assert resolve_video_id(conn, pid, "V1") == "V1_v2"
    create_session(conn, pid, "V1_v2")
    assert resolve_video_id(conn, pid, "V1") == "V1_v3"


def test_delete_session(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    assert delete_session(conn, sid) is True
    assert get_session(conn, sid) is None


# ---------------------------------------------------------------------------
# Autorrelatos
# ---------------------------------------------------------------------------
def test_upsert_self_report_insert_then_update(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")

    rid1 = upsert_self_report(
        conn, sid, alegria_intensity=7.0, concordance="Concordo",
    )
    assert rid1 >= 1

    rid2 = upsert_self_report(
        conn, sid, alegria_intensity=9.0, concordance="Não concordo",
        veracity="Verdadeiro",
    )
    assert rid2 == rid1, "upsert deve atualizar a mesma linha"

    r = get_self_report(conn, sid)
    assert r['alegria_intensity'] == 9.0
    assert r['concordance'] == "Não concordo"
    assert r['veracity'] == "Verdadeiro"


def test_upsert_self_report_cascade(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    upsert_self_report(conn, sid, alegria_intensity=5.0)
    delete_session(conn, sid)
    assert get_self_report(conn, sid) is None


# ---------------------------------------------------------------------------
# Série de índices
# ---------------------------------------------------------------------------
def _make_indices_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        't_window': np.arange(n, dtype=float) * 2.5,
        **{c: np.random.RandomState(0).rand(n) for c in INDEX_COLUMNS},
    })


def test_save_and_get_indices_roundtrip(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    df_in = _make_indices_df(20)

    n = save_indices(conn, sid, df_in)
    assert n == 20

    df_out = get_indices(conn, sid)
    assert len(df_out) == 20
    np.testing.assert_allclose(df_out['t_window'].values, df_in['t_window'].values)
    for col in INDEX_COLUMNS:
        np.testing.assert_allclose(df_out[col].values, df_in[col].values)


def test_save_indices_replaces_existing(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    save_indices(conn, sid, _make_indices_df(20))
    save_indices(conn, sid, _make_indices_df(5))
    assert len(get_indices(conn, sid)) == 5


def test_save_indices_missing_column_raises(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    bad = pd.DataFrame({'t_window': [0.0]})  # sem colunas de índice
    with pytest.raises(ValueError, match="Colunas faltando"):
        save_indices(conn, sid, bad)


def test_save_indices_invalid_session_fk(conn: sqlite3.Connection) -> None:
    df = _make_indices_df(3)
    with pytest.raises(sqlite3.IntegrityError):
        save_indices(conn, session_id=99999, indices_df=df)


def test_save_indices_handles_nan(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    df = _make_indices_df(3)
    df.loc[1, 'atencao'] = np.nan
    save_indices(conn, sid, df)
    out = get_indices(conn, sid)
    assert pd.isna(out.loc[1, 'atencao'])


def test_delete_indices(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001")
    sid = create_session(conn, pid, "V1")
    save_indices(conn, sid, _make_indices_df(7))
    assert delete_indices(conn, sid) == 7
    assert len(get_indices(conn, sid)) == 0
