"""Testes da tabela mestra e filtros."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.aggregations import (
    INDEX_MEAN_COLUMNS,
    TRAIT_COLUMNS,
    age_group,
    aggregate_timeseries,
    apply_filters,
    build_master_table,
)
from db.queries import (
    INDEX_COLUMNS,
    create_participant,
    create_session,
    save_indices,
    upsert_self_report,
)
from db.schema import init_db


@pytest.fixture
def conn(tmp_path: Path):
    c = init_db(tmp_path / "agg.db")
    yield c
    c.close()


def _seed_indices(conn: sqlite3.Connection, session_id: int, value: float) -> None:
    df = pd.DataFrame({
        't_window': np.arange(10) * 2.5,
        **{c: np.full(10, value) for c in INDEX_COLUMNS},
    })
    save_indices(conn, session_id, df)


def test_age_group_buckets() -> None:
    assert age_group(None) is None
    assert age_group(17) == '<18'
    assert age_group(18) == '18-24'
    assert age_group(24) == '18-24'
    assert age_group(25) == '25-34'
    assert age_group(34) == '25-34'
    assert age_group(45) == '45-54'
    assert age_group(55) == '55+'
    assert age_group(90) == '55+'


def test_build_master_table_columns(conn: sqlite3.Connection) -> None:
    pid = create_participant(
        conn, code="P001", gender="feminino", age=28,
        political_position="centro",
        trait_anger=4.0, trait_fear=3.0, trait_stress=5.0,
        trait_narcissism=2.0, trait_humility=7.0, trait_mysticism=3.0,
        trait_habits=6.0,
    )
    sid = create_session(conn, pid, "V1", video_type="desinformação")
    _seed_indices(conn, sid, value=1.5)

    df = build_master_table(conn)
    assert len(df) == 1
    expected = {
        'participant_code', 'gender', 'age', 'political_position',
        'video_id', 'video_type', 'session_id', 'age_group',
        *TRAIT_COLUMNS, *INDEX_MEAN_COLUMNS,
    }
    assert expected <= set(df.columns)
    assert df.iloc[0]['age_group'] == '25-34'
    assert df.iloc[0]['atencao_mean'] == pytest.approx(1.5)
    # Todos os 7 traços estão presentes e preenchidos
    assert df.iloc[0]['trait_habits'] == 6.0
    assert df.iloc[0]['trait_anger'] == 4.0
    assert df.iloc[0]['trait_fear'] == 3.0


def test_build_master_table_multiple_sessions(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001", age=30)
    s1 = create_session(conn, pid, "V1")
    s2 = create_session(conn, pid, "V2")
    _seed_indices(conn, s1, value=1.0)
    _seed_indices(conn, s2, value=2.0)

    df = build_master_table(conn)
    assert len(df) == 2
    assert sorted(df['video_id'].tolist()) == ['V1', 'V2']
    means = df.set_index('video_id')['atencao_mean']
    assert means['V1'] == pytest.approx(1.0)
    assert means['V2'] == pytest.approx(2.0)


def test_build_master_table_with_self_report(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001", age=30)
    sid = create_session(conn, pid, "V1")
    _seed_indices(conn, sid, value=1.0)
    upsert_self_report(conn, sid, alegria_intensity=7.0, concordance="Concordo")

    df = build_master_table(conn)
    assert df.iloc[0]['alegria_intensity'] == 7.0
    assert df.iloc[0]['concordance'] == "Concordo"


def test_build_master_table_empty(conn: sqlite3.Connection) -> None:
    df = build_master_table(conn)
    assert df.empty
    assert 'age_group' in df.columns


def test_build_master_table_session_without_indices(conn: sqlite3.Connection) -> None:
    pid = create_participant(conn, code="P001", age=30)
    create_session(conn, pid, "V1")  # sem índices

    df = build_master_table(conn)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]['atencao_mean'])


def test_apply_filters_gender(conn: sqlite3.Connection) -> None:
    p1 = create_participant(conn, code="P1", gender="feminino", age=30)
    p2 = create_participant(conn, code="P2", gender="masculino", age=30)
    s1 = create_session(conn, p1, "V1"); _seed_indices(conn, s1, 1.0)
    s2 = create_session(conn, p2, "V1"); _seed_indices(conn, s2, 2.0)

    df = build_master_table(conn)
    f = apply_filters(df, genders=["feminino"])
    assert len(f) == 1
    assert f.iloc[0]['gender'] == "feminino"


def test_aggregate_timeseries_single_session(conn: sqlite3.Connection) -> None:
    """Com 1 participante, a agregação reproduz a série original."""
    pid = create_participant(conn, code="P1", age=30)
    sid = create_session(conn, pid, "V1")
    _seed_indices(conn, sid, value=2.0)

    agg = aggregate_timeseries(conn, "V1", ["P1"], "atencao")
    assert len(agg) == 10  # 10 janelas
    assert (agg['mean'] == 2.0).all()
    assert (agg['n'] == 1).all()
    # Com n=1, sd é NaN; sem indefinido — apenas mean é confiável
    assert agg['sd'].isna().all() or (agg['sd'] == 0).all()


def test_aggregate_timeseries_multiple_participants(conn: sqlite3.Connection) -> None:
    """Média e SE batem com numpy."""
    p1 = create_participant(conn, code="P1", age=30)
    p2 = create_participant(conn, code="P2", age=30)
    p3 = create_participant(conn, code="P3", age=30)
    _seed_indices(conn, create_session(conn, p1, "V1"), value=1.0)
    _seed_indices(conn, create_session(conn, p2, "V1"), value=2.0)
    _seed_indices(conn, create_session(conn, p3, "V1"), value=3.0)

    agg = aggregate_timeseries(conn, "V1", ["P1", "P2", "P3"], "atencao")
    assert len(agg) == 10
    assert (agg['n'] == 3).all()
    # Media = (1+2+3)/3 = 2.0
    np.testing.assert_allclose(agg['mean'], 2.0)
    # SD = std de [1, 2, 3] com ddof=1 = 1.0
    np.testing.assert_allclose(agg['sd'], 1.0)
    # SE = 1 / sqrt(3) ≈ 0.5774
    np.testing.assert_allclose(agg['sem'], 1.0 / np.sqrt(3))
    # CI = mean ± 1.96 * SE
    np.testing.assert_allclose(agg['ci_lo'], 2.0 - 1.96 / np.sqrt(3))
    np.testing.assert_allclose(agg['ci_hi'], 2.0 + 1.96 / np.sqrt(3))


def test_aggregate_timeseries_empty_codes(conn: sqlite3.Connection) -> None:
    agg = aggregate_timeseries(conn, "V1", [], "atencao")
    assert agg.empty
    assert list(agg.columns) == ['t_window', 'mean', 'sd', 'sem', 'n', 'ci_lo', 'ci_hi']


def test_aggregate_timeseries_invalid_index_raises(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="index_col"):
        aggregate_timeseries(conn, "V1", ["P1"], "indice_inexistente")


def test_aggregate_timeseries_no_matching_data(conn: sqlite3.Connection) -> None:
    """Code existe mas não tem sessão V1."""
    pid = create_participant(conn, code="P1", age=30)
    _seed_indices(conn, create_session(conn, pid, "V2"), value=1.0)
    agg = aggregate_timeseries(conn, "V1", ["P1"], "atencao")
    assert agg.empty


def test_apply_filters_combined(conn: sqlite3.Connection) -> None:
    p1 = create_participant(conn, code="P1", gender="feminino", age=22,
                            political_position="esquerda")
    p2 = create_participant(conn, code="P2", gender="feminino", age=40,
                            political_position="direita")
    s1 = create_session(conn, p1, "V1"); _seed_indices(conn, s1, 1.0)
    s2 = create_session(conn, p2, "V1"); _seed_indices(conn, s2, 2.0)

    df = build_master_table(conn)
    f = apply_filters(df, genders=["feminino"], age_groups=["18-24"])
    assert len(f) == 1
    assert f.iloc[0]['participant_code'] == "P1"
