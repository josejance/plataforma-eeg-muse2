"""Testes da exportação Jamovi."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.aggregations import (
    INDEX_MEAN_COLUMNS,
    TRAIT_COLUMNS,
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
from exports.jamovi import (
    build_long,
    build_metadata,
    build_wide,
    to_csv_zip_bytes,
    to_xlsx_bytes,
)


@pytest.fixture
def populated_master(tmp_path: Path) -> pd.DataFrame:
    """Banco com 2 participantes × 4 vídeos."""
    conn = init_db(tmp_path / "jamovi.db")
    rng = np.random.default_rng(0)

    for code in ['P001', 'P002']:
        pid = create_participant(
            conn, code=code,
            gender='feminino' if code == 'P001' else 'masculino',
            age=30 if code == 'P001' else 40,
            political_position='centro',
            trait_fear=5.0, trait_anger=4.0, trait_stress=6.0,
            trait_narcissism=3.0, trait_humility=7.0, trait_mysticism=4.0,
            trait_habits=5.5,
        )
        for vid in ['V1', 'V2', 'V3', 'V4']:
            sid = create_session(conn, pid, vid, video_duration_expected=120.0,
                                 quality_score=0.9, n_blinks_per_min=10.0,
                                 n_samples_valid=300, n_samples_total=320)
            df = pd.DataFrame({
                't_window': np.arange(20) * 2.5,
                **{c: rng.normal(1.0, 0.3, 20) for c in INDEX_COLUMNS},
            })
            save_indices(conn, sid, df)
            upsert_self_report(
                conn, sid,
                alegria_intensity=3.0, medo_raiva_intensity=7.0,
                tristeza_intensity=2.0, serenidade_intensity=1.0,
                concordance='Concordo' if vid == 'V1' else 'Não concordo',
                veracity='Verdadeiro' if vid == 'V2' else 'Mentiroso',
                sharing_intent='Não compartilharia esse vídeo',
            )
    return build_master_table(conn)


# ----------- Long format -----------
def test_build_long_one_row_per_session(populated_master) -> None:
    long_df = build_long(populated_master)
    assert len(long_df) == 8  # 2 participantes × 4 vídeos


def test_build_long_strips_mean_suffix(populated_master) -> None:
    long_df = build_long(populated_master)
    for c in INDEX_MEAN_COLUMNS:
        clean = c[:-len('_mean')]
        assert clean in long_df.columns, f'falta {clean}'
        assert c not in long_df.columns, f'_mean ainda em {c}'


def test_build_long_drops_internal_ids(populated_master) -> None:
    long_df = build_long(populated_master)
    assert 'participant_id' not in long_df.columns
    assert 'session_id' not in long_df.columns
    assert 'participant_code' in long_df.columns


def test_build_long_replicates_traits(populated_master) -> None:
    """Traços do participante aparecem em todas as linhas do mesmo participante."""
    long_df = build_long(populated_master)
    p1 = long_df[long_df['participant_code'] == 'P001']
    assert p1['trait_fear'].nunique() == 1
    assert p1['trait_fear'].iloc[0] == 5.0


# ----------- Wide format -----------
def test_build_wide_one_row_per_participant(populated_master) -> None:
    wide = build_wide(populated_master)
    assert len(wide) == 2  # 2 participantes


def test_build_wide_has_per_video_index_columns(populated_master) -> None:
    wide = build_wide(populated_master)
    for vid in ['V1', 'V2', 'V3', 'V4']:
        assert f'atencao_{vid}' in wide.columns
        assert f'eng_afetivo_{vid}' in wide.columns
        assert f'faa_{vid}' in wide.columns


def test_build_wide_has_per_video_self_report(populated_master) -> None:
    wide = build_wide(populated_master)
    for vid in ['V1', 'V2', 'V3', 'V4']:
        assert f'concordance_{vid}' in wide.columns
        assert f'alegria_intensity_{vid}' in wide.columns


def test_build_wide_has_overall_means(populated_master) -> None:
    wide = build_wide(populated_master)
    for c in ['atencao', 'eng_cognitivo', 'eng_afetivo', 'faa', 'estresse']:
        assert f'{c}_mean' in wide.columns


def test_build_wide_preserves_traits(populated_master) -> None:
    wide = build_wide(populated_master)
    for trait in TRAIT_COLUMNS:
        assert trait in wide.columns
    p1 = wide[wide['participant_code'] == 'P001'].iloc[0]
    assert p1['trait_fear'] == 5.0
    assert p1['trait_habits'] == 5.5


def test_build_wide_values_match_per_video(populated_master) -> None:
    """O valor pivotado por vídeo deve bater com a linha original do long."""
    wide = build_wide(populated_master)
    long_df = build_long(populated_master)

    p1_long = long_df[long_df['participant_code'] == 'P001']
    expected_v1 = p1_long[p1_long['video_id'] == 'V1']['atencao'].iloc[0]
    p1_wide = wide[wide['participant_code'] == 'P001'].iloc[0]
    assert p1_wide['atencao_V1'] == pytest.approx(expected_v1)


# ----------- Metadata -----------
def test_build_metadata_lists_all_columns() -> None:
    meta = build_metadata()
    described = set(meta['variable'])
    must_have = {
        'participant_code', 'gender', 'age', 'political_position', 'age_group',
        *TRAIT_COLUMNS,
        'video_id', 'video_duration_expected', 'quality_score',
        'atencao', 'eng_cognitivo', 'eng_afetivo', 'faa', 'arousal', 'estresse',
        'alegria_intensity', 'medo_raiva_intensity',
        'concordance', 'veracity', 'sharing_intent',
    }
    missing = must_have - described
    assert not missing, f'variáveis sem descrição: {missing}'


def test_build_metadata_has_required_columns() -> None:
    meta = build_metadata()
    assert list(meta.columns) == ['variable', 'description', 'type', 'scale_or_values']


# ----------- Empacotamento -----------
def test_to_xlsx_bytes_returns_valid_excel(populated_master) -> None:
    wide = build_wide(populated_master)
    long_df = build_long(populated_master)
    meta = build_metadata()
    xlsx = to_xlsx_bytes({'wide': wide, 'long': long_df, 'metadata': meta})

    # Magic bytes do XLSX (ZIP-based: PK\x03\x04)
    assert xlsx[:4] == b'PK\x03\x04'
    # Roundtrip: pandas consegue ler as 3 abas
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert set(sheets.keys()) == {'wide', 'long', 'metadata'}
    assert len(sheets['wide']) == len(wide)
    assert len(sheets['long']) == len(long_df)


def test_to_csv_zip_bytes_returns_valid_zip(populated_master) -> None:
    wide = build_wide(populated_master)
    long_df = build_long(populated_master)
    meta = build_metadata()
    zb = to_csv_zip_bytes({'wide': wide, 'long': long_df, 'metadata': meta})

    assert zb[:4] == b'PK\x03\x04'
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        names = set(zf.namelist())
        assert names == {'wide.csv', 'long.csv', 'metadata.csv'}
        wide_csv = zf.read('wide.csv').decode('utf-8-sig')
        # BOM removido pelo decode utf-8-sig, mas presente nos bytes
        assert zf.read('wide.csv')[:3] == bytes([0xEF, 0xBB, 0xBF])
        assert 'atencao_V1' in wide_csv


def test_empty_master_builds_empty_wide_and_long(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "empty.db")
    master = build_master_table(conn)
    wide = build_wide(master)
    long_df = build_long(master)
    assert len(wide) == 0
    assert len(long_df) == 0
