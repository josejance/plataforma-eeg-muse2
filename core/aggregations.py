"""Construção da tabela mestra: 1 linha por sessão com médias dos 8 índices.

A tabela é o substrato para todas as análises agregadas (boxplots, correlações,
scatter). Cada linha cola dados do participante + da sessão + autorrelato +
médias dos índices EEG. Vídeos e participantes ficam expostos como categorias.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from db.queries import INDEX_COLUMNS


# Traços auto-relatados — pontuação numérica calculada a partir das respostas
# do instrumento; um valor por participante, replicado em todas as sessões dele.
TRAIT_COLUMNS = [
    'trait_anger', 'trait_fear', 'trait_stress',
    'trait_narcissism', 'trait_humility', 'trait_mysticism',
    'trait_habits',
]

TRAIT_LABELS = {
    'trait_anger': 'Raiva',
    'trait_fear': 'Medo',
    'trait_stress': 'Estresse',
    'trait_narcissism': 'Narcisismo',
    'trait_humility': 'Humildade intelectual',
    'trait_mysticism': 'Misticismo',
    'trait_habits': 'Hábitos',
}

SELF_REPORT_NUMERIC = [
    'alegria_intensity', 'medo_raiva_intensity',
    'tristeza_intensity', 'serenidade_intensity',
    'alegria_seconds', 'medo_raiva_seconds',
    'tristeza_seconds', 'serenidade_seconds',
]

SELF_REPORT_CATEGORICAL = ['concordance', 'veracity', 'sharing_intent']

INDEX_MEAN_COLUMNS = [f"{c}_mean" for c in INDEX_COLUMNS]


def build_master_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Devolve uma linha por sessão com todos os dados denormalizados.

    As médias dos 8 índices vêm diretamente das colunas escalares
    ``sessions.<index>_mean`` (gravadas pelo ``save_indices`` ou
    ``update_session_means``). Sessões sem dado aparecem com NaN.
    """
    indices_select = ", ".join(f"s.{c}_mean" for c in INDEX_COLUMNS)
    sql = f"""
        SELECT
            p.id   AS participant_id,
            p.code AS participant_code,
            p.gender, p.age, p.political_position,
            p.trait_anger, p.trait_fear, p.trait_stress,
            p.trait_narcissism, p.trait_humility, p.trait_mysticism,
            p.trait_habits,
            s.id   AS session_id,
            s.video_id, s.video_type, s.video_duration_expected,
            s.quality_score, s.n_blinks_per_min,
            s.n_samples_valid, s.n_samples_total,
            sr.alegria_intensity, sr.medo_raiva_intensity,
            sr.tristeza_intensity, sr.serenidade_intensity,
            sr.alegria_seconds, sr.medo_raiva_seconds,
            sr.tristeza_seconds, sr.serenidade_seconds,
            sr.concordance, sr.veracity, sr.sharing_intent,
            {indices_select}
        FROM sessions s
        JOIN participants p ON s.participant_id = p.id
        LEFT JOIN self_reports sr ON sr.session_id = s.id
        ORDER BY p.code, s.video_id
    """
    df = pd.read_sql_query(sql, conn)
    if not df.empty:
        df['age_group'] = df['age'].apply(age_group)
    else:
        df['age_group'] = pd.Series(dtype='object')
    return df


def age_group(age: Optional[float]) -> Optional[str]:
    """Mapeia idade contínua em faixas etárias categóricas."""
    if age is None or pd.isna(age):
        return None
    age = int(age)
    if age < 18:
        return '<18'
    if age < 25:
        return '18-24'
    if age < 35:
        return '25-34'
    if age < 45:
        return '35-44'
    if age < 55:
        return '45-54'
    return '55+'


def aggregate_timeseries(
    conn,
    video_id: str,
    participant_codes: list[str],
    index_col: str,
) -> pd.DataFrame:
    """Agrega a série temporal de um índice por janela, para um conjunto de participantes em um vídeo.

    Combina sessões diferentes alinhando pelo ``t_window`` (todas as sessões usam
    janelas de 5 s com passo de 2,5 s, então os tempos coincidem). Devolve:
        t_window | mean | sd | sem | n | ci_lo | ci_hi

    Onde ``ci_lo``/``ci_hi`` é o IC 95% (mean ± 1.96 × SE).
    """
    cols_validas = {'atencao', 'eng_cognitivo', 'eng_afetivo', 'evocacao',
                    'aderencia', 'faa', 'arousal', 'estresse'}
    if index_col not in cols_validas:
        raise ValueError(f"index_col deve ser um de {cols_validas}, recebeu {index_col!r}")

    if not participant_codes:
        return pd.DataFrame(columns=['t_window', 'mean', 'sd', 'sem', 'n', 'ci_lo', 'ci_hi'])

    placeholders = ','.join(['?'] * len(participant_codes))
    sql = f"""
        SELECT ei.t_window, ei.{index_col} AS valor
        FROM eeg_indices ei
        JOIN sessions s ON ei.session_id = s.id
        JOIN participants p ON s.participant_id = p.id
        WHERE s.video_id = ? AND p.code IN ({placeholders})
    """
    df = pd.read_sql_query(sql, conn, params=[video_id, *participant_codes])
    if df.empty:
        return pd.DataFrame(columns=['t_window', 'mean', 'sd', 'sem', 'n', 'ci_lo', 'ci_hi'])

    grouped = df.groupby('t_window')['valor'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    grouped.columns = ['t_window', 'mean', 'sd', 'n']
    grouped['sem'] = grouped['sd'] / grouped['n'].pow(0.5)
    grouped['ci_lo'] = grouped['mean'] - 1.96 * grouped['sem']
    grouped['ci_hi'] = grouped['mean'] + 1.96 * grouped['sem']
    return grouped[['t_window', 'mean', 'sd', 'sem', 'n', 'ci_lo', 'ci_hi']]


def apply_filters(
    df: pd.DataFrame,
    genders: Optional[list[str]] = None,
    age_groups: Optional[list[str]] = None,
    political_positions: Optional[list[str]] = None,
    video_ids: Optional[list[str]] = None,
    concordances: Optional[list[str]] = None,
    veracities: Optional[list[str]] = None,
    sharing_intents: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aplica filtros multi-seleção sobre a tabela mestra.

    Os filtros se acumulam (AND). Listas vazias / None são ignoradas.
    """
    out = df
    if genders:
        out = out[out['gender'].isin(genders)]
    if age_groups:
        out = out[out['age_group'].isin(age_groups)]
    if political_positions:
        out = out[out['political_position'].isin(political_positions)]
    if video_ids:
        out = out[out['video_id'].isin(video_ids)]
    if concordances:
        out = out[out['concordance'].isin(concordances)]
    if veracities:
        out = out[out['veracity'].isin(veracities)]
    if sharing_intents:
        out = out[out['sharing_intent'].isin(sharing_intents)]
    return out
