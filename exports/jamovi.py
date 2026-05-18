"""Exportação para Jamovi nos formatos wide e long.

- Wide: uma linha por participante; cada índice/autorrelato vira N colunas
  (uma por vídeo) mais a média entre vídeos.
- Long: uma linha por (participante × vídeo); traços do participante são
  replicados; ideal para modelos mistos.
- Metadata: descrição de cada variável (tipo, escala, valores possíveis) —
  abre em uma aba separada do XLSX ou como CSV avulso.

Empacotamento em XLSX (3 abas) e ZIP (3 CSVs) — Jamovi lê ambos.
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict

import pandas as pd

from core.aggregations import (
    INDEX_MEAN_COLUMNS,
    SELF_REPORT_CATEGORICAL,
    SELF_REPORT_NUMERIC,
    TRAIT_COLUMNS,
)


# Colunas que pertencem ao participante (não variam por vídeo)
PARTICIPANT_STATIC = (
    ['participant_code', 'gender', 'age', 'age_group', 'political_position']
    + TRAIT_COLUMNS
)

# Por sessão (vídeo): colunas numéricas que viram pivot wide
PER_VIDEO_NUMERIC_EXTRA = [
    'quality_score', 'n_blinks_per_min', 'n_samples_valid', 'n_samples_total',
]


def _strip_mean_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Remove o sufixo ``_mean`` das colunas de índice (já agregadas por sessão)."""
    rename_map = {c: c[: -len('_mean')] for c in INDEX_MEAN_COLUMNS if c in df.columns}
    return df.rename(columns=rename_map)


def build_long(master_df: pd.DataFrame) -> pd.DataFrame:
    """Formato long: 1 linha por sessão. Sufixos ``_mean`` removidos.

    A coluna ``participant_id`` (interna do banco) é descartada — fica só
    ``participant_code`` como identificador externo.
    """
    df = master_df.drop(columns=['participant_id', 'session_id'], errors='ignore').copy()
    df = _strip_mean_suffix(df)
    # Reordena com participant_code primeiro, video_id depois dos identificadores
    front = [c for c in
             ['participant_code', 'gender', 'age', 'age_group', 'political_position']
             if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def build_wide(master_df: pd.DataFrame) -> pd.DataFrame:
    """Formato wide: 1 linha por participante.

    Colunas pivotadas:
        <métrica>_<video_id>  para índices EEG e intensidades/segundos
        concordance_<video_id>, veracity_<video_id>, sharing_intent_<video_id>
        <índice>_mean  para a média entre vídeos (1 valor por participante)
    """
    df = _strip_mean_suffix(master_df.copy())
    index_cols_clean = [c[: -len('_mean')] for c in INDEX_MEAN_COLUMNS]

    # 1. Estáticos do participante (1 linha por code)
    static = (
        df[PARTICIPANT_STATIC]
        .drop_duplicates(subset='participant_code')
        .reset_index(drop=True)
    )

    # 2. Pivot numérico por vídeo
    numeric_pivot_cols = [
        c for c in index_cols_clean + SELF_REPORT_NUMERIC + PER_VIDEO_NUMERIC_EXTRA
        if c in df.columns
    ]
    pivot_num = df.pivot_table(
        index='participant_code', columns='video_id',
        values=numeric_pivot_cols, aggfunc='first',
    )
    pivot_num.columns = [f"{metric}_{vid}" for metric, vid in pivot_num.columns]
    pivot_num = pivot_num.reset_index()

    # 3. Pivot categórico por vídeo (concord, verac, sharing)
    cat_cols = [c for c in SELF_REPORT_CATEGORICAL if c in df.columns]
    if cat_cols:
        pivot_cat = df.pivot_table(
            index='participant_code', columns='video_id',
            values=cat_cols, aggfunc='first',
        )
        pivot_cat.columns = [f"{metric}_{vid}" for metric, vid in pivot_cat.columns]
        pivot_cat = pivot_cat.reset_index()
    else:
        pivot_cat = pd.DataFrame({'participant_code': df['participant_code'].unique()})

    # 4. Médias entre os 4 vídeos para cada índice
    overall = df.groupby('participant_code')[index_cols_clean].mean(numeric_only=True)
    overall.columns = [f"{c}_mean" for c in overall.columns]
    overall = overall.reset_index()

    wide = (
        static
        .merge(pivot_num, on='participant_code', how='left')
        .merge(pivot_cat, on='participant_code', how='left')
        .merge(overall, on='participant_code', how='left')
    )
    return wide


METADATA_ROWS: list[tuple[str, str, str, str]] = [
    # (variable, description, type, scale_or_values)
    ('participant_code',     'Identificador único do participante',                'nominal',     '-'),
    ('gender',               'Gênero auto-declarado',                              'nominal',     'feminino/masculino/não-binário/prefere não informar'),
    ('age',                  'Idade em anos',                                      'continuous',  '0–120'),
    ('age_group',            'Faixa etária categórica',                            'nominal',     '<18, 18-24, 25-34, 35-44, 45-54, 55+'),
    ('political_position',   'Posição política auto-declarada',                    'nominal',     'esquerda/centro-esquerda/centro/centro-direita/direita/prefere não informar'),
    ('trait_fear',           'Escore Medo (instrumento Likert)',                   'continuous',  '0–10 (decimal)'),
    ('trait_anger',          'Escore Raiva',                                       'continuous',  '0–10 (decimal)'),
    ('trait_stress',         'Escore Estresse',                                    'continuous',  '0–10 (decimal)'),
    ('trait_narcissism',     'Escore Narcisismo',                                  'continuous',  '0–10 (decimal)'),
    ('trait_humility',       'Escore Humildade Intelectual',                       'continuous',  '0–10 (decimal)'),
    ('trait_mysticism',      'Escore Misticismo',                                  'continuous',  '0–10 (decimal)'),
    ('trait_habits',         'Escore Hábitos',                                     'continuous',  '0–10 (decimal)'),
    ('video_id',             'Identificador do vídeo apresentado',                 'nominal',     'V1, V2, V3, V4'),
    ('video_type',           'Tipo de conteúdo do vídeo (texto livre)',            'nominal',     '-'),
    ('video_duration_expected', 'Duração esperada do vídeo (s)',                   'continuous',  '> 0 segundos'),
    ('quality_score',        'Qualidade do sinal EEG (0=ruim, 1=ótimo)',           'continuous',  '0–1'),
    ('n_blinks_per_min',     'Taxa de piscadas detectadas por minuto',             'continuous',  '≥ 0'),
    ('n_samples_valid',      'Amostras EEG válidas após filtro HSI',               'integer',     '≥ 0'),
    ('n_samples_total',      'Amostras EEG totais lidas',                          'integer',     '≥ 0'),
    ('atencao',              'Atenção: β / (α+θ) frontal (Pope et al. 1995)',      'continuous',  '> 0'),
    ('eng_cognitivo',        'Engajamento cognitivo: (β+γ)/α frontal',             'continuous',  '> 0'),
    ('eng_afetivo',          'Engajamento afetivo: |FAA| + (β+γ)/α frontal',       'continuous',  '≥ 0'),
    ('evocacao',             'Evocação de memórias: θ posterior absoluto',         'continuous',  '> 0'),
    ('aderencia',            'Aderência: (γ_F + γ_P) / θ_P',                       'continuous',  '> 0'),
    ('faa',                  'FAA: ln(α_AF8) − ln(α_AF7); sinal preservado',       'continuous',  'real (+ aproximação, − retração)'),
    ('arousal',              'Arousal: β / α total (4 canais)',                    'continuous',  '> 0'),
    ('estresse',             'Estresse: (β/α) + (γ/θ) (Arsalan et al. 2019)',      'continuous',  '> 0'),
    ('alegria_intensity',    'Intensidade autorrelatada de Alegria pós-vídeo',     'continuous',  '0–10'),
    ('medo_raiva_intensity', 'Intensidade autorrelatada de Medo/Raiva pós-vídeo',  'continuous',  '0–10'),
    ('tristeza_intensity',   'Intensidade autorrelatada de Tristeza pós-vídeo',    'continuous',  '0–10'),
    ('serenidade_intensity', 'Intensidade autorrelatada de Serenidade pós-vídeo',  'continuous',  '0–10'),
    ('alegria_seconds',      'Tempo (s) em que sentiu Alegria durante o vídeo',    'continuous',  '≥ 0'),
    ('medo_raiva_seconds',   'Tempo (s) em que sentiu Medo/Raiva',                 'continuous',  '≥ 0'),
    ('tristeza_seconds',     'Tempo (s) em que sentiu Tristeza',                   'continuous',  '≥ 0'),
    ('serenidade_seconds',   'Tempo (s) em que sentiu Serenidade',                 'continuous',  '≥ 0'),
    ('concordance',          'Concordância com o conteúdo do vídeo',               'nominal',     'Concordo, Não concordo, Indiferente'),
    ('veracity',             'Veracidade percebida do conteúdo',                   'nominal',     'Verdadeiro, Mentiroso, Não sei'),
    ('sharing_intent',       'Intenção de compartilhamento',                       'nominal',     'Compartilharia esse vídeo, Não compartilharia esse vídeo, Talvez, Prefere não responder'),
]


def build_metadata() -> pd.DataFrame:
    """DataFrame com descrição de cada variável."""
    return pd.DataFrame(
        METADATA_ROWS,
        columns=['variable', 'description', 'type', 'scale_or_values'],
    )


def to_xlsx_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Empacota os DataFrames em um único XLSX, uma aba cada.

    Requer ``openpyxl`` instalado.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            # Nome de aba do Excel é limitado a 31 chars
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


def to_csv_zip_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Empacota os DataFrames em um ZIP de CSVs (alternativa sem openpyxl)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in sheets.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False).encode('utf-8-sig'))
    return buf.getvalue()
