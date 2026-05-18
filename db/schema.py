"""Schema do banco SQLite e helpers de conexão.

A persistência guarda metadados de participantes/sessões e a série temporal
de índices por janela (tabela ``eeg_indices``) — assim a tela de visualização
não precisa reprocessar o CSV.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

# Tabelas conforme spec. ``IF NOT EXISTS`` permite chamar init_db em qualquer ponto.
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    gender TEXT,
    age INTEGER,
    political_position TEXT,
    trait_anger REAL,
    trait_fear REAL,
    trait_stress REAL,
    trait_narcissism REAL,
    trait_humility REAL,
    trait_mysticism REAL,
    trait_habits REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id INTEGER NOT NULL,
    video_id TEXT NOT NULL,
    video_type TEXT,
    video_duration_expected REAL,
    file_path TEXT,
    csv_filename TEXT,
    quality_score REAL,
    n_blinks_per_min REAL,
    n_samples_valid INTEGER,
    n_samples_total INTEGER,
    atencao_mean REAL,
    eng_cognitivo_mean REAL,
    eng_afetivo_mean REAL,
    evocacao_mean REAL,
    aderencia_mean REAL,
    faa_mean REAL,
    arousal_mean REAL,
    estresse_mean REAL,
    created_at TEXT NOT NULL,
    UNIQUE (participant_id, video_id),
    FOREIGN KEY (participant_id) REFERENCES participants(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS self_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL UNIQUE,
    alegria_intensity REAL,
    medo_raiva_intensity REAL,
    tristeza_intensity REAL,
    serenidade_intensity REAL,
    alegria_seconds REAL,
    medo_raiva_seconds REAL,
    tristeza_seconds REAL,
    serenidade_seconds REAL,
    concordance TEXT,
    veracity TEXT,
    sharing_intent TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eeg_indices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    t_window REAL NOT NULL,
    atencao REAL,
    eng_cognitivo REAL,
    eng_afetivo REAL,
    evocacao REAL,
    aderencia REAL,
    faa REAL,
    arousal REAL,
    estresse REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_participant ON sessions(participant_id);
CREATE INDEX IF NOT EXISTS idx_eeg_indices_session ON eeg_indices(session_id);
"""


def get_connection(path: Union[str, Path]) -> sqlite3.Connection:
    """Abre conexão SQLite com chaves estrangeiras ativas e row_factory de dict."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Union[str, Path]) -> sqlite3.Connection:
    """Cria o banco (ou aplica migrações sobre um banco existente) e devolve conexão."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    conn.executescript(SCHEMA_SQL)
    _apply_migrations(conn)
    conn.commit()
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Aplica migrações idempotentes a bancos criados em versões anteriores.

    Mantém os dados existentes — apenas adiciona colunas que ainda não existam.
    """
    existing_p = {row['name'] for row in conn.execute("PRAGMA table_info(participants)")}
    if 'trait_habits' not in existing_p:
        conn.execute("ALTER TABLE participants ADD COLUMN trait_habits REAL")

    # Médias escalares na tabela sessions (separadas da série temporal em eeg_indices)
    existing_s = {row['name'] for row in conn.execute("PRAGMA table_info(sessions)")}
    for col in [
        'atencao_mean', 'eng_cognitivo_mean', 'eng_afetivo_mean', 'evocacao_mean',
        'aderencia_mean', 'faa_mean', 'arousal_mean', 'estresse_mean',
    ]:
        if col not in existing_s:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} REAL")
