"""CRUD de participantes, sessões, autorrelatos e séries de índices."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# Colunas da série temporal salva por sessão
INDEX_COLUMNS = [
    'atencao', 'eng_cognitivo', 'eng_afetivo', 'evocacao',
    'aderencia', 'faa', 'arousal', 'estresse',
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Participantes
# ---------------------------------------------------------------------------
def create_participant(
    conn: sqlite3.Connection,
    code: str,
    gender: Optional[str] = None,
    age: Optional[int] = None,
    political_position: Optional[str] = None,
    trait_anger: Optional[float] = None,
    trait_fear: Optional[float] = None,
    trait_stress: Optional[float] = None,
    trait_narcissism: Optional[float] = None,
    trait_humility: Optional[float] = None,
    trait_mysticism: Optional[float] = None,
    trait_habits: Optional[float] = None,
) -> int:
    """Insere participante; levanta sqlite3.IntegrityError se ``code`` duplicado."""
    cur = conn.execute(
        """INSERT INTO participants (
            code, gender, age, political_position,
            trait_anger, trait_fear, trait_stress, trait_narcissism,
            trait_humility, trait_mysticism, trait_habits, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (code, gender, age, political_position,
         trait_anger, trait_fear, trait_stress, trait_narcissism,
         trait_humility, trait_mysticism, trait_habits, _now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_participant(conn: sqlite3.Connection, participant_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM participants WHERE id = ?", (participant_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_participant_by_code(conn: sqlite3.Connection, code: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM participants WHERE code = ?", (code,)
    ).fetchone()
    return _row_to_dict(row)


def list_participants(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM participants ORDER BY code").fetchall()
    return [dict(r) for r in rows]


def update_participant(conn: sqlite3.Connection, participant_id: int, **fields: Any) -> bool:
    """Atualiza colunas arbitrárias. Levanta ValueError se nenhuma coluna válida."""
    allowed = {
        'code', 'gender', 'age', 'political_position',
        'trait_anger', 'trait_fear', 'trait_stress',
        'trait_narcissism', 'trait_humility', 'trait_mysticism',
        'trait_habits',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        raise ValueError("Nenhuma coluna válida para atualizar.")
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    cur = conn.execute(
        f"UPDATE participants SET {set_clause} WHERE id = ?",
        (*updates.values(), participant_id),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_participant(conn: sqlite3.Connection, participant_id: int) -> bool:
    cur = conn.execute("DELETE FROM participants WHERE id = ?", (participant_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Sessões
# ---------------------------------------------------------------------------
def create_session(
    conn: sqlite3.Connection,
    participant_id: int,
    video_id: str,
    video_type: Optional[str] = None,
    video_duration_expected: Optional[float] = None,
    file_path: Optional[str] = None,
    csv_filename: Optional[str] = None,
    quality_score: Optional[float] = None,
    n_blinks_per_min: Optional[float] = None,
    n_samples_valid: Optional[int] = None,
    n_samples_total: Optional[int] = None,
) -> int:
    """Insere sessão; levanta IntegrityError em (participant_id, video_id) duplicado.

    Use :func:`resolve_video_id` antes para gerar sufixo "_v2" automaticamente.
    """
    cur = conn.execute(
        """INSERT INTO sessions (
            participant_id, video_id, video_type, video_duration_expected,
            file_path, csv_filename, quality_score, n_blinks_per_min,
            n_samples_valid, n_samples_total, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (participant_id, video_id, video_type, video_duration_expected,
         file_path, csv_filename, quality_score, n_blinks_per_min,
         n_samples_valid, n_samples_total, _now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def resolve_video_id(conn: sqlite3.Connection, participant_id: int, base_video_id: str) -> str:
    """Devolve o próximo ID livre: ``base``, ``base_v2``, ``base_v3``..."""
    existing = {
        r['video_id'] for r in conn.execute(
            "SELECT video_id FROM sessions WHERE participant_id = ? AND "
            "(video_id = ? OR video_id LIKE ?)",
            (participant_id, base_video_id, f"{base_video_id}_v%"),
        )
    }
    if base_video_id not in existing:
        return base_video_id
    n = 2
    while f"{base_video_id}_v{n}" in existing:
        n += 1
    return f"{base_video_id}_v{n}"


def get_session(conn: sqlite3.Connection, session_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return _row_to_dict(row)


def list_sessions(
    conn: sqlite3.Connection,
    participant_id: Optional[int] = None,
    video_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []
    if participant_id is not None:
        where.append("participant_id = ?")
        params.append(participant_id)
    if video_id is not None:
        where.append("video_id = ?")
        params.append(video_id)
    sql = "SELECT * FROM sessions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def delete_session(conn: sqlite3.Connection, session_id: int) -> bool:
    cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Autorrelatos
# ---------------------------------------------------------------------------
SELF_REPORT_FIELDS = [
    'alegria_intensity', 'medo_raiva_intensity', 'tristeza_intensity', 'serenidade_intensity',
    'alegria_seconds', 'medo_raiva_seconds', 'tristeza_seconds', 'serenidade_seconds',
    'concordance', 'veracity', 'sharing_intent',
]


def upsert_self_report(conn: sqlite3.Connection, session_id: int, **fields: Any) -> int:
    """Cria ou atualiza o autorrelato de uma sessão. Retorna o ID da linha."""
    payload = {k: fields.get(k) for k in SELF_REPORT_FIELDS}
    cols = ', '.join(['session_id'] + SELF_REPORT_FIELDS)
    placeholders = ', '.join(['?'] * (1 + len(SELF_REPORT_FIELDS)))
    update_clause = ', '.join(f"{k} = excluded.{k}" for k in SELF_REPORT_FIELDS)

    cur = conn.execute(
        f"""INSERT INTO self_reports ({cols}) VALUES ({placeholders})
            ON CONFLICT(session_id) DO UPDATE SET {update_clause}""",
        (session_id, *payload.values()),
    )
    conn.commit()
    # lastrowid não é confiável após UPDATE — busca explícita
    row = conn.execute(
        "SELECT id FROM self_reports WHERE session_id = ?", (session_id,)
    ).fetchone()
    return int(row['id'])


def get_self_report(conn: sqlite3.Connection, session_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM self_reports WHERE session_id = ?", (session_id,)
    ).fetchone()
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Séries de índices por janela
# ---------------------------------------------------------------------------
def save_indices(conn: sqlite3.Connection, session_id: int, indices_df: pd.DataFrame) -> int:
    """Substitui a série de índices da sessão pelo conteúdo de ``indices_df``.

    Após salvar a série, também atualiza as colunas escalares ``<index>_mean``
    na tabela ``sessions`` com a média da série — para que ``build_master_table``
    leia direto e não precise mais agregar por janela.

    Linhas anteriores da mesma sessão são apagadas em transação. Retorna o
    nº de linhas inseridas.
    """
    expected = ['t_window'] + INDEX_COLUMNS
    missing = [c for c in expected if c not in indices_df.columns]
    if missing:
        raise ValueError(f"Colunas faltando em indices_df: {missing}")

    rows = [
        (session_id, float(r['t_window']),
         *[None if pd.isna(r[c]) else float(r[c]) for c in INDEX_COLUMNS])
        for _, r in indices_df.iterrows()
    ]

    means = {c: float(indices_df[c].mean()) if indices_df[c].notna().any() else None
             for c in INDEX_COLUMNS}

    with conn:  # transação automática
        conn.execute("DELETE FROM eeg_indices WHERE session_id = ?", (session_id,))
        cols = 'session_id, t_window, ' + ', '.join(INDEX_COLUMNS)
        placeholders = ', '.join(['?'] * (2 + len(INDEX_COLUMNS)))
        conn.executemany(
            f"INSERT INTO eeg_indices ({cols}) VALUES ({placeholders})",
            rows,
        )
        # Atualiza médias escalares
        mean_cols = [f'{c}_mean' for c in INDEX_COLUMNS]
        set_clause = ', '.join(f'{c} = ?' for c in mean_cols)
        conn.execute(
            f"UPDATE sessions SET {set_clause} WHERE id = ?",
            (*[means[c] for c in INDEX_COLUMNS], session_id),
        )

    return len(rows)


def update_session_means(
    conn: sqlite3.Connection, session_id: int, means: Dict[str, Optional[float]],
) -> bool:
    """Atualiza só as 8 colunas escalares ``<index>_mean`` na tabela sessions.

    Útil para importar valores agregados externos sem mexer em ``eeg_indices``.
    Aceita dict com chaves dos 8 índices (sem sufixo ``_mean``).
    """
    mean_cols = [f'{c}_mean' for c in INDEX_COLUMNS]
    set_clause = ', '.join(f'{c} = ?' for c in mean_cols)
    cur = conn.execute(
        f"UPDATE sessions SET {set_clause} WHERE id = ?",
        (*[means.get(c) for c in INDEX_COLUMNS], session_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_indices(conn: sqlite3.Connection, session_id: int) -> pd.DataFrame:
    """Recupera a série de índices ordenada por ``t_window``."""
    cols = 't_window, ' + ', '.join(INDEX_COLUMNS)
    df = pd.read_sql_query(
        f"SELECT {cols} FROM eeg_indices WHERE session_id = ? ORDER BY t_window",
        conn,
        params=(session_id,),
    )
    return df


def delete_indices(conn: sqlite3.Connection, session_id: int) -> int:
    cur = conn.execute("DELETE FROM eeg_indices WHERE session_id = ?", (session_id,))
    conn.commit()
    return cur.rowcount
