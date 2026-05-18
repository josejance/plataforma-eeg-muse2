"""Importa o CSV agregado com os 8 índices EEG já calculados por sessão.

CSV esperado: 1 linha por (participante × vídeo) com colunas:
    Codigo identificador, Gênero, Idade, Posicionamento Político,
    Raiva, Medo, Estresse, Narcisismo, Humildade Intelectual, Misticismo, Habitos,
    Identificador do Vídeo, Concordância, Veracidade, Compartilhamento,
    Atenção, Engajamento Cognitivo, Engajamento Afetivo, Memorias, Aderência,
    faa_mean, arousal_mean, estresse_mean, age_group

Para cada linha:
  1. Atualiza/insere o participante (demografia + 7 traços).
  2. Atualiza/insere a sessão (preservando file_path/quality já existentes).
  3. Faz upsert do autorrelato (apenas as 3 categóricas).
  4. SUBSTITUI a série de índices em ``eeg_indices`` por UMA única linha
     com ``t_window=0`` contendo os 8 valores médios da planilha — assim a
     agregação ``AVG`` sobre essa tabela devolve exatamente esses valores
     em todos os módulos da plataforma (master_table, exports Jamovi etc.).

Idempotente. ``age_group`` da planilha é ignorado (calculado no Python).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from config import DB_PATH, VIDEO_DURATIONS  # noqa: E402
from db.queries import (  # noqa: E402
    INDEX_COLUMNS,
    create_participant,
    create_session,
    get_participant_by_code,
    list_sessions,
    update_participant,
    update_session_means,
    upsert_self_report,
)
from db.schema import get_connection, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Mapeamentos do CSV → schema
# ---------------------------------------------------------------------------
PARTICIPANT_FIELDS = {
    'Gênero': 'gender',
    'Idade': 'age',
    'Posicionamento Político': 'political_position',
    'Medo': 'trait_fear',
    'Raiva': 'trait_anger',
    'Estresse': 'trait_stress',
    'Narcisismo': 'trait_narcissism',
    'Humildade Intelectual': 'trait_humility',
    'Misticismo': 'trait_mysticism',
    'Habitos': 'trait_habits',
}

INDEX_FIELDS = {
    'Atenção': 'atencao',
    'Engajamento Cognitivo': 'eng_cognitivo',
    'Engajamento Afetivo': 'eng_afetivo',
    'Memorias': 'evocacao',
    'Aderência': 'aderencia',
    'faa_mean': 'faa',
    'arousal_mean': 'arousal',
    'estresse_mean': 'estresse',
}

SELF_REPORT_FIELDS = {
    'Concordância': 'concordance',
    'Veracidade': 'veracity',
    'Compartilhamento': 'sharing_intent',
}


def _to_str(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _to_float(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(',', '.')
    return None if not s or s.lower() == 'nan' else float(s)


def _to_int(v):
    return None if pd.isna(v) else int(float(v))


def import_csv(csv_path: Path, conn) -> dict:
    df = pd.read_csv(csv_path)
    required = (
        ['Codigo identificador', 'Identificador do Vídeo']
        + list(PARTICIPANT_FIELDS.keys())
        + list(INDEX_FIELDS.keys())
        + list(SELF_REPORT_FIELDS.keys())
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    stats = {
        'linhas_csv': len(df),
        'participantes_criados': 0,
        'participantes_atualizados': 0,
        'sessoes_criadas': 0,
        'sessoes_preservadas': 0,
        'indices_substituidos': 0,
        'autorrelatos_gravados': 0,
        'erros': [],
    }

    # Dedup participantes (CSV tem 4 linhas por code)
    participant_ids: dict[str, int] = {}
    seen_codes: set[str] = set()

    for _, row in df.iterrows():
        code = _to_str(row['Codigo identificador'])
        if not code:
            continue

        # ---------- 1. Participante ----------
        if code not in seen_codes:
            seen_codes.add(code)
            p_kwargs = {}
            for orig, db_field in PARTICIPANT_FIELDS.items():
                if db_field == 'age':
                    p_kwargs[db_field] = _to_int(row.get(orig))
                elif db_field in ('gender', 'political_position'):
                    p_kwargs[db_field] = _to_str(row.get(orig))
                else:
                    p_kwargs[db_field] = _to_float(row.get(orig))
            try:
                existing = get_participant_by_code(conn, code)
                if existing:
                    update_participant(conn, existing['id'], **p_kwargs)
                    participant_ids[code] = int(existing['id'])
                    stats['participantes_atualizados'] += 1
                else:
                    pid = create_participant(conn, code=code, **p_kwargs)
                    participant_ids[code] = pid
                    stats['participantes_criados'] += 1
            except Exception as exc:  # noqa: BLE001
                stats['erros'].append((code, f'participant: {exc}'))
                continue

        pid = participant_ids.get(code)
        if pid is None:
            continue

        # ---------- 2. Sessão ----------
        video_id = _to_str(row.get('Identificador do Vídeo'))
        if not video_id:
            continue

        try:
            existing = list_sessions(conn, participant_id=pid, video_id=video_id)
            if existing:
                sid = existing[0]['id']
                stats['sessoes_preservadas'] += 1
            else:
                sid = create_session(
                    conn, participant_id=pid, video_id=video_id,
                    video_duration_expected=VIDEO_DURATIONS.get(video_id),
                )
                stats['sessoes_criadas'] += 1

            # ---------- 3. Autorrelato ----------
            sr_kwargs = {
                db_field: _to_str(row.get(orig))
                for orig, db_field in SELF_REPORT_FIELDS.items()
            }
            upsert_self_report(conn, sid, **sr_kwargs)
            stats['autorrelatos_gravados'] += 1

            # ---------- 4. Atualiza médias escalares em sessions ----------
            means = {db_field: _to_float(row.get(orig))
                     for orig, db_field in INDEX_FIELDS.items()}
            # Garante chaves de todos os 8 índices (faltantes ficam None)
            for col in INDEX_COLUMNS:
                means.setdefault(col, None)
            update_session_means(conn, sid, means)
            stats['indices_substituidos'] += 1
        except Exception as exc:  # noqa: BLE001
            stats['erros'].append((f'{code}/{video_id}', f'session/indices: {exc}'))

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--db', type=Path, default=DB_PATH)
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Arquivo não encontrado: {args.csv_path}", file=sys.stderr)
        return 1

    init_db(args.db)
    conn = get_connection(args.db)
    stats = import_csv(args.csv_path, conn)
    conn.close()

    print(f"Linhas no CSV                  : {stats['linhas_csv']}")
    print(f"Participantes criados          : {stats['participantes_criados']}")
    print(f"Participantes atualizados      : {stats['participantes_atualizados']}")
    print(f"Sessões criadas (esqueleto)    : {stats['sessoes_criadas']}")
    print(f"Sessões preservadas (já existem): {stats['sessoes_preservadas']}")
    print(f"Autorrelatos gravados          : {stats['autorrelatos_gravados']}")
    print(f"Índices substituídos           : {stats['indices_substituidos']}")
    print(f"Erros                          : {len(stats['erros'])}")
    for label, msg in stats['erros']:
        print(f"  - {label}: {msg}")
    return 0 if not stats['erros'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
