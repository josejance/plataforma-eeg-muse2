"""Importa cadastro + respostas pós-vídeo a partir de um CSV agregado.

O CSV tem uma linha por (participante × vídeo). Para cada linha:
  1. Atualiza (ou cria) o participante com os 7 traços + demografia.
  2. Cria ou atualiza a sessão (participant_id, video_id):
     - Se a sessão JÁ tem CSV EEG importado (``file_path`` não-nulo), preserva
       todo o lado EEG — apenas atualiza/cria o autorrelato.
     - Se não existe, cria uma sessão "esqueleto" sem EEG, com
       ``video_duration_expected`` preenchido do dicionário VIDEO_DURATIONS.
  3. Faz upsert do autorrelato com as 3 categóricas do CSV
     (concordância, veracidade, compartilhamento).

Idempotente: pode rodar várias vezes; o estado final reflete o CSV.

Mapeamento das colunas:
    participante           -> code
    Gênero                 -> gender
    Idade                  -> age
    Posicionamento Político -> political_position
    Medo                   -> trait_fear
    Raiva                  -> trait_anger
    Estresse               -> trait_stress
    Narcisismo             -> trait_narcissism
    Humildade Intelectual  -> trait_humility
    Misticismo             -> trait_mysticism
    Habitos                -> trait_habits
    Identificador do Vídeo -> session.video_id
    Concordância           -> self_report.concordance
    Veracidade             -> self_report.veracity
    Compartilhamento       -> self_report.sharing_intent
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
    create_participant,
    create_session,
    get_participant_by_code,
    list_sessions,
    update_participant,
    upsert_self_report,
)
from db.schema import get_connection, init_db  # noqa: E402


PARTICIPANT_MAP = {
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

TRAIT_FIELDS = [
    'trait_fear', 'trait_anger', 'trait_stress',
    'trait_narcissism', 'trait_humility', 'trait_mysticism', 'trait_habits',
]


def _to_float(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(',', '.')
    return None if not s or s.lower() == 'nan' else float(s)


def _to_int(v):
    return None if pd.isna(v) else int(float(v))


def _to_str(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def import_csv(csv_path: Path, conn) -> dict:
    df = pd.read_csv(csv_path, decimal=',', encoding='utf-8')
    required = list(PARTICIPANT_MAP.keys()) + [
        'participante', 'Identificador do Vídeo',
        'Concordância', 'Veracidade', 'Compartilhamento',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    stats = {
        'linhas_csv': len(df),
        'participantes_criados': 0,
        'participantes_atualizados': 0,
        'sessoes_criadas': 0,
        'sessoes_preservadas': 0,
        'autorrelatos_atualizados': 0,
        'erros': [],
    }

    # Dedup participantes por code para reduzir UPDATEs (CSV tem 4 linhas por p.)
    participant_ids: dict[str, int] = {}
    seen_codes: set[str] = set()

    for _, row in df.iterrows():
        code = _to_str(row['participante'])
        if not code:
            continue

        # 1. Upsert participante (uma vez por code)
        if code not in seen_codes:
            seen_codes.add(code)
            p_kwargs = {
                'gender': _to_str(row.get('Gênero')),
                'age': _to_int(row.get('Idade')),
                'political_position': _to_str(row.get('Posicionamento Político')),
            }
            for orig_col, db_col in PARTICIPANT_MAP.items():
                if db_col.startswith('trait_'):
                    p_kwargs[db_col] = _to_float(row.get(orig_col))
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

        # 2. Sessão por vídeo: preserva EEG já importado
        video_id = _to_str(row.get('Identificador do Vídeo'))
        if not video_id:
            continue

        try:
            existing_sessions = list_sessions(conn, participant_id=pid, video_id=video_id)
            if existing_sessions:
                sid = existing_sessions[0]['id']
                stats['sessoes_preservadas'] += 1
            else:
                sid = create_session(
                    conn, participant_id=pid, video_id=video_id,
                    video_duration_expected=VIDEO_DURATIONS.get(video_id),
                )
                stats['sessoes_criadas'] += 1

            # 3. Autorrelato (apenas as 3 categóricas — emoções ficam para a UI)
            upsert_self_report(
                conn, sid,
                concordance=_to_str(row.get('Concordância')),
                veracity=_to_str(row.get('Veracidade')),
                sharing_intent=_to_str(row.get('Compartilhamento')),
            )
            stats['autorrelatos_atualizados'] += 1
        except Exception as exc:  # noqa: BLE001
            stats['erros'].append((f'{code}/{video_id}', f'session: {exc}'))

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

    print(f"Linhas no CSV               : {stats['linhas_csv']}")
    print(f"Participantes criados       : {stats['participantes_criados']}")
    print(f"Participantes atualizados   : {stats['participantes_atualizados']}")
    print(f"Sessões criadas (esqueleto) : {stats['sessoes_criadas']}")
    print(f"Sessões preservadas (já EEG): {stats['sessoes_preservadas']}")
    print(f"Autorrelatos gravados       : {stats['autorrelatos_atualizados']}")
    print(f"Erros                       : {len(stats['erros'])}")
    for code, msg in stats['erros']:
        print(f"  - {code}: {msg}")
    return 0 if not stats['erros'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
