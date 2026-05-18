"""Importa cadastro de participantes a partir de um CSV pré-existente.

Mapeia colunas em português para o schema do banco:
    participante                -> code
    Gênero                      -> gender
    Idade                       -> age
    Posicionamento Político     -> political_position
    Medo                        -> trait_fear
    Raiva                       -> trait_anger
    Estresse                    -> trait_stress
    Narcisismo                  -> trait_narcissism
    Humildade Intelectual       -> trait_humility
    Misticismo                  -> trait_mysticism
    Habitos                     -> trait_habits

Idempotente: se o ``code`` já existir, atualiza os campos; senão, cria.
Aceita CSV no formato brasileiro (vírgula decimal, valores numéricos
podem vir com ou sem aspas).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from config import DB_PATH  # noqa: E402
from db.queries import (  # noqa: E402
    create_participant,
    get_participant_by_code,
    update_participant,
)
from db.schema import get_connection, init_db  # noqa: E402


COLUMN_MAP = {
    'participante': 'code',
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


def _to_float(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(',', '.')
    if not s or s.lower() == 'nan':
        return None
    return float(s)


def _to_str(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _to_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(float(value))


def import_csv(csv_path: Path, conn) -> dict:
    df = pd.read_csv(csv_path, decimal=',', encoding='utf-8')
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    df = df.rename(columns=COLUMN_MAP)

    created = 0
    updated = 0
    errors: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        code = _to_str(row['code'])
        if not code:
            continue

        kwargs = {
            'gender': _to_str(row.get('gender')),
            'age': _to_int(row.get('age')),
            'political_position': _to_str(row.get('political_position')),
        }
        for tf in TRAIT_FIELDS:
            kwargs[tf] = _to_float(row.get(tf))

        try:
            existing = get_participant_by_code(conn, code)
            if existing:
                update_participant(conn, existing['id'], **kwargs)
                updated += 1
            else:
                create_participant(conn, code=code, **kwargs)
                created += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((code, str(exc)))

    return {
        'total_no_csv': len(df),
        'criados': created,
        'atualizados': updated,
        'erros': errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path, help='Caminho do CSV de participantes')
    parser.add_argument('--db', type=Path, default=DB_PATH,
                        help='Caminho do banco SQLite (default: config.DB_PATH)')
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Arquivo não encontrado: {args.csv_path}", file=sys.stderr)
        return 1

    init_db(args.db)
    conn = get_connection(args.db)
    result = import_csv(args.csv_path, conn)
    conn.close()

    print(f"Total de linhas no CSV : {result['total_no_csv']}")
    print(f"Criados                : {result['criados']}")
    print(f"Atualizados            : {result['atualizados']}")
    print(f"Erros                  : {len(result['erros'])}")
    for code, msg in result['erros']:
        print(f"  - {code}: {msg}")
    return 0 if not result['erros'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
