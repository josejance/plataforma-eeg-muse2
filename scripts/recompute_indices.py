"""Recalcula a série de índices de TODAS as sessões com CSV salvo.

Necessário quando uma fórmula muda (por exemplo, eng_afetivo) — atualiza
as linhas em ``eeg_indices`` lendo o CSV original via ``file_path``.

Idempotente: chamável quantas vezes for preciso.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import DB_PATH  # noqa: E402
from core.indices import compute_all_indices  # noqa: E402
from core.parser import MindMonitorParseError, load_csv  # noqa: E402
from core.preprocessing import quality_filter  # noqa: E402
from db.queries import save_indices  # noqa: E402
from db.schema import get_connection, init_db  # noqa: E402


def recompute_all(conn) -> dict:
    rows = conn.execute(
        "SELECT s.id, s.file_path, p.code, s.video_id "
        "FROM sessions s JOIN participants p ON s.participant_id = p.id "
        "WHERE s.file_path IS NOT NULL"
    ).fetchall()

    stats = {'total': len(rows), 'recalculadas': 0, 'puladas': 0, 'erros': []}

    for row in rows:
        sid, file_path, code, video_id = row['id'], row['file_path'], row['code'], row['video_id']
        path = Path(file_path) if file_path else None
        if not path or not path.exists():
            stats['puladas'] += 1
            stats['erros'].append((f'{code}/{video_id}', f'arquivo ausente: {file_path}'))
            continue
        try:
            df = load_csv(path)
            df_filt, _ = quality_filter(df)
            if df_filt.empty:
                stats['puladas'] += 1
                continue
            indices = compute_all_indices(df_filt)
            if indices.empty:
                stats['puladas'] += 1
                continue
            save_indices(conn, sid, indices)
            stats['recalculadas'] += 1
        except (MindMonitorParseError, Exception) as exc:  # noqa: BLE001
            stats['erros'].append((f'{code}/{video_id}', str(exc)))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DB_PATH)
    args = parser.parse_args()

    init_db(args.db)
    conn = get_connection(args.db)
    stats = recompute_all(conn)
    conn.close()

    print(f"Sessões com file_path  : {stats['total']}")
    print(f"Recalculadas           : {stats['recalculadas']}")
    print(f"Puladas (vazio/falta)  : {stats['puladas']}")
    print(f"Erros                  : {len(stats['erros'])}")
    for label, msg in stats['erros']:
        print(f"  - {label}: {msg}")
    return 0 if not stats['erros'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
