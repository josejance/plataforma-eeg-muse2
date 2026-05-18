"""Exporta o banco SQLite atual como um único arquivo .db.

Uso:
    python scripts/export_backup.py [--out CAMINHO]

O arquivo gerado pode ser enviado para a app online via:
    Página inicial → 🔧 Administração → 'Restaurar banco a partir de backup'

Apenas o ``.db`` SQLite é exportado (CSVs brutos do Mind Monitor permanecem
locais). O backup contém: 96 participantes, 384 sessões, autorrelatos,
~18.600 linhas de série temporal e todas as médias agregadas.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import DB_PATH  # noqa: E402


def export_backup(out_path: Path | None = None) -> Path:
    """Copia o banco para ``out_path`` (default: ``./eeg_backup_<timestamp>.db``)."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Banco não encontrado: {DB_PATH.resolve()}")

    if out_path is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = Path.cwd() / f"eeg_backup_{stamp}.db"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Usa a API de backup do SQLite — garante DB consistente mesmo se houver
    # processo ativo gravando (ex.: Streamlit aberto). Mais seguro que copy.
    src = sqlite3.connect(str(DB_PATH))
    try:
        dest = sqlite3.connect(str(out_path))
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None,
                        help='Caminho do .db de saída (default: ./eeg_backup_<timestamp>.db)')
    args = parser.parse_args()

    try:
        out = export_backup(args.out)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    size_mb = out.stat().st_size / 1024 / 1024

    # Sumário do conteúdo
    conn = sqlite3.connect(str(out))
    try:
        n_p = conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0]
        n_s = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_sr = conn.execute("SELECT COUNT(*) FROM self_reports").fetchone()[0]
        n_ei = conn.execute("SELECT COUNT(*) FROM eeg_indices").fetchone()[0]
    finally:
        conn.close()

    print(f"Backup criado: {out.resolve()}")
    print(f"Tamanho      : {size_mb:.2f} MB")
    print(f"Conteúdo     : {n_p} participantes · {n_s} sessões · "
          f"{n_sr} autorrelatos · {n_ei:,} linhas de série temporal")
    print()
    print("Para usar na app online:")
    print("  1. Acesse o deploy no Streamlit Cloud")
    print("  2. Página inicial → expander '🔧 Administração'")
    print("  3. Em 'Restaurar banco a partir de backup', suba este arquivo")
    print("  4. Recarregue — todos os dados estarão lá")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
