"""Configuração centralizada de logging.

Saída em ``logs/app.log`` com rotação (10 MB × 3 arquivos). Idempotente:
chamar ``setup_logging()`` mais de uma vez não duplica handlers.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from config import LOG_DIR, LOG_FILE


_HANDLER_TAG = "plataforma_eeg_file_handler"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura o logger raiz para gravar em ``logs/app.log``.

    Returns:
        O logger raiz, já configurado.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()

    # Idempotência: se já adicionamos nosso handler, não repete
    for h in root.handlers:
        if getattr(h, '_pe_tag', None) == _HANDLER_TAG:
            return root

    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8',
    )
    handler._pe_tag = _HANDLER_TAG  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s · %(levelname)-7s · %(name)s · %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    root.addHandler(handler)
    if root.level > level:
        root.setLevel(level)
    return root


def get_logger(name: str) -> logging.Logger:
    """Conveniência: garante setup e devolve logger nomeado."""
    setup_logging()
    return logging.getLogger(name)
