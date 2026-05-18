"""Testes da configuração de logging."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path


def _purge_pe_handlers() -> None:
    """Remove qualquer handler nosso já anexado ao root (limpa estado entre testes)."""
    import core.logging_setup as ls
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, '_pe_tag', None) == ls._HANDLER_TAG:
            root.removeHandler(h)
            h.close()


def test_setup_logging_creates_file(tmp_path: Path, monkeypatch) -> None:
    """Após setup, mensagens info são escritas no arquivo configurado."""
    import config
    monkeypatch.setattr(config, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(config, 'LOG_FILE', 'test_app.log')

    import core.logging_setup as ls
    importlib.reload(ls)
    _purge_pe_handlers()
    ls.setup_logging()

    logger = logging.getLogger('test.logging.setup')
    logger.info("mensagem-de-teste-única-xyz")

    for h in logging.getLogger().handlers:
        h.flush()

    log_file = tmp_path / 'test_app.log'
    assert log_file.exists()
    text = log_file.read_text(encoding='utf-8')
    assert "mensagem-de-teste-única-xyz" in text
    assert "INFO" in text

    _purge_pe_handlers()


def test_setup_logging_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Chamar setup duas vezes não duplica handlers."""
    import config
    monkeypatch.setattr(config, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(config, 'LOG_FILE', 'idem.log')

    import core.logging_setup as ls
    importlib.reload(ls)
    _purge_pe_handlers()

    ls.setup_logging()
    n1 = sum(1 for h in logging.getLogger().handlers
             if getattr(h, '_pe_tag', None) == ls._HANDLER_TAG)
    ls.setup_logging()
    n2 = sum(1 for h in logging.getLogger().handlers
             if getattr(h, '_pe_tag', None) == ls._HANDLER_TAG)
    assert n1 == 1
    assert n2 == 1

    _purge_pe_handlers()


def test_get_logger_returns_named_logger(tmp_path: Path, monkeypatch) -> None:
    import config
    monkeypatch.setattr(config, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(config, 'LOG_FILE', 'named.log')

    import core.logging_setup as ls
    importlib.reload(ls)
    logger = ls.get_logger('teste.modulo')
    assert logger.name == 'teste.modulo'
    assert isinstance(logger, logging.Logger)
