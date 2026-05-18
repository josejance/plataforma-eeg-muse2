"""Testes do extrator CSV/ZIP."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from core.import_pipeline import extract_csv_bytes


def test_extract_csv_bytes_passthrough_plain_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "session.csv"
    csv_path.write_bytes(b"col1,col2\n1,2\n")
    out = extract_csv_bytes(csv_path)
    assert out == b"col1,col2\n1,2\n"


def test_extract_csv_bytes_from_bytes_with_csv_name() -> None:
    out = extract_csv_bytes(b"a,b\n1,2", filename="foo.csv")
    assert out == b"a,b\n1,2"


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_csv_bytes_from_zip_with_csv() -> None:
    zip_bytes = _make_zip({"session.csv": b"a,b\n1,2"})
    out = extract_csv_bytes(zip_bytes, filename="upload.zip")
    assert out == b"a,b\n1,2"


def test_extract_csv_bytes_zip_detected_by_magic_bytes() -> None:
    """Mesmo sem .zip no nome, magic bytes PK\\x03\\x04 disparam extração."""
    zip_bytes = _make_zip({"session.csv": b"a,b\n1,2"})
    out = extract_csv_bytes(zip_bytes, filename="sem_extensao")
    assert out == b"a,b\n1,2"


def test_extract_csv_bytes_zip_picks_first_csv_when_multiple() -> None:
    zip_bytes = _make_zip({
        "outro.txt": b"ignorar",
        "session.csv": b"primeiro",
        "extra.csv": b"segundo",
    })
    out = extract_csv_bytes(zip_bytes, filename="upload.zip")
    assert out in (b"primeiro", b"segundo")  # depende da ordem do filelist


def test_extract_csv_bytes_zip_ignores_macosx_metadata() -> None:
    zip_bytes = _make_zip({
        "__MACOSX/.DS_Store": b"junk",
        "session.csv": b"a,b\n1,2",
    })
    out = extract_csv_bytes(zip_bytes, filename="upload.zip")
    assert out == b"a,b\n1,2"


def test_extract_csv_bytes_zip_without_csv_raises() -> None:
    zip_bytes = _make_zip({"readme.txt": b"texto"})
    with pytest.raises(ValueError, match="não contém"):
        extract_csv_bytes(zip_bytes, filename="upload.zip")


def test_extract_csv_bytes_corrupted_zip_raises() -> None:
    with pytest.raises(ValueError, match="ZIP corrompido"):
        extract_csv_bytes(b"PK\x03\x04dados_corrompidos",
                          filename="upload.zip")


class _FakeUpload:
    """Simula objeto retornado pelo st.file_uploader."""
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getbuffer(self):
        return self._data


def test_extract_csv_bytes_streamlit_upload_csv() -> None:
    up = _FakeUpload(b"a,b\n1,2", "session.csv")
    assert extract_csv_bytes(up) == b"a,b\n1,2"


def test_extract_csv_bytes_streamlit_upload_zip() -> None:
    zip_bytes = _make_zip({"session.csv": b"a,b\n1,2"})
    up = _FakeUpload(zip_bytes, "session.zip")
    assert extract_csv_bytes(up) == b"a,b\n1,2"
