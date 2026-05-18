"""Coloca a raiz do projeto no sys.path para que os testes importem `config` e `core.*`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
