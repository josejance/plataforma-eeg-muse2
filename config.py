"""Constantes da plataforma: canais, bandas, limiares de qualidade e janelamento."""
from pathlib import Path

# Canais Muse 2
FRONTAL_CHANNELS = ['AF7', 'AF8']
POSTERIOR_CHANNELS = ['TP9', 'TP10']
ALL_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

# Bandas de frequência exportadas pelo Mind Monitor (em log10 de potência)
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Janelamento
WINDOW_SIZE = 5.0          # segundos
WINDOW_OVERLAP = 0.5       # fração (0.5 = 50%)
SMOOTHING_WINDOW = 5.0     # segundos (média móvel sobre série de índices)

# Limiares de qualidade
HSI_BAD_THRESHOLD = 4              # HSI >= 4 indica sem contato
HSI_MEAN_WARN = 2.5
BLINKS_PER_MIN_WARN = 40
HEADBAND_OFF_MAX_RATIO = 0.10
MIN_SESSION_DURATION_SEC = 30.0

# Logging
LOG_DIR = Path('logs')
LOG_FILE = 'app.log'

# Persistência local
DATA_DIR = Path('data')
DB_PATH = DATA_DIR / 'eeg.db'
CSV_STORAGE_DIR = DATA_DIR / 'csv'

# Opções de dropdown da UI
GENDER_OPTIONS = ['feminino', 'masculino', 'não-binário', 'prefere não informar']
POLITICAL_OPTIONS = [
    'esquerda', 'centro-esquerda', 'centro',
    'centro-direita', 'direita', 'prefere não informar',
]
CONCORDANCE_OPTIONS = ['Concordo', 'Não concordo', 'Indiferente']
VERACITY_OPTIONS = ['Verdadeiro', 'Mentiroso', 'Não sei']
SHARING_OPTIONS = [
    'Compartilharia esse vídeo', 'Não compartilharia esse vídeo',
    'Talvez', 'Prefere não responder',
]

# Vídeos do experimento — IDs e durações esperadas em segundos
VIDEO_IDS = ['V1', 'V2', 'V3', 'V4']
VIDEO_DURATIONS = {
    'V1': 130.0,
    'V2': 130.0,
    'V3': 84.0,
    'V4': 118.0,
}
