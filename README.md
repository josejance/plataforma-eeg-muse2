# Plataforma EEG · Muse 2 / Mind Monitor

Plataforma local em Python/Streamlit para análise de resposta neural a vídeos
de desinformação política. Processa CSVs do Mind Monitor (Muse 2, 4 canais),
calcula 8 índices neurocognitivos por sessão, permite anotar metadados,
gera visualizações de linha do tempo e exporta planilhas formatadas para
análise estatística no Jamovi.

## Stack

- **Python 3.11+** (testado em 3.14)
- **Streamlit 1.30+** — UI local multi-página
- **pandas / numpy / scipy** — processamento e estatística
- **Plotly** — gráficos interativos
- **SQLite** — persistência (banco único `data/eeg.db`)
- **openpyxl** — exportação XLSX

Roda local em macOS, Windows e Linux — sem dependências de nuvem.

## Instalação

```bash
git clone <url>
cd plataforma_eeg
python -m pip install -r requirements.txt
```

## Como rodar

```bash
streamlit run app.py
```

A app abre em `http://localhost:8501` automaticamente. Na primeira execução,
cria o banco SQLite em `data/eeg.db` e a pasta de logs em `logs/app.log`.

## Estrutura de arquivos

```
plataforma_eeg/
├── README.md                          # este arquivo
├── EXEMPLO_FLUXO.md                   # passo-a-passo de uso
├── requirements.txt
├── app.py                             # entrada Streamlit + administração
├── config.py                          # constantes (canais, fórmulas, paths)
├── conftest.py                        # bootstrap sys.path para pytest
├── core/
│   ├── parser.py                      # leitura+validação do CSV Mind Monitor
│   ├── preprocessing.py               # filtro de qualidade + log→linear
│   ├── windowing.py                   # janelas 5s/50% + suavização
│   ├── indices.py                     # 8 fórmulas + pipeline completo
│   ├── statistics.py                  # mediana, std, % acima da mediana
│   ├── aggregations.py                # build_master_table + filtros
│   ├── import_pipeline.py             # CSV/ZIP → banco
│   └── logging_setup.py               # logs/app.log rotativo
├── db/
│   ├── schema.py                      # SCHEMA_SQL + migrações
│   └── queries.py                     # CRUD participants/sessions/...
├── pages/
│   ├── 1_Cadastrar_Participante.py
│   ├── 2_Importar.py                  # 4 slots de vídeo, CSV ou ZIP
│   ├── 3_Sessao_Individual.py         # 6 gráficos + tabela resumo
│   ├── 4_Analise_Agregada.py          # tabela mestra + boxplots + Spearman
│   └── 5_Exportar_Jamovi.py           # wide/long/metadata em XLSX
├── visualizations/
│   ├── timeline.py                    # gráficos por índice + FAA especial
│   ├── comparisons.py                 # boxplots + scatter com tendência
│   └── correlations.py                # Spearman + heatmap com p-valores
├── exports/
│   └── jamovi.py                      # build_wide, build_long, metadata
├── scripts/
│   ├── import_participants.py         # CSV → cadastro
│   ├── import_responses.py            # CSV agregado → cadastro + sessões esqueleto
│   └── recompute_indices.py           # re-aplicar fórmulas a sessões existentes
├── tests/                             # 140 testes (pytest)
├── data/                              # gerado em runtime
│   ├── eeg.db                         #   ← banco SQLite
│   └── csv/<participant>/<video>.csv  #   ← cópia dos uploads
└── logs/
    └── app.log                        # rotativo (10 MB × 3)
```

## Fluxo recomendado de uso

```
        ┌───────────────────────┐
        │ Cadastrar Participante │  ← 1 vez por participante (7 traços decimais)
        └────────────┬───────────┘
                     ↓
        ┌──────────────────────────┐
        │   Importar (V1–V4)        │  ← 4 slots; CSV ou ZIP do Mind Monitor
        │  • Upload                 │     processamento automático após upload
        │  • Preview de qualidade   │     → grava: sessão + autorrelato + 8 índices
        │  • Autorrelato emocional  │
        └────────────┬──────────────┘
                     ↓
        ┌───────────────────────┐         ┌───────────────────────┐
        │ Sessão Individual     │  ←→     │  Análise Agregada     │
        │ 6 gráficos · tabela   │         │  filtros · boxplot ·  │
        │ estatísticas          │         │  Spearman · scatter   │
        └────────────┬──────────┘         └────────────┬──────────┘
                     ↓                                 ↓
                     └─────────────┬───────────────────┘
                                   ↓
                     ┌──────────────────────────┐
                     │  Exportar Jamovi         │
                     │  XLSX (wide/long/meta)   │
                     │  ZIP de CSVs · CSV avulso│
                     └──────────────────────────┘
```

## Diagrama do fluxo de dados

```
  Mind Monitor          parser.py             preprocessing.py
  ┌──────────┐         ┌──────────┐          ┌──────────────────┐
  │ CSV 44 cols ├─→ load_csv → DataFrame ├──→  quality_filter   │
  │ (log10)    │         │ + t_sec  │          │ (HSI ≥ 4 → drop) │
  └──────────┘         └──────────┘          └─────────┬────────┘
                                                       ↓
                                            ┌──────────────────┐
                                            │ log_to_linear    │
                                            │ (cria *_lin)     │
                                            └─────────┬────────┘
                                                      ↓
                                            ┌──────────────────┐
                              indices.py    │  make_windows    │
                                            │  (5s, overlap 50)│
                                            └─────────┬────────┘
                                                      ↓
              ┌───────────────────────────────────────┴────────────────┐
              ↓                ↓                ↓                       ↓
        Atenção·Pope   Eng.Cog (β+γ)/α    Eng.Afet |FAA|+ratio   ... mais 5
              └────────────────┴────────────────┴────────────────┬─────┘
                                                                 ↓
                                            ┌──────────────────┐
                                            │  moving_average  │
                                            │  (5s sobre série)│
                                            └─────────┬────────┘
                                                      ↓
                                            ┌──────────────────┐
                                            │  save_indices    │
                                            │  → eeg_indices   │
                                            │  (SQLite)         │
                                            └─────────┬────────┘
                                                      ↓
                              ┌───────────────────────┴──────────────────┐
                              ↓                                          ↓
                  ┌──────────────────┐                       ┌──────────────────┐
                  │ Sessão Individual │                       │  Análise Agregada│
                  │ 6 gráficos Plotly │                       │  master_table SQL│
                  │ tabela resumo     │                       │  Spearman heatmap│
                  └──────────────────┘                       └─────────┬────────┘
                                                                       ↓
                                                          ┌──────────────────┐
                                                          │  Exportar Jamovi │
                                                          │  wide+long+meta  │
                                                          │  XLSX / ZIP / CSV│
                                                          └──────────────────┘
```

## Índices calculados

Todos sobre potências em **escala linear** (após `10^valor` das bandas log10
do Mind Monitor), exceto FAA, que opera em log natural.

| Índice | Fórmula | Canais | Referência |
|---|---|---|---|
| Atenção | β / (α + θ) | AF7, AF8 | Pope, Bogart & Bartolome (1995) |
| Eng. cognitivo | (β + γ) / α | AF7, AF8 | síntese |
| Eng. afetivo | \|FAA\| + (β + γ) / α | AF7, AF8 | composto |
| FAA | ln(α_AF8) − ln(α_AF7) | AF7, AF8 | (com sinal preservado) |
| Evocação | θ posterior | TP9, TP10 | absoluto |
| Aderência | (γ_F + γ_P) / θ_P | todos | proxy operacional |
| Arousal | β / α | todos os 4 | médias gerais |
| Estresse | (β/α) + (γ/θ) | todos os 4 | Arsalan et al. (2019) |

**Janelamento**: 5 segundos com 50% de sobreposição (passo de 2,5 s).
**Suavização**: média móvel de 5 s sobre a série de índices.
**Filtro de qualidade**: descartar amostras com ≥ 3 dos 4 canais com HSI ≥ 4.

## Importação em lote

Para pré-carregar cadastro + respostas a partir de planilhas:

```bash
# Apenas cadastro de participantes (demografia + traços)
python scripts/import_participants.py "caminho/cadastro.csv"

# Cadastro + respostas pós-vídeo (cria sessões esqueleto)
python scripts/import_responses.py "caminho/cadastro_respostas.csv"
```

Ambos são **idempotentes** — preservam EEG já importado e atualizam só os
campos novos.

Para recalcular índices após mudanças de fórmula:

```bash
python scripts/recompute_indices.py
```

Ou use o botão **🔁 Recalcular índices** na seção de Administração da app.

## Testes

```bash
python -m pytest tests/ -v
```

140 testes cobrindo: fórmulas, parser, pré-processamento, janelamento, CRUD,
extração CSV/ZIP, agregações, correlações Spearman, visualizações e
exportação Jamovi.

## Configuração

Edite `config.py` para ajustar:

- `WINDOW_SIZE`, `WINDOW_OVERLAP`, `SMOOTHING_WINDOW`
- `HSI_BAD_THRESHOLD`, `HSI_MEAN_WARN`, `BLINKS_PER_MIN_WARN`
- `MIN_SESSION_DURATION_SEC`
- `VIDEO_IDS`, `VIDEO_DURATIONS`
- `DB_PATH`, `CSV_STORAGE_DIR`, `LOG_DIR`, `LOG_FILE`

## Logs

Os módulos gravam em `logs/app.log` com rotação de 10 MB × 3 arquivos.
Última seção do log é visualizável na expansão **🔧 Administração** da página inicial.

## Deploy no Streamlit Community Cloud

O repo já vem configurado:
- `runtime.txt` → Python 3.12
- `.streamlit/config.toml` → headless, upload até 200 MB, tema claro
- `requirements.txt` → todas as deps pinned por versão mínima

Passos no [share.streamlit.io](https://share.streamlit.io/):
1. Sign in com GitHub
2. **New app** → selecionar `josejance/plataforma-eeg-muse2`
3. Branch: `main` · Main file: `app.py`
4. Authorize Streamlit Cloud a acessar o repo privado (uma única vez)
5. Deploy → fica em `https://<seu-app>.streamlit.app`

### Auto-seed do banco

A pasta `seed/eeg_seed.db` é o snapshot "factory default" carregado no
primeiro start da app (e a cada redeploy do Streamlit Cloud, dado que o
filesystem dele é efêmero). No startup, `app.py` copia esse arquivo para
`data/eeg.db` se este último ainda não existir.

Para **atualizar** o seed antes de um novo deploy:
```bash
# Roda o export, sobrescrevendo o seed comitado
python scripts/export_backup.py --out seed/eeg_seed.db
git add seed/eeg_seed.db
git commit -m "atualiza seed para deploy"
git push
```

> ⚠️ **Aviso ético/legal**: o `seed/eeg_seed.db` contém dados de 96
> participantes humanos (demografia, traços, posição política, índices EEG).
> O repo é **privado** e o deploy no Streamlit Cloud é **acesso restrito**;
> ainda assim, isso constitui hospedagem em infra de terceiros (GitHub +
> Streamlit/AWS). Verifique que o TCLE e o parecer ético do seu programa
> autorizam essa hospedagem. Se em dúvida, mantenha o repo privado e gere
> backups apenas localmente (use a feature **Restaurar banco** na app online
> para uploads pontuais sob seu controle).

## Licença

Uso acadêmico — dissertação de mestrado em comunicação (IDP).
