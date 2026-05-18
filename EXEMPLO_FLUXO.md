# Exemplo de fluxo completo de uso

Este documento mostra um caso completo de uso da plataforma — do primeiro
participante à exportação para Jamovi.

## 0. Pré-requisitos

- Python 3.11+ instalado
- Dependências instaladas (`pip install -r requirements.txt`)
- CSV do Mind Monitor exportado para cada sessão (formato padrão, 44 colunas)

## 1. Subir a plataforma

```bash
cd plataforma_eeg
streamlit run app.py
```

Abre em `http://localhost:8501`. Na primeira execução:
- Cria `data/eeg.db` (banco SQLite)
- Cria `logs/app.log`
- A página inicial mostra 4 métricas (todos com 0).

## 2. Pré-carregar cadastro em lote (opcional, recomendado)

Se você já tem uma planilha com os 96 participantes e suas respostas
ao questionário pré-experimento:

```bash
python scripts/import_responses.py "Cadastro_com_respostas.csv"
```

O CSV deve ter colunas: `participante, Gênero, Idade, Posicionamento Político,
Medo, Raiva, Estresse, Narcisismo, Humildade Intelectual, Misticismo, Habitos,
Identificador do Vídeo, Concordância, Veracidade, Compartilhamento`
(1 linha por participante × vídeo).

Saída esperada:
```
Linhas no CSV               : 384
Participantes criados       : 96
Sessões criadas (esqueleto) : 384
Autorrelatos gravados       : 384
Erros                       : 0
```

Volte ao navegador, recarregue a aba — agora a página inicial mostra:
- **Participantes**: 96
- **Sessões registradas**: 384
- **Com EEG importado**: 0 (esperado — só o EEG falta)

## 3. Cadastrar um participante novo manualmente (alternativa)

Se preferir cadastrar um a um via interface:

1. **Menu lateral → 📋 Cadastrar Participante**
2. **Código**: digite, por exemplo, `P001`
   - Se já existir, autopopula em modo de **edição**
   - Se novo, mostra mensagem **"Cadastro novo"**
3. Preencha **Gênero**, **Idade**, **Posição política**
4. Preencha os 7 escores de traços (Medo, Raiva, Estresse, Narcisismo,
   Humildade intelectual, Misticismo, Hábitos). Aceita decimais (ex.: `7,40`).
   Pode deixar em branco se algum instrumento não foi aplicado.
5. Clique em **"Cadastrar participante"**
6. A tabela "Participantes já cadastrados" abaixo é atualizada.

## 4. Importar uma sessão EEG

1. **Menu lateral → 📥 Importar sessões EEG**
2. **Sidebar → Selecionar**: escolha o código do participante (ex.: `F20Ca3Gk9N4j`)
   - O expander "Traços do participante" mostra os 7 valores cadastrados
3. **Linha de status** com 4 cards mostra o estado de cada vídeo:
   - ✅ `V1` — EEG já importado
   - ⏳ `V2` — só pré-cadastro (respostas do questionário)
   - ❌ `V3` — sem dados
4. **Abas dos 4 vídeos** — abra `📹 V1 · 130s`:
   - **Avaliação pré-cadastrada** mostra concordância/veracidade/compartilhamento
     já preenchidos do questionário externo
   - **Upload**: arraste o arquivo `.csv` (ou `.zip` com CSV dentro) do Mind Monitor
   - **Painel de qualidade** aparece automaticamente:
     - Score (0=ruim, 1=ótimo)
     - Amostras válidas / total (% descartado)
     - Piscadas/min
     - Duração
     - HSI por canal (TP9, AF7, AF8, TP10)
     - Alertas (se houver: HSI alto, duração curta, headband off, etc.)
   - **Autorrelato pós-vídeo** (opcional):
     - Intensidade de Alegria, Medo/Raiva, Tristeza, Serenidade (0–10)
     - Tempo sentido (segundos)
   - Clique em **💾 Importar V1**
5. Mensagem verde: **✅ V1 importado · sessão #X · NN janelas**
6. Repita para V2, V3, V4 — ou use o botão **📦 Importar todos os slots prontos**
   no final da página para processar em bloco.

## 5. Visualizar uma sessão

1. **Menu lateral → 👤 Sessao Individual**
2. **Sidebar → Participante**: `F20Ca3Gk9N4j`
3. **Sidebar → Sessão**: `V1 — desinformação` (ou o `video_type` que você tenha)
4. **Cabeçalho** mostra:
   - Dados demográficos do participante
   - Tipo do vídeo, duração esperada, arquivo CSV original
   - Score de qualidade, piscadas/min, amostras válidas
   - Expander 📝 com o autorrelato pós-vídeo
5. **Toggles na sidebar**:
   - ☑ Linha da mediana — adiciona linha tracejada nos gráficos
   - ☐ Marcadores de piscadas — recarrega o CSV e marca os tempos das piscadas
6. **6 gráficos Plotly** (rolar para baixo):
   - **Atenção** (azul) · `β / (α + θ)` frontal
   - **Engajamento cognitivo** (laranja) · `(β + γ) / α` frontal
   - **Engajamento afetivo** (verde) · `|FAA| + (β + γ) / α` frontal
   - **FAA** (renderer especial com áreas azul/vermelha · ln natural)
   - **Evocação de memórias** (vermelho) · `θ posterior`
   - **Aderência** (roxo) · `(γ_F + γ_P) / θ_P`
   - Hover mostra `t = X.Xs · valor = Y.YYYY`
   - Toolbar do Plotly tem ícone de câmera para **exportar PNG** de cada gráfico
7. **Tabela de estatísticas** (mediana, média, std, min, max, % acima da mediana)
   para as 8 medidas.

## 6. Análise agregada

1. **Menu lateral → 📊 Analise Agregada**
2. **Sidebar — filtros multi-seleção** (acumulativos):
   - Gênero (feminino / masculino / ...)
   - Faixa etária (18-24, 25-34, ...)
   - Posição política
   - Vídeo (V1, V2, V3, V4)
3. **Tabela mestra** com 40 colunas (role lateralmente para ver todas).
4. **Botões de download** abaixo da tabela:
   - 📥 **Tabela filtrada** (CSV) — respeita os filtros
   - 📥 **Tabela completa** (CSV) — ignora filtros
   - Ambos com BOM UTF-8 (Excel/Jamovi abrem com acentos corretos)
5. **Comparações entre grupos**:
   - Dropdown "Agrupar por": gender / age_group / political_position / video_id
   - Dropdown "Índice EEG": atencao_mean / eng_cognitivo_mean / ...
   - Mostra boxplot com pontos sobrepostos (jitter)
6. **Correlações Spearman**:
   - **Heatmap 1**: 7 traços × 8 índices EEG (nível participante)
   - **Heatmap 2**: 8 autorrelatos × 8 índices EEG (nível sessão)
   - Asterisco `*` em células com `p < 0,05`
7. **Scatter exploratório**:
   - Dropdowns X, Y e (opcional) Color
   - Linha de tendência OLS tracejada

## 7. Exportar para Jamovi

1. **Menu lateral → 💾 Exportar Jamovi**
2. **Cabeçalho** mostra: 96 participantes · 384 sessões · 80 colunas wide · 38 colunas long
3. **Preview em 3 abas**:
   - 📊 **Wide** (96 × 80) — 1 linha por participante; pivota cada métrica por vídeo
   - 📈 **Long** (384 × 38) — 1 linha por (participante × vídeo); ideal para modelos mistos
   - 📋 **Metadados** (38 variáveis descritas)
4. **Downloads**:
   - ⭐ **Pacote XLSX** (3 abas: wide, long, metadata) — recomendado
   - 📦 **Pacote ZIP** (3 CSVs com BOM UTF-8)
   - 📥 CSVs individuais (wide, long, metadata)

## 8. Abrir no Jamovi

1. Jamovi → `File → Open → Browse → Computer`
2. Selecione `Excel files (*.xlsx)` e abra o pacote
3. Use a aba **wide** para análises por participante:
   - Correlações entre traços (7 × 7)
   - Correlações entre traços e índices médios (7 × 8)
   - Comparações entre grupos demográficos
4. Use a aba **long** com módulo **GAMLj** para modelos mistos:
   - Variável dependente: `atencao` (ou outro índice)
   - Efeito fixo: `video_id`, `trait_raiva`, interações
   - Efeito aleatório: `1 | participant_code`
5. A aba **metadata** documenta tipo/escala de cada variável — use como referência
   ao configurar o Data Editor do Jamovi.

## 9. Administração e manutenção

Na página inicial, expanda **🔧 Administração**:

- Caminhos absolutos do banco e do log
- Botão **🔁 Recalcular índices de todas as sessões com EEG** — útil depois
  de mudanças de fórmula (não toca em dados nem nos arquivos CSV originais)
- **Últimas linhas do log** — visualização rápida de erros e atividade

## 10. Caso de erro: re-importação

Se você importou um EEG mas o arquivo estava errado:

1. Vá em **📥 Importar sessões EEG**
2. Selecione o participante
3. Abra a aba do vídeo certo
4. Faça **novo upload** do CSV correto
5. Veja o badge: *"Já existe EEG para `V1` desse participante. Subir um novo
   arquivo substitui o anterior."*
6. Clique em **💾 Importar V1** → confirma substituição.

O EEG antigo é deletado, o novo é processado, índices recalculados, e o
autorrelato pré-cadastrado é preservado.

---

**Tempo total estimado para o fluxo completo de 1 participante**:
- Cadastro: ~30 s (ou 0 s se pré-carregado)
- Importação de 4 vídeos: ~1 min (~15 s por arquivo)
- Visualização e exportação: ~30 s

Para N=96 participantes × 4 vídeos = 384 sessões, é viável processar tudo em
algumas horas de trabalho.
