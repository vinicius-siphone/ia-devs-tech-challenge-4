# Tech-Challenge (4ª Fase)

### Processamento e análise de vídeo e áudio em sessões terapêuticas.

### Alunos (6IADT)
- Luis Gustavo de Araújo Silva — RM 366233  
- Vinicius Santos de Oliveira — RM 366276

## Análise de Áudio (`audio-analysis/audio-analysis.py`)
Esta etapa transcreve o áudio, separa falas por heurística (`Paciente`/`Profissional`), gera tópicos (LDA) e calcula sinais sensíveis (não clínicos), usando **somente as falas da Paciente** para os indicadores finais.

### 1. Pré-requisitos
- Python 3.10+
- `pip`
- `ffmpeg` instalado no sistema (necessário para `pydub`)

No Windows (PowerShell), uma opção rápida para instalar `ffmpeg`:

```powershell
winget install Gyan.FFmpeg
```

Depois valide:

```powershell
ffmpeg -version
```

### 2. Instalação de dependências
Na raiz do projeto, execute:

```powershell
py -m pip install -r requirements.txt
```

### 3. Arquivo de entrada
Coloque o arquivo de áudio em:

```text
audio-analysis/audio-analysis.wav
```

Observação:
- O nome do arquivo está definido na constante `AUDIO_FILE` dentro de `audio-analysis/audio-analysis.py`.
- Se quiser outro nome, altere essa constante.

### 4. Execução do script
Opção A (a partir da raiz do projeto):

```powershell
py audio-analysis/audio-analysis.py
```

Opção B (entrando na pasta `audio-analysis`):

```powershell
cd audio-analysis
py audio-analysis.py
```

### 5. Saídas geradas
Após rodar, serão gerados:

- `audio-analysis/transcricao-audio.txt`
  - Transcrição com timestamps e falas separadas por `Paciente`/`Profissional` (heurística).

- `audio-analysis/relatorio_audio.json`
  - Relatório estruturado com:
  - metadados de transcrição,
  - tópicos (`lda_topics`),
  - sinais sensíveis (`risk_detection`),
  - resumo final (`final_summary`).

### 6. Interpretação rápida do relatório
- `risk_detection.overall_level` e `risk_detection.avg_score`
  - Indicam o nível geral de sensibilidade lexical.

- `risk_detection.top_chunks`
  - Mostra os trechos com maior evidência de termos sensíveis.

- `final_summary.assuntos_discutidos`
  - Tópicos mais presentes da conversa (LDA).

- `final_summary.conclusion`
  - Classificação textual final (não clínica) baseada em regras lexicais.

### 7. Observações importantes
- O indicador é **não clínico** e não substitui avaliação profissional.
- A separação de falas é **heurística** (alternância por sentença), não diarização acústica real.
- A análise final foi configurada para usar apenas falas da `Paciente` (`ANALYSIS_ROLE = "Paciente"`).

### 8. Problemas comuns
- Aviso do `pydub` sobre `ffmpeg` não encontrado:
  - Instale `ffmpeg` e reinicie o terminal.

- Erro de dependências:
  - Reinstale com `py -m pip install -r requirements.txt`.

- Arquivo de áudio não encontrado:
  - Verifique se existe `audio-analysis/audio-analysis.wav`.
