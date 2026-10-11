# Plano de Implementação e Divisão de Tarefas

**Personal Squat AI Analyzer – Módulo de Organização e Armazenamento de Dados de Segmentos Biomecânicos (Arquitetura Abstrata `SegmentsWriter` e Implementações por Segmento)**

---

## 📊 Métricas e Progresso Geral

* **Total de Tarefas:** 16
* **Concluídas:** 0 (0%)
* **Em Andamento:** 0 (0%)
* **Pendentes:** 16 (100%)

---

## 📌 Especificação dos Segmentos e Landmarks Analisados

Os dados brutos e estatísticos gerados para cada repetição concluída gravam planilhas sob a estrutura `planilhas/<plano>/<lado>/dados_<segmento>/`.

| Segmento Biomecânico (`<segmento>`) | Plano | Landmarks MediaPipe Analisados | Índices PoseLandmarker | Coordenadas / Eixos Coletados e Metodologia de Cálculo |
| :--- | :--- | :--- | :--- | :--- |
| **cabeca** | Sagital / Frontal | Nariz, Olhos e Orelha de referência (`nose`, `eyes`, `ear`) | 0 (nariz), 2/5 (olhos), 7/8 (orelhas) | Coordenadas normalizadas $(X, Y, Z)$ e visibilidade |
| **tronco** | Sagital / Frontal | Ombros (`left_shoulder`, `right_shoulder`) e Quadril | 11/12 (ombros), 23/24 (quadris) | Coordenadas normalizadas $(X, Y, Z)$ e inclinação |
| **quadril** | Sagital / Frontal | Articulação coxofemoral (`left_hip`, `right_hip`) | 23 (quadril esquerdo), 24 (quadril direito) | Coordenadas normalizadas $(X, Y, Z)$ |
| **joelho (frontal)** | Frontal | Joelho de apoio (`knee`), quadril e tornozelo | 23/24 (quadril), 25/26 (joelho), 27/28 (tornozelo) | Alinhamento angular frontal / HKA, valgo dinâmico |
| **joelho (sagital)** | Sagital | Flexão de joelho de apoio e alinhamento com pé | 25/26 (joelho), 27/28 (tornozelo), 23/24 (quadril) | Ângulo de flexão articular, profundidade e avanço sobre a ponta do pé |
| **pe (frontal)** | Frontal | Tornozelo, calcanhar e hálux (`ankle`, `heel`, `foot_index`) | 27/28 (tornozelo), 29/30 (calcanhar), 31/32 (hálux) | Pronação/supinação dinâmica e rotação lateral do pé |
| **pe (sagital)** | Sagital | Calcanhar, tornozelo e hálux (`heel`, `ankle`, `foot_index`) | 27/28 (tornozelo), 29/30 (calcanhar), 31/32 (hálux) | Dorsiflexão de tornozelo e elevação/descolamento do calcanhar do solo |

---

## 🏗️ Especificação Arquitetural: Pacote `segments_writer` e Abstração `SegmentsWriter`

Todos os escritores de dados em Excel dos segmentos agora devem seguir estritamente o mesmo padrão baseado em **orientação a objetos com herança e classes abstratas (`abc.ABC`)**.

### 1. Separação Metodológica de Cálculos (Pé e Joelho por Plano)
Como o plano **Frontal** e o plano **Sagital** utilizam variáveis biomecânicas, colunas temporais e metodologias matemáticas distintas para **pé** e **joelho**, cada um possuirá sua própria classe especializada e arquivo dedicado:
- **Pé Frontal vs Sagital:**
  - Frontal avalia pronação dinâmica e desvios laterais do tornozelo/hálux.
  - Sagital avalia dorsiflexão, ângulos de elevação de calcanhar e projeção anteroposterior.
- **Joelho Frontal vs Sagital:**
  - Frontal avalia ângulo HKA (valgo/varo dinâmico) e colapso medial.
  - Sagital avalia grau de flexão máxima, velocidade angular de descida e projeção sobre o pé.

### 2. Organização e Localização de Pastas e Arquivos
Dentro do diretório `src/classes/excel/`, todos os arquivos de implementação dos escritores de segmentos devem residir dentro do subdiretório `segments_writer/`:
```text
src/classes/excel/
├── __init__.py
├── set_folders.py
├── squat_report_excel_writer/
└── segments_writer/
    ├── __init__.py
    ├── segments_writer.py                  # Classe base abstrata: SegmentsWriter (ABC)
    ├── frontal_foot_data_excel_writer.py   # Herda de SegmentsWriter (FrontalFootDataExcelWriter)
    ├── sagital_foot_data_excel_writer.py   # Herda de SegmentsWriter (SagitalFootDataExcelWriter)
    ├── frontal_knee_data_excel_writer.py   # Herda de SegmentsWriter (FrontalKneeDataExcelWriter)
    ├── sagital_knee_data_excel_writer.py   # Herda de SegmentsWriter (SagitalKneeDataExcelWriter)
    ├── hip_data_excel_writer.py            # Herda de SegmentsWriter (HipDataExcelWriter)
    ├── trunk_data_excel_writer.py          # Herda de SegmentsWriter (TrunkDataExcelWriter)
    └── head_data_excel_writer.py           # Herda de SegmentsWriter (HeadDataExcelWriter)
```
> **Nota de Migração:** A implementação existente em `src/classes/excel/foot_data_excel_writer.py` (focada no plano frontal) deve ser refatorada para `src/classes/excel/segments_writer/frontal_foot_data_excel_writer.py` (com alias de compatibilidade para `FootDataExcelWriter` se necessário durante a transição), herdando obrigatoriamente de `SegmentsWriter`.

### 3. Classe Pai Abstrata (`SegmentsWriter`)
A classe pai abstrata deve ter o mesmo nome do diretório (`SegmentsWriter` em `segments_writer.py`):
- **Herança:** `abc.ABC`.
- **Inicialização (`__init__`):**
  - `repetition` (int): número da repetição.
  - `segment_repeat_data` (DataFrame / list de dicts): dados coletados do segmento no ciclo da repetição.
  - `person_name` (str): nome do voluntário.
  - `plane_folder_name` (str): `'frontal'` ou `'sagital'`.
  - `side` (str): `'direito'` ou `'esquerdo'`.
  - `segment_name` (str): `'pe'`, `'joelho'`, `'quadril'`, `'tronco'`, `'cabeca'` (definido na subclasse).
- **Métodos Concretos Compartilhados (Template / Reutilizáveis):**
  - `_create_folder_raw_data()`: utiliza `SetFolders` configurado para o segmento específico.
  - `_create_folder_statistic_data()`: utiliza `SetFolders` configurado para o segmento específico.
  - `_build_file_name()`: mantém rigorosamente a convenção de nome base existente para o segmento, incluindo ao final a respectiva repetição analisada:
    - Formato: `dados_<segmento>_<person_slug>_rep<repetition>.xlsx` (ex: `dados_pe_joao_silva_rep1.xlsx`, `dados_joelho_maria_souza_rep2.xlsx`).
  - **Exportação Completa de Todas as Repetições:** É mandatório exportar sempre **TODOS os dados de todas as repetições concluídas**, sem omissão ou descarte de ciclos, gerando o arquivo individual com o sufixo da repetição e/ou agregando incrementalmente todas as repetições do voluntário.
  - `_convert_data_to_raw_pandas()`: serialização flatten padronizada de toda a série temporal (`['voluntario', 'repeticao', 'dado_0', ...]`).
  - Gravação incremental/persistente com `openpyxl` + `pd.concat` com `ignore_index=True`.
- **Métodos Abstratos / Especializados por Segmento e Plano (`@abstractmethod`):**
  - `segment_name` (property abstrata): nome identificador da pasta do segmento.
  - `convert_data_to_statistic_pandas()`: cálculo específico de métricas estatísticas e biomecânicas adequadas ao plano do segmento (ex: colunas de pronação no pé frontal vs elevação no pé sagital; colapso medial no joelho frontal vs flexão no joelho sagital).
  - `write_raw_data()` e `write_statistic_data()`: pontos de entrada para exportar e salvar todos os dados brutos e estatísticos da repetição.

---

## 🚀 Fase 1: Extensão da Estrutura de Pastas e SetFolders
Garantir resolução dinâmica de diretórios para qualquer segmento biomecânico.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Consultar `docs/RULES.md` e `docs/ARCHITECTURE.md` para assegurar conformidade do diretório raiz `planilhas/` | Backend | Crítica | 🔴 Não Iniciado |
| 1.2 | Atualizar/generalizar `SetFolders` para receber parâmetro opcional/obrigatório `segment_name` e criar caminho dinâmico `planilhas/<plano>/<lado>/dados_<segmento>/` para `pe`, `joelho`, `quadril`, `tronco`, `cabeca` | Backend | Crítica | 🔴 Não Iniciado |
| 1.3 | Implementar criação determinística com `os.makedirs(..., exist_ok=True)` para as subpastas `dados brutos/` e `dados estatisticos/` de cada segmento | Backend | Alta | 🔴 Não Iniciado |

---

## ⚙️ Fase 2: Implementação da Abstração `SegmentsWriter` e Writers por Segmento
Criar o pacote `src/classes/excel/segments_writer/` com classe abstrata pai e classes filhas especializadas por plano e segmento.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | Criar o diretório `src/classes/excel/segments_writer/` e a classe base abstrata `SegmentsWriter` (`segments_writer.py`) com métodos comuns de serialização e persistência Excel | Backend | Crítica | 🔴 Não Iniciado |
| 2.2 | Refatorar/migrar implementação do pé frontal para `src/classes/excel/segments_writer/frontal_foot_data_excel_writer.py` (`FrontalFootDataExcelWriter`), herdando de `SegmentsWriter` e aplicando métricas de pronação/eixos frontais | Backend | Crítica | 🔴 Não Iniciado |
| 2.3 | Implementar `sagital_foot_data_excel_writer.py` (`SagitalFootDataExcelWriter`) herdando de `SegmentsWriter`, especializado na metodologia de cálculo sagital (elevação de calcanhar, dorsiflexão) | Backend | Crítica | 🔴 Não Iniciado |
| 2.4 | Implementar `frontal_knee_data_excel_writer.py` (`FrontalKneeDataExcelWriter`) herdando de `SegmentsWriter`, especializado na metodologia de cálculo frontal (alinhamento HKA, valgo/varo dinâmico) | Backend | Crítica | 🔴 Não Iniciado |
| 2.5 | Implementar `sagital_knee_data_excel_writer.py` (`SagitalKneeDataExcelWriter`) herdando de `SegmentsWriter`, especializado na metodologia de cálculo sagital (ângulo de flexão, profundidade) | Backend | Crítica | 🔴 Não Iniciado |
| 2.6 | Implementar `hip_data_excel_writer.py` (`HipDataExcelWriter`) herdando de `SegmentsWriter` para segmento `quadril` | Backend | Alta | 🔴 Não Iniciado |
| 2.7 | Implementar `trunk_data_excel_writer.py` (`TrunkDataExcelWriter`) herdando de `SegmentsWriter` para segmento `tronco` | Backend | Alta | 🔴 Não Iniciado |
| 2.8 | Implementar `head_data_excel_writer.py` (`HeadDataExcelWriter`) herdando de `SegmentsWriter` para segmento `cabeca` | Backend | Alta | 🔴 Não Iniciado |

---

## 📈 Fase 3: Bufferização de Landmarks por Segmento nos Analisadores
Coleta contínua e segmentação por repetição nos analisadores sagital e frontal.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | Implementar buffers de repetição para cabeça, tronco, quadril e joelho frontal em `BaseFrontal`, `RightFrontal` e `LeftFrontal` | Backend | Alta | 🔴 Não Iniciado |
| 3.2 | Implementar buffers de repetição para cabeça, tronco, joelho sagital e pé sagital em `BaseSaggital`, `RightSaggital` e `LeftSaggital` | Backend | Alta | 🔴 Não Iniciado |
| 3.3 | Adicionar recorte exato dos frames de cada segmento por repetição (auto-curação via histórico/timestamps de repetição) | Backend / Data | Alta | 🔴 Não Iniciado |

---

## 🛡️ Fase 4: Integração no Service Layer e Validação de Fluxo
Gravação segura sem concorrência e sem acoplamento à camada web.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | Criar/estender `SegmentDataService` instanciando os respectivos escritores de `src/classes/excel/segments_writer/` (direcionando frontal/sagital para pé e joelho) para exportar obrigatoriamente TODOS os dados de cada repetição concluída (`rep1`, `rep2`, ..., `repN`) ao término de cada repetição ou do agachamento completo | Backend | Alta | 🔴 Não Iniciado |
| 4.2 | Garantir tratamento de exceções em bloco `try/except` com logging para que falhas de I/O em planilhas não interrompam o processamento de `SquatAnalysisService` | Backend | Crítica | 🔴 Não Iniciado |
| 4.3 | Validar integridade da árvore de diretórios gerada em `planilhas/` para planos sagital e frontal, lados direito e esquerdo de todos os segmentos | QA / Backend | Média | 🔴 Não Iniciado |



