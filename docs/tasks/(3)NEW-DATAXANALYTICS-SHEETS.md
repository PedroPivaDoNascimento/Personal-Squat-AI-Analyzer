# Plano de Implementação e Divisão de Tarefas

**Personal Squat AI Analyzer – Módulo de Organização e Armazenamento de Dados de Pronação e Segmentos Biomecânicos**

---

## 📊 Métricas e Progresso Geral

* **Total de Tarefas:** 12
* **Concluídas:** 0 (0%)
* **Em Andamento:** 0 (0%)
* **Pendentes:** 12 (100%)

---

## 📌 Especificação dos Segmentos e Landmarks Analisados

Os dados brutos (séries temporais de coordenadas $X, Y, Z$ e visibilidade) e estatísticos (descritores agregados) gravados em `dados_do_segmento/` compreendem os marcos anatômicos do MediaPipe Pose correspondentes a cada segmento monitorado:

| Segmento Biomecânico | Landmarks MediaPipe Analisados | Índices PoseLandmarker | Coordenadas / Eixos de Interesse |
| :--- | :--- | :--- | :--- |
| **Cabeça** | Nariz, Olhos e Orelha de referência (`ear_y`, `nose`, `eye`) | 0 (nariz), 2/5 (olhos), 7/8 (orelhas) | Posição vertical ($Y$) para máquina de fases e alinhamento axial |
| **Tronco** | Ombros (`left_shoulder`, `right_shoulder`) e Linha da coluna | 11 (ombro esquerdo), 12 (ombro direito) | Inclinação anterior/posterior do tronco em relação à vertical ($X, Y$) |
| **Quadril** | Articulação coxofemoral (`left_hip`, `right_hip`) | 23 (quadril esquerdo), 24 (quadril direito) | Báscula pélvica no plano frontal e deslocamento sagital ($X, Y$) |
| **Joelho Frontal** | Joelho de apoio (`knee`) e alinhamento com quadril/tornozelo | 25 (joelho esquerdo), 26 (joelho direito) | Ângulo de projeção frontal (Valgo Dinâmico / HKA) ($X, Y$) |
| **Joelho Sagital** | Flexão do joelho de apoio e projeção sobre a linha do pé | 25 (joelho esquerdo), 26 (joelho direito) | Ângulo de flexão do joelho e relação joelho-calcanhar/ponta ($X, Y$) |
| **Pé Frontal** | Tornozelo, calcanhar e hálux (`ankle`, `heel`, `foot_index`) | 27/28 (tornozelo), 29/30 (calcanhar), 31/32 (hálux) | Deslocamento medial/lateral para predição de pronação dinâmica ($X, Y, Z$) |
| **Pé Sagital** | Alinhamento longitudinal calcanhar-hálux e elevação | 27/28 (tornozelo), 29/30 (calcanhar), 31/32 (hálux) | Elevação precoce do calcanhar e dorsiflexão do tornozelo ($X, Y$) |

---

## 🚀 Fase 1: Mapeamento de Diretórios e Criação Dinâmica de Pastas
Garantir que a estrutura de diretórios aninhada seja resolvida e criada dinamicamente antes da geração das planilhas de exportação.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Consultar `docs/RULES.md` e `docs/ARCHITECTURE.md` para validar os contratos de persistência do `SetFolders` e relatórios | Backend | Crítica | 🔴 Não Iniciado |
| 1.2 | Mapear o serviço de gerenciamento de pastas para construir dinamicamente o caminho `planilhas/<plano>/<lado>/dados_do_segmento/` | Backend | Crítica | 🔴 Não Iniciado |
| 1.3 | Implementar rotina determinística de verificação e criação (`os.makedirs` / `Path.mkdir`) para as subpastas `estatistica/` e `dados brutos/` | Backend | Alta | 🔴 Não Iniciado |

---

## ⚙️ Fase 2: Extração e Serialização de Dados Brutos (`dados brutos/`)
Coleta, estruturação e exportação das séries temporais de coordenadas dos landmarks de cada segmento (tronco, quadril, cabeça, joelho frontal, joelho sagital, pé sagital e pé frontal).

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | Extrair coordenadas $(X, Y, Z)$ e visibilidade dos landmarks de cada segmento (tronco, quadril, cabeça, joelho frontal, joelho sagital, pé sagital e pé frontal) frame a frame durante as repetições válidas | Backend / Data | Crítica | 🔴 Não Iniciado |
| 2.2 | Estruturar o DataFrame de dados brutos relacionando `frame`, `timestamp`, `segmento`, `landmark_id` e coordenadas tridimensionais | Data Engineer | Alta | 🔴 Não Iniciado |
| 2.3 | Implementar a gravação automatizada dos dados brutos em formato `.xlsx` dentro do diretório `planilhas/<plano>/<lado>/dados_do_segmento/dados brutos/` | Backend | Crítica | 🔴 Não Iniciado |

---

## 📈 Fase 3: Processamento e Salvamento de Métricas Estatísticas (`estatistica/`)
Cálculo dos descritores estatísticos agregados por repetição e persistência dos dados consolidados para consumo analítico e modelos de Machine Learning.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | Calcular métricas estatísticas das séries temporais de cada segmento (média, desvio padrão, amplitude e intervalo interquartil - IQR) | Data Engineer | Crítica | 🔴 Não Iniciado |
| 3.2 | Formatar o relatório estatístico dos segmentos no padrão compatível com os modelos Scikit-Learn (`modelo_pe_frontal_direito.pkl` / `esquerdo.pkl`) e diagnósticos clínicos | Data / Machine Learning | Alta | 🔴 Não Iniciado |
| 3.3 | Implementar o salvamento da planilha consolidada na pasta `planilhas/<plano>/<lado>/dados_do_segmento/estatistica/` | Backend | Crítica | 🔴 Não Iniciado |

---

## 🛡️ Fase 4: Integração, Tratamento de Exceções e Validação de Fluxo
Validação do salvamento sem concorrência e garantia de integridade da árvore de arquivos.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | Garantir que erros de escrita ou ausência de dados de segmentos não interrompam o processamento principal no `SquatAnalysisService` | Backend | Alta | 🔴 Não Iniciado |
| 4.2 | Validar se o nome dos arquivos salvos nas subpastas segue o padrão descritivo contendo o nome do voluntário e a repetição | Backend | Média | 🔴 Não Iniciado |
| 4.3 | Atualizar o histórico de decisões e arquitetura no arquivo `docs/GEMINI.md` e a documentação técnica relevante | Backend | Média | 🔴 Não Iniciado |
