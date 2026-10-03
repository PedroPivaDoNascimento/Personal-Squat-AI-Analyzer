# 🧠 Project Memory (Memória do Projeto e Arquivo de Decisões)

**Personal Squat AI Analyzer – Registro Histórico de Decisões de Arquitetura (ADR)**

| Campo | Informação |
| :--- | :--- |
| **Última Atualização** | 03 de Outubro de 2026 - 15:45 |
| **Fase Atual** | Operação e Refinamento do Pipeline Web/IA |
| **Status Geral** | Migração Streamlit para Django MVT Concluída com Sucesso |

---

## 📌 Registros de Decisões de Arquitetura (ADRs)

### ADR-001 — Adoção de MediaPipe Pose Landmarker Tasks (`pose_landmarker_full.task`)
- **Data:** 15/05/2026
- **Status:** ✅ Aceito
- **Contexto:** A biblioteca legada MediaPipe Solutions (`mp.solutions.pose`) foi depreciada pelo Google em prol da nova API MediaPipe Tasks Python.
- **Decisão:** Utilizar `vision.PoseLandmarker.create_from_model_path` com o modelo offline `models/pose_landmarker_full.task` encapsulado na classe `PoseDetector`.
- **Consequências:** Detecção de 33 marcos anatômicos com alta precisão 3D e sem dependência de download de modelo em tempo de execução.

### ADR-002 — Migração da Interface Monolítica Streamlit para Django MVT
- **Data:** 20/07/2026
- **Status:** ✅ Aceito
- **Contexto:** Streamlit gerenciava estado por reexecução total do script a cada interação, dificultando downloads seguros, parametrização controlada de requisições e persistência multi-usuário.
- **Decisão:** Manter o core em `src/classes` intacto e introduzir aplicação Django (`squat_analysis_app` e `squat_analyzer`) orquestrada pelo padrão Service Layer (`SquatAnalysisService`), com controllers em `views.py` e validações em `validators/file_validators.py`.
- **Consequências:** Facilidade de deploy web em containers Docker, rotas RESTful para downloads de relatórios e separação clara entre camada de apresentação e lógica biomecânica.

### ADR-003 — Classificação de Pronação Subtalar via Modelos Scikit-Learn
- **Data:** 10/08/2026
- **Status:** ✅ Aceito
- **Contexto:** A pronação subtalar no plano frontal apresenta sutilezas difíceis de serem captadas apenas por regras heurísticas estáticas lineares.
- **Decisão:** Treinar modelos supervisionados baseados em descritores estatísticos das séries temporais das coordenadas do pé (média, desvio padrão, IQR, amplitude) e serializá-los via Joblib (`modelo_pe_frontal_direito.pkl` e `modelo_pe_frontal_esquerdo.pkl`).
- **Consequências:** Classificação binária robusta da pisada integrada ao relatório da repetição.

### ADR-004 — Máquina de Estados para Detecção de Ciclo de Repetições
- **Data:** 25/08/2026
- **Status:** ✅ Aceito
- **Contexto:** Contadores simples de agachamento baseados apenas em ângulo do joelho falhavam com ruídos de câmera ou movimentos incompletos.
- **Decisão:** Implementar histerese vertical usando a coordenada Y da orelha (`ear_y`), calibrada nos primeiros 10 frames com quatro fases distintas: `inicial`, `descendo`, `subindo` e `final`, com limiares percentuais configuráveis (`DESCENT_THRESHOLD` e `ASCENT_RETURN_THRESHOLD`).
- **Consequências:** Segmentação confiável de até 3 repetições consecutivas com timestamps de fechamento de ciclo.

---

## 🛠️ Histórico de Alterações (Changelog da Sessão)

### 03/10/2026 — Sessão: Padronização e Atualização da Documentação do Projeto

| # | Ação | Arquivos alterados/criados | Detalhes |
| :--- | :--- | :--- | :--- |
| 1 | Mapeamento do Repositório | `src/*`, `squat_analyzer/*`, `squat_analysis_app/*`, `models/*` | Análise completa dos fluxos sagital, frontal, validações e exportação Excel |
| 2 | Substituição do PRD.md | `docs/PRD.md` | Eliminação do conteúdo do projeto legado; definição de requisitos biomecânicos do SLS |
| 3 | Atualização da Arquitetura | `docs/ARCHITECTURE.md` | Diagrama MVT + Service Layer, pipeline OpenCV/MediaPipe/ML e topologia Docker |
| 4 | Especificação de Design | `docs/DESIGN.md` | Padrões de interface Django, feedback postural, estados OK/DESVIO e formulários |
| 5 | Atualização do Histórico e Memória | `docs/GEMINI.md` | Registro dos ADRs 001 a 004 e alinhamento do status operacional do repositório |
| 6 | Padronização das Regras de Desenvolvimento | `docs/RULES.md` | Diretrizes estritas de visão computacional, SOLID, validação de arquivos e Excel writers |

### Observações Relevantes
- O pipeline de cálculo vetorial (`VectorCalculator`) é compartilhado entre todos os analisadores e garante que ângulos orientados sejam calculados via produto vetorial e produto escalar com `np.arctan2`.
- As saídas em Excel são salvas na raiz `planilhas/` e servidas via `views.download_excel` utilizando streaming seguro com `FileResponse`.
- Modelos `.task` e `.pkl` residem na pasta `models/` e devem ser mantidos versionados para garantia de reprodutibilidade dos testes.

---

## 🟢 Status dos Ingressos e Dependências de Infraestrutura
- **MediaPipe Pose Task:** Operacional (`models/pose_landmarker_full.task`).
- **Modelos de Pronação de Pé:** Operacionais (`models/modelo_pe_frontal_direito.pkl` e `esquerdo.pkl`).
- **Docker & Compose:** Configurados para porta 8000 com volumes persistentes.
- **Django Server:** Compatível com execução local (`python manage.py runserver`) e containerizada.
