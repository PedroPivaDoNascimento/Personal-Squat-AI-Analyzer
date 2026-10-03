# Plano de Implementação e Divisão de Tarefas

**Personal Squat AI Analyzer – Módulo de Otimização e Performance (RAM & CPU)**

---

## 📊 Métricas e Progresso Geral

* **Total de Tarefas:** 15
* **Concluídas:** 0 (0%)
* **Em Andamento:** 0 (0%)
* **Pendentes:** 15 (100%)

---

## 🚀 Fase 1: Mapeamento de Contexto, Profiling e Diagnóstico (RAM & CPU)
Leitura dos documentos do projeto para entendimento da arquitetura e auditoria dos gargalos de memória e consumo de processador durante a análise de vídeos.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Realizar leitura obrigatória dos arquivos `docs/RULES.md`, `docs/ARCHITECTURE.md`, `docs/PRD.md` e `docs/GEMINI.md` | Backend | Crítica | 🔴 Não Iniciado |
| 1.2 | Consultar o arquivo `(1)CLEAR_HISTORY_OF_SHEETS.md` como base de conhecimento sobre as rotinas de limpeza já implementadas | Backend | Média | 🔴 Não Iniciado |
| 1.3 | Mapear o pipeline do `SquatAnalysisService` e identificar pontos de alto consumo de CPU (loops pesados por frame) e de alocação de RAM | Backend | Crítica | 🔴 Não Iniciado |

---

## ⚡ Fase 2: Otimização de Streaming e Storage de Upload (Django Core - RAM)
Ajustes nas configurações e handlers do Django para impedir que uploads de vídeos MP4 fiquem retidos na memória RAM (`InMemoryUploadedFile`).

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | Configurar `FILE_UPLOAD_MAX_MEMORY_SIZE` no `settings.py` para forçar o uso de `TemporaryUploadedFile` | Backend / DevOps | Crítica | 🔴 Não Iniciado |
| 2.2 | Validar/Ajustar `FILE_UPLOAD_HANDLERS` para garantir o streaming direto de uploads para o disco temporário do servidor | Backend | Crítica | 🔴 Não Iniciado |
| 2.3 | Garantir a exclusão física determinística dos arquivos de vídeo temporários no bloco `finally` após o processamento | Backend | Alta | 🔴 Não Iniciado |

---

## ⚙️ Fase 3: Otimização do Pipeline de Visão Computacional (Processamento & CPU)
Redução da carga de trabalho computacional sobre a CPU ao ler e extrair pose dos frames de vídeo.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | Implementar redimensionamento prévio de frame (*downscaling* para 720p ou 480p) caso a resolução do vídeo enviado seja excessiva (ex: 4K/1080p) | Backend | Crítica | 🔴 Não Iniciado |
| 3.2 | Configurar o modelo MediaPipe (`PoseLandmarker`) para focar estritamente na análise unipodal (`num_poses=1`) e desativar desenhos GUI (`draw=False`) | Backend | Alta | 🔴 Não Iniciado |
| 3.3 | Avaliar e aplicar técnicas de *Frame Subsampling* (processar a cada N frames caso o FPS do vídeo seja muito elevado) mantendo a precisão da máquina de estados | Backend | Alta | 🔴 Não Iniciado |
| 3.4 | Garantir vetorização via NumPy para todos os cálculos trigonométricos e eixos articulares no `VectorCalculator`, evitando laços `for` em Python puro | Backend | Alta | 🔴 Não Iniciado |

---

## 🛡️ Fase 4: Liberação de Recursos de IA, Memory Management e Garbage Collection (RAM)
Garantia de destruição explícita dos objetos pesados da visão computacional (OpenCV e MediaPipe) e execução do garbage collector pós-resposta.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | Implementar fechamento e liberação estrita de `cv2.VideoCapture` e `PoseLandmarker` (`pose_detector.close()`) em bloco `finally` | Backend | Crítica | 🔴 Não Iniciado |
| 4.2 | Desalocar explicitamente referências a buffers de frames e matrizes NumPy pesadas (`del frame`, `del landmarks`) ao fim da iteração | Backend | Alta | 🔴 Não Iniciado |
| 4.3 | Invocar a coleta de lixo explícita (`import gc; gc.collect()`) no `SquatAnalysisService` antes de devolver a resposta HTTP ao usuário | Backend | Crítica | 🔴 Não Iniciado |

---

## 🐳 Fase 5: Profiling, Testes de Carga e Atualização da Documentação (`GEMINI.md`)
Validação do comportamento do servidor sob estresse e registro das correções efetuadas na memória do projeto.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 5.1 | Executar testes de estresse com uploads simultâneos simulando vídeos pesados e monitorar o uso de CPU e curva de uso de RAM no container Docker | QA / DevOps | Alta | 🔴 Não Iniciado |
| 5.2 | Validar se o tempo de resposta do servidor caiu e se a memória RAM/CPU retorna ao nível baseline após a conclusão da análise | QA / Backend | Alta | 🔴 Não Iniciado |
| 5.3 | Atualizar o histórico de alterações no arquivo `docs/GEMINI.md` detalhando todas as otimizações de RAM e CPU aplicadas no sistema | Backend | Média | 🔴 Não Iniciado |