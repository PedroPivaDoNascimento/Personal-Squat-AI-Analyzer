# Plano de Implementação e Divisão de Tarefas

**Personal Squat AI Analyzer – Módulo de Limpeza Autônoma de Relatórios (.xlsx)**

---

## 📊 Métricas e Progresso Geral

* **Total de Tarefas:** 12
* **Concluídas:** 0 (0%)
* **Em Andamento:** 0 (0%)
* **Pendentes:** 12 (100%)

---

## 🚀 Fase 1: Infraestrutura e Agendador de Tarefas (Background Job)
Configuração do serviço em segundo plano para execução periódica autônoma 24/7 (a cada 1 hora), sem depender de acessos de usuários ao site.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Definir e configurar a estratégia do agendador em segundo plano (APScheduler ou Cron isolado no Docker) | DevOps / Backend | Crítica | 🟡 Em Andamento |
| 1.2 | Configurar a rotina de disparo periódico com intervalo exato de 1 hora para execução contínua da limpeza | DevOps | Crítica | 🟡 Em Andamento |
| 1.3 | Configurar permissões de leitura/escrita no volume Docker para o diretório `planilhas/` | DevOps | Alta | 🟡 Em Andamento |

---

## ⚡ Fase 2: Core Engine de Limpeza (Django Management Command & Python)
Desenvolvimento do comando do Django responsável pela varredura, identificação da planilha `.xlsx` com a data/hora mais antiga e sua remoção física imediata.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | Criar o Django Management Command customizado `cleanup_oldest_sheet` | Backend | Crítica | 🟡 Em Andamento |
| 2.2 | Implementar varredura recursiva na pasta `planilhas/` e subpastas (`frontal/`, `sagital/`) filtrando apenas arquivos `.xlsx` | Backend | Crítica | 🟡 Em Andamento |
| 2.3 | Desenvolver algoritmo de comparação de timestamp (`mtime`/`ctime`) para selecionar unicamente a planilha mais antiga do repositório | Backend | Crítica | 🟡 Em Andamento |
| 2.4 | Implementar a remoção física permanente do arquivo selecionado a cada disparo (1 hora) | Backend | Alta | 🟡 Em Andamento |

---

## 🛡️ Fase 3: Tratamento de Exceções, Resiliência e Logging
Garantia de que a rotina de remoção a cada 1 hora rode com estabilidade, tratando exceções e registrando auditoria.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | Implementar verificação de segurança para tratar graciosamente pastas sem planilhas sem quebrar a rotina das próximas horas | Backend | Alta | 🟡 Em Andamento |
| 3.2 | Configurar logging estruturado no Django registrando o nome do arquivo deletado, caminho completo e timestamp da exclusão | Backend | Média | 🟡 Em Andamento |
| 3.3 | Escrever testes unitários e de integração simulando remoções consecutivas a cada ciclo | QA / Backend | Alta | 🟡 Em Andamento |

---

## 🐳 Fase 4: Conteinerização e Documentação
Ajustes na infraestrutura Docker e atualização da documentação técnica do projeto.

| # | Tarefa | Responsável | Prioridade | Status |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | Atualizar o `docker-compose.yml` para orquestrar o container/worker da rotina horária de limpeza | DevOps | Alta | 🟡 Em Andamento |
| 4.2 | Atualizar a documentação técnica (`docs/ARCHITECTURE.md` e `README.md`) detalhando a regra de exclusão horária da planilha mais antiga | Backend | Média | 🟡 Em Andamento |