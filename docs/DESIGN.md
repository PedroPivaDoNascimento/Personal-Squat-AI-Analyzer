# Sistema de Design (UI/UX) - Análise Biomecânica

**Personal Squat AI Analyzer – Design System e Interface Web para Feedback Postural**

---

## 1. Princípios Gerais do Design
* **Clareza de Diagnóstico Biomecânico:** Exibição imediata e compreensível do status de cada segmento corporal avaliado (Tronco, Joelho, Cabeça, Quadril, Calcanhar e Pé).
* **Feedback Postural Acionável:** Fornecer dicas orientadas a ações corretivas imediatas e linguagem amigável ao paciente/atleta.
* **Transparência de Parâmetros:** Permitir parametrização visual dos limiares de sensibilidade (thresholds de descida, retorno e contagem de erros).

---

## 2. Paleta de Cores e Estados

| Estado | Hex / Variável | Aplicação no Fluxo de Análise |
| :--- | :--- | :--- |
| **Primária** | `#0D6EFD` / `#0052FF` | Botões de ação, envio de formulário, botões de download Excel |
| **Sucesso (OK)** | `#198754` / `#00C853` | Repetição executada sem desvios significativos (`OK ✅`) |
| **Alerta / Desvio** | `#DC3545` / `#D50000` | Desvio detectado no segmento corporal (`DESVIO ❌`) |
| **Não Identificado** | `#6C757D` | Repetição incompleta ou marcos anatômicos perdidos (`-1`) |
| **Fundo de Destaque** | `#F8F9FA` | Cards de repetição, agrupamentos de métricas e formulários |
| **Dica / Feedback** | `#0DCAF0` / `#FFF3CD` | Caixas de recomendação técnica e mensagens preventivas |

---

## 3. Componentes Específicos para Fluxo de Análise Biomecânica

### 3.1. Indicador de Status por Repetição (Cards de Repetição)
* **Card Consolidado:** Agrupa o timestamp de término da repetição e o status individual de cada articulação.
* **Ícone e Badge Visual:**
  - `OK ✅`: Segmento permaneceu dentro da faixa de tolerância biomecânica.
  - `DESVIO ❌`: Contagem de frames em desalinhamento ultrapassou o limiar de sensibilidade configurado.
* **Bloco de Recomendações:** Dicas personalizadas extraídas de `feedback_messages.py` associadas a cada falha técnica identificada.

### 3.2. Controles de Formulário e Parâmetros de Thresholds
* **Input de Voluntário e Vídeo:** Campo textual para nome da pessoa e upload de arquivo restrito a arquivos `.mp4` até $100\text{ MB}$.
* **Seleção de Repetições (Plano Frontal):** Checkboxes opcionais para repetições 1, 2 e 3 (`options_marcadas`) destinadas ao registro estatístico dos dados do pé.
* **Campos Numéricos com Valores Padrão (Sensibilidade):**
  - Plano Frontal: `descent_th` (0.05), `hip_err_th` (1), `ascent_return_th` (0.02), `knee_valgus_th` (5 ou 12), `foot_pronation_th` (7).
  - Plano Sagital: `descent_th` (0.05), `trunk_err_th` (23), `head_err_th` (2), `ascent_return_th` (0.02), `knee_err_th` (6), `foot_err_th` (8), `user_height_cm` (170cm).

---

## 4. Estratégia de Feedback Visual no Django (Templates)

```
[ Usuário Seleciona Plano/Lado e Envia Vídeo ]
                       |
                       v
       [ Validação de Arquivo MP4 no Backend ]
         (Se inválido -> Alerta Bootstrap / Django Messages com Erro)
                       |
                       v
       [ Processamento pelo SquatAnalysisService ]
                       |
                       v
    [ Renderização do Template de Resultados ]
       ├── Tabela Resumo das Repetições Detectadas
       ├── Badges de Erro por Segmento Corporal (Tronco, Joelho, Cabeça, Quadril, Calcanhar, Pé)
       ├── Caixas Informativas com Recomendações Corretivas
       └── Botão de Download Direto da Planilha Gerada (.xlsx)
```