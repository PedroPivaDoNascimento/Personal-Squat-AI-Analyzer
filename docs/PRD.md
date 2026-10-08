# Documento de Requisitos do Produto (PRD)

**Personal Squat AI Analyzer – Sistema Inteligente de Avaliação Biomecânica do Agachamento Unipodal (SLS)**

| Campo | Valor |
| :--- | :--- |
| **Versão** | 1.0.0-PROD |
| **Data** | 03 de Outubro de 2026 |
| **Autor** | Pedro Piva |
| **Status** | Aprovado para Arquitetura e Implementação |
| **Lançamento Alvo** | Avaliação Biomecânica Clínica e Esportiva |

---

## 1. Visão Geral do Produto
O **Personal Squat AI Analyzer** é uma plataforma baseada em visão computacional e inteligência artificial desenhada para analisar a execução biomecânica do agachamento unipodal (*Single Leg Squat - SLS*). O sistema processa vídeos gravados nos planos sagital e frontal (lados direito e esquerdo), detectando marcos anatômicos com Google MediaPipe Pose, computando desvios angulares e deslocamentos espaciais, classificando padrões de movimento (incluindo modelos de machine learning pré-treinados para pronação do pé) e gerando relatórios quantitativos em planilhas Excel (.xlsx) e feedback postural corretivo.

## 2. Declaração do Problema
A avaliação postural do agachamento unipodal tradicionalmente depende de inspeção visual subjetiva por fisioterapeutas ou treinadores físicos, ou de sistemas de captura de movimento tridimensionais (MoCap) de custo proibitivo. Erros compensatórios comuns — como valgo dinâmico de joelho, inclinação pélvica, flexão excessiva do tronco, pronação subtalar e elevação precoce do calcanhar — aumentam o risco de lesões no ligamento cruzado anterior (LCA), síndrome da dor patelofemoral e sobrecarga lombar. Existe a necessidade de uma ferramenta automatizada, precisa, reprodutível e acessível que receba vídeos convencionais (MP4), quantifique desvios frame a frame e gere laudos padronizados.

## 3. Objetivos Métricos (SLA / SLO)
* **Taxa de Detecção de Marcos Anatômicos:** $> 95\%$ dos frames com landmarks corporais válidos via MediaPipe Pose Landmarker Full (`pose_landmarker_full.task`).
* **Precisão de Detecção de Repetições:** Identificar de 1 a 3 repetições completas com detecção de ciclo baseada na excursão vertical da orelha (fases: inicial, descendo, subindo, final) com erro inferior a $\pm 1$ repetição em relação ao padrão ouro manual.
* **Tempo de Processamento:** Processar vídeos de até 30 segundos com 30 FPS em menos de 45 segundos em ambiente padrão CPU/Container.
* **Integridade de Exportação de Dados:** $100\%$ das análises finalizadas com planilha Excel exportada contendo abas de status por repetição (-1 não identificado, 0 sem erro, 1 com erro), contagem de frames de erro e regra de decisão por maioria.
* **Segurança de Upload:** Rejeição determinística de arquivos não-MP4 ou superiores a $100\text{ MB}$.

## 4. Usuários-Alvo e Perfis de Acesso
* **Fisioterapeutas e Clínicos:** Realizam upload de filmagens dos pacientes, ajustam thresholds biomecânicos e exportam relatórios para evolução de reabilitação.
* **Treinadores e Avaliadores Físicos:** Monitoram técnica, valgo dinâmico e fadiga muscular em atletas durante o teste de agachamento unipodal.
* **Atletas e Praticantes:** Visualizam feedback corretivo contextualizado (orientações para peito, joelhos, cabeça e pés).
* **Pesquisadores em Biomecânica:** Utilizam as séries temporais brutas e estatísticas (IQR, desvio padrão, média, min, max) exportadas nas pastas `planilhas/` para validação científica.

## 5. Requisitos Funcionais Principais
1. **Upload e Validação Rigorosa de Mídia:** Validador customizado `MP4VideoValidator` checando extensão (`.mp4`), tipo MIME (`video/mp4`, `video/x-m4v`) e limite de $100\text{ MB}$.
2. **Motor de Rastreamento Pose (MediaPipe Tasks):** Extração de coordenadas 3D normalizadas de 33 marcos anatômicos com `PoseDetector` (`pose_landmarker_full.task`).
3. **Análise Biomecânica no Plano Sagital (Lado Direito e Esquerdo):**
   - Calibração antropométrica proporcional: razão normalizada tíbia/estatura x altura real do usuário (`user_height_cm`) calculando `scale_factor_cm`.
   - Inclinação/Flexão do Tronco: cálculo analítico das equações lineares dos eixos do tronco (quadril-ombro) e da tíbia (tornozelo-joelho), identificando o ponto de interseção e avanço excessivo.
   - Translação Anterior do Joelho: verificação do avanço do joelho em relação ao hálux com tolerância proporcional ao comprimento do pé ($19\%$).
   - Alinhamento da Cabeça: cálculo do ângulo absoluto do plano de Frankfurt (orelha-olho) relativo ao horizonte.
   - Elevação do Calcanhar: detecção de elevação do ponto do calcanhar com validação contra falso positivos de proximidade e avanço do tornozelo.
4. **Análise Biomecânica no Plano Frontal (Lado Direito e Esquerdo):**
   - Inclinação Pélvica / Desvio de Quadril: cálculo do ângulo horizontal entre as cristas ilíacas/trocânteres direito e esquerdo.
   - Valgo Dinâmico do Joelho: ângulo tridimensional HKA (Quadril-Joelho-Tornozelo) calculado via produto vetorial e escalar com `arctan2`.
   - Pronação Subtalar do Pé via Machine Learning: extração de descritores estatísticos das coordenadas do pé e inferência via modelos Scikit-Learn (`modelo_pe_frontal_direito.pkl` e `modelo_pe_frontal_esquerdo.pkl`).
5. **Máquina de Estados de Repetição:** Detecção de transição de fases (`inicial`, `descendo`, `subindo`, `final`) por histerese vertical baseada nos limiares `descent_threshold` e `ascent_return_threshold`.
6. **Emissão de Relatórios Estruturados (Excel Writer):** Geração automática de planilhas `.xlsx` via `openpyxl`/`pandas` com consolidação das repetições e julgamento final por critério de maioria.
7. **Interface Gráfica Dupla:** Aplicação Web Django com templates interativos e rota de download, mantendo retrocompatibilidade com protótipo Streamlit (`src/gui/main_app.py`).

## 6. Requisitos Não-Funcionais Críticos
* **Modularidade e Extensibilidade (SOLID):** Separação estrita de contratos com classes base abstratas (`BaseAI`, `BaseFrontal`, `BaseSaggital`, `BaseSquatReportExcelWriter`).
* **Segurança Web Django:** Headers de proteção HTTP Strict Transport Security (`HSTS`), `X-Frame-Options: DENY`, `SECURE_CONTENT_TYPE_NOSNIFF`, CSRF protection e gestão de credenciais via `django-environ`.
* **Containerização:** Empacotamento completo em Docker e Docker Compose com montagem de volumes persistentes para `media` e `db.sqlite3`.
* **Portabilidade de Modelos:** Modelos em formato serializado padronizado (`.task` para MediaPipe e `.pkl` via Joblib para scikit-learn).