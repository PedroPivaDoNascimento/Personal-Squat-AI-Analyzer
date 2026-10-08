# Arquitetura do Sistema Personal Squat AI Analyzer

Este documento detalha a arquitetura técnica para processamento de visão computacional, extração de landmarks e análise biomecânica do agachamento unipodal utilizando a stack **Python, Django, MediaPipe Tasks, OpenCV, Scikit-Learn e Docker**.

---

## 1. Visão Geral da Arquitetura (Pipeline em Camadas)

```
[ Cliente Web / Navegador ] --(HTTP Multipart / Form)--> [ Django Controller (views.py) ]
                                                                     |
                                                          [ MP4VideoValidator ]
                                                                     |
                                                        [ SquatAnalysisService ]
                                                                     |
                                          +--------------------------+--------------------------+
                                          |                                                     |
                             [ SagittalAI (Plano Sagital) ]                         [ FrontalAI (Plano Frontal) ]
                                          |                                                     |
                         +----------------+----------------+                   +----------------+----------------+
                         |                                 |                   |                                 |
                 [ PoseDetector ]                  [ Left / Right ]    [ PoseDetector ]                  [ Left / Right ]
            (MediaPipe Pose Tasks)                    Saggital      (MediaPipe Pose Tasks)                   Frontal
                         |                                 |                   |                                 |
                         +----------------+----------------+                   +----------------+----------------+
                                          |                                                     |
                               [ VectorCalculator ]                                   [ VectorCalculator & ML ]
                         (Distâncias, Retas, Ângulos)                        (Ângulo HKA, Joblib / scikit-learn)
                                          |                                                     |
                                          +--------------------------+--------------------------+
                                                                     |
                                                      [ BaseSquatReportExcelWriter ]
                                                  (Frontal & Sagittal Report Writers)
                                                                     |
                                                      +--------------+--------------+
                                                      |                             |
                                            [ Planilha Excel .xlsx ]       [ Resposta HTTP HTML ]
                                            (Diretório `planilhas/`)        (Tabelas, Status, Dicas)
```

---

## 2. Estratégia de Desbloqueio e Concorrência (Gargalos Eliminados)

### 2.1. O Problema do Processamento de Vídeo Pesado
A extração de pose frame a frame com redes neurais profundas (`pose_landmarker_full.task`) e cálculos trigonométricos consecutivos em CPU consome alta memória e tempo de processamento contínuo. Manipulações indevidas de vídeo em memória e locks de escrita em disco causam estouro de buffer (*Out of Memory*).

### 2.2. A Solução: Pipeline Sequencial em Disco Temporário + Desacoplamento Web
1. **Vídeo Temporário Isolado (`NamedTemporaryFile`):** O stream de upload é descarregado de forma segura em arquivo temporário com limpeza determinística em bloco `finally`.
2. **Desacoplamento UI/Core (Clean Architecture):** `SquatAnalysisService` opera como facade independente de framework web, permitindo execução tanto via views Django quanto pela interface legada Streamlit (`src/gui/main_app.py`).
3. **Isolamento de Estado por Execução:** Cada análise instancia seus próprios acumuladores temporais e buffers de erro, eliminando concorrência e vazamento de estado entre diferentes vídeos ou voluntários.

---

## 3. Detalhamento da Stack Técnica

| Camada | Tecnologia | Configuração e Uso |
| :--- | :--- | :--- |
| **Camada Web (Controller/View)** | Django 4.2+ / 6.0 | MVT, templates HTML nativos, rotas de upload e FileResponse para `.xlsx` |
| **Validação de Entrada** | Python / Django Core | `MP4VideoValidator` checando MIME type (`video/mp4`), extensão e tamanho máximo de 100MB |
| **Visão Computacional** | OpenCV (`opencv-python`) | Decodificação de frames de vídeo, cálculo de FPS/tempo e conversão BGR/RGB |
| **Detecção de Pose (IA)** | MediaPipe Tasks (`mediapipe==0.10.14`) | Modelo `pose_landmarker_full.task` para extração de 33 marcos anatômicos 3D |
| **Computação Geométrica** | NumPy & `VectorCalculator` | Cálculo de distâncias euclidianas, equações de retas, interseções e ângulos `arctan2` |
| **Machine Learning** | Scikit-Learn & Joblib | Inferência em descritores estatísticos do pé (`modelo_pe_frontal_direito.pkl` / `esquerdo.pkl`) |
| **Relatórios e Dados** | Pandas & OpenPyXL | Estruturação de séries temporais, estatísticas (IQR, STD) e geração de pastas/planilhas em `planilhas/` |
| **Containerização** | Docker & Docker Compose | Base Python 3.12-slim, dependências de sistema para OpenCV (`libgl1`, `libglib2.0-0`), volumes para mídia e banco |

---

## 4. Dimensionamento e Infraestrutura em Containers

### 4.1. Configuração do Docker Compose (`docker-compose.yml`)
```yaml
services:
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    restart: always
    env_file:
      - .env
    volumes:
      - .:/app
      - sqlite_data:/app/db_data
      - media_volume:/app/media
    ports:
      - "8000:8000"

volumes:
  sqlite_data:
  media_volume:
```

### 4.2. Otimizações de Recursos e Execução
* **Bibliotecas Gráficas Headless:** O processamento em container/servidor roda com flags `draw=False, display=False`, desativando janelas GUI (`cv2.imshow`) e poupando ciclos de CPU.
* **Volume Persistente de Dados:** Montagem persistente de volumes para relatórios gerados e banco SQLite de sessões/autenticação.

---

## 5. Estrutura de Pastas e Dados Gerados

### 5.1. Organização do Diretório `planilhas/`
```text
planilhas/
├── frontal/
│   ├── direito/
│   │   ├── dados_pe/
│   │   │   ├── dados brutos/
│   │   │   └── dados estatisticos/
│   │   └── Relatorio_[Nome]_frontal_direito.xlsx
│   └── esquerdo/
│       ├── dados_pe/
│       │   ├── dados brutos/
│       │   └── dados estatisticos/
│       └── Relatorio_[Nome]_frontal_esquerdo.xlsx
└── sagital/
    ├── direito/
    │   └── Relatorio_[Nome]_sagital_direito.xlsx
    └── esquerdo/
        └── Relatorio_[Nome]_sagital_esquerdo.xlsx
```

### 5.2. Índices e Colunas Críticas das Planilhas de Relatório
1. **Metadados:** `Nome do Voluntário`, `Tipo de Análise`, `Lado Analisado`.
2. **Status por Repetição:** `Repetição 1`, `Repetição 2`, `Repetição 3` (Valores: `-1` = Não Identificado, `0` = Sem Erro, `1` = Com Erro).
3. **Contagem Quantitativa:** `Número de erros Repetição 01`, `02`, `03` (Frames consecutivos que violaram os thresholds).
4. **Decisão por Maioria:** `Resultado` consolidado (`1` se $\ge 2$ repetições apresentaram erro; `-1` se $\ge 2$ não identificadas; `0` se aprovado).