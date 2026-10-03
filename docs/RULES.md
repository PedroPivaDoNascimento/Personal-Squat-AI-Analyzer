# Regras de Desenvolvimento e Padrões de Código

**Personal Squat AI Analyzer – Diretrizes Estruturais para Processamento Biomecânico e Aplicação Web**

--- 

## Observação Importante!!!

Antes de começar qualquer task ou alteração no código-fonte, é necessária a **leitura obrigatória** dos arquivos `docs/ARCHITECTURE.md`, `docs/DESIGN.md` , `docs/PRD.md` e `docs/GEMINI.md` , garantindo que toda modificação respeite rigorosamente os contratos arquiteturais e biomecânicos estabelecidos.

---

## 1. Regras Absolutas de Visão Computacional e Matemática Vetorial

1. **Cálculos Trigonométricos e Vetoriais:**
   - Todos os cálculos de distância espacial, equações de retas, projeções e ângulos articulares DEVEM ser centralizados em `VectorCalculator` (`src/classes/vector_calculator.py`).
   - Ângulos com sentido ou orientação articular (ex: ângulo HKA de joelho valgo) DEVEM usar produto vetorial e produto escalar com `np.arctan2` ou `math.atan2`, retornando graus no intervalo $[-180.0, +180.0]$. NUNCA use aproximações empíricas não documentadas.
2. **Normalização e Escala Antropométrica:**
   - No plano sagital, medições dependentes de comprimento real em centímetros (ex: comprimento da tíbia, avanço do joelho) DEVEM respeitar o fator de escala `scale_factor_cm`, derivado da estatura real fornecida pelo usuário (`user_height_cm`) e da altura normalizada capturada no vídeo.
3. **Manejo de Recursos do MediaPipe e OpenCV:**
   - Toda captura de vídeo com `cv2.VideoCapture` e toda instância de `PoseDetector` DEVEM ser explicitamente liberadas e fechadas (`cap.release()`, `cv2.destroyAllWindows()`, `pose_detector.close()`) dentro de blocos `finally`.

---

## 2. Padrões Obrigatórios da Camada Web (Django & Service Layer)

1. **Desacoplamento de Regra de Negócio (Princípio da Responsabilidade Única):**
   - As views Django (`squat_analyzer/views.py`) atuam estritamente como **Controllers**: recebem requisições HTTP, executam validações de formulário e delegam o processamento ao `SquatAnalysisService`.
   - NUNCA instancie diretamente `FrontalAI` ou `SagittalAI` dentro das views do Django.
2. **Validação de Entrada de Vídeo:**
   - Todo upload de arquivo DEVE passar pela validação de `MP4VideoValidator` (`squat_analyzer/validators/file_validators.py`). Arquivos não-MP4, MIME types divergentes ou cargas superiores a $100\text{ MB}$ DEVEM ser barrados com `ValidationError` amigável.
3. **Manejo de Arquivos Temporários:**
   - Vídeos enviados para análise DEVEM ser gravados em arquivos temporários (`tempfile.NamedTemporaryFile`) e DELETADOS imediatamente após o término do processamento no bloco `finally` do serviço.

---

## 3. Padrões de Geração de Dados e Relatórios (Excel Writer)

1. **Estrutura de Pastas Padronizada:**
   - Pastas de saída DEVEM seguir o padrão gerenciado por `SetFolders`: `planilhas/<plano>/<lado>/`.
2. **Consistência de Codificação de Status:**
   - Nas tabelas e planilhas, os status de cada repetição seguem estritamente:
     - `-1`: Não Identificado / Landmark ausente / Movimento incompleto.
     - `0`: Execução Adequada (Sem Erro).
     - `1`: Desvio Biomecânico Confirmado (Com Erro).
3. **Regra de Decisão por Maioria:**
   - O resultado final consolidado para cada segmento corporal deve computar a maioria das 3 repetições avaliadas:
     - Se contagem de `-1` $\ge 2 \rightarrow$ Resultado `= -1`.
     - Se contagem de `1` $\ge 2 \rightarrow$ Resultado `= 1`.
     - Caso contrário $\rightarrow$ Resultado `= 0`.

---

## 4. Padrões para Modelos de Machine Learning e Serialização

1. **Carregamento Seguro de Modelos:**
   - Modelos `.pkl` devem ser carregados via `joblib.load()` apontando para caminhos resolvidos a partir de `models/` usando referências absolutas ou relativas baseadas em `os.path.dirname(__file__)`.
2. **Compatibilidade com MediaPipe Tasks:**
   - O modelo `pose_landmarker_full.task` deve ser mantido no diretório `models/` e instanciado com `vision.PoseLandmarker.create_from_model_path`.