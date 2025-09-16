# Personal Squat AI Analyzer

Este é um aplicativo de visão computacional em tempo real que utiliza inteligência artificial para analisar a forma de agachamento de um usuário, fornecendo feedback instantâneo para ajudar a prevenir lesões e melhorar a técnica.

## 🌟 Recursos

- **Análise em Tempo Real:** Processa quadros de vídeo em tempo real para feedback instantâneo sobre a forma.
- **Detecção de Postura:** Utiliza modelos de IA para detectar pontos-chave do corpo e calcular ângulos de articulação.
- **Contador de Repetições:** Conta automaticamente o número de agachamentos concluídos.
- **Feedback de Forma:** Fornece alertas para problemas comuns, como:
  - Joelhos que se projetam para a frente além dos dedos dos pés.
  - Costas arredondadas.
  - Calcanhares se levantando do chão.
- **Compatibilidade:** Suporta análise de um arquivo de vídeo pré-gravado.

## 💻 Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **OpenCV:** Biblioteca de visão computacional para processamento de vídeo e imagem.
- **MediaPipe:** Framework do Google para detecção de pontos de referência do corpo humano (pose estimation).

## 🚀 Instalação

Siga estes passos para configurar e executar o projeto localmente:

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/PedroPivaDoNascimento/Personal-Squat-AI-Analyzer.git](https://github.com/PedroPivaDoNascimento/Personal-Squat-AI-Analyzer.git)
    ```
2.  Navegue até o diretório do projeto:
    ```bash
    cd Personal-Squat-AI-Analyzer
    ```
3.  Instale as dependências necessárias. Recomenda-se o uso de um ambiente virtual.
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Como usar

1.  Execute o script principal para iniciar a análise (o nome do arquivo pode variar, verifique o seu repositório):
    ```bash
    streamlit run src/main.py
    ```
2.  Coloque o nome da planilha que será gerada com os resultados e o vídeo que será processado.

## 📬 Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para abrir uma issue neste repositório ou entrar em contato comigo em pedropiva9@gmail.com.
