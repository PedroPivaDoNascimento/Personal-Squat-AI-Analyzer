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
- **Django:** Framework web de alto nível para o desenvolvimento do servidor e painel de controle.
- **Scikit Learn:** Biblioteca para Machine Learning

## 🚀 Instalação

Siga estes passos para configurar e executar o projeto localmente:

1. Clone o repositório:
   ```bash
   git clone https://github.com/PedroPivaDoNascimento/Personal-Squat-AI-Analyzer.git
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd Personal-Squat-AI-Analyzer
   ```

3. Crie e ative um ambiente virtual:
   ```bash
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   venv\Scripts\Activate
   ```

4. Instale as dependências necessárias (certifique-se de que o `django` está listado no arquivo):
   ```bash
   pip install -r requirements.txt
   ```

5. Execute as migrações para configurar o banco de dados do Django:
   ```bash
   python manage.py migrate
   ```

## 🏃 Como usar (sem o docker)

1. Inicialize o servidor de desenvolvimento do Django:
   ```bash
   python manage.py runserver
   ```

2. Abra o seu navegador e acesse o endereço local:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

3. Na interface web do sistema, informe o nome da planilha que será gerada com os resultados e faça o upload do vídeo que será processado.

## 🐋 Como usar (com o docker)

### Pré-requisitos

1. Antes de iniciar, certifique-se de possuir instalado em sua máquina:

- Docker
- Docker Compose (ou Docker Compose Plugin)

Você pode verificar se ambos estão instalados executando:

```bash
docker --version
docker compose version
```

2. Construa as imagens e inicialize os containers:

```bash
docker compose up --build
```

Na primeira execução o processo pode levar alguns minutos, pois as imagens serão construídas e todas as dependências serão instaladas.

Após a inicialização, o sistema estará disponível em:

```
http://localhost:8000
```

ou

```
http://127.0.0.1:8000
```

---


## 📬 Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para entrar em contato comigo em **pedropiva9@gmail.com**.

