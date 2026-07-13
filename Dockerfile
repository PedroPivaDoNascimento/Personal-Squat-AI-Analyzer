FROM python:3.12-slim

# Define variáveis de ambiente para evitar buffers de output e warnings do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cria o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para o OpenCV e outras libs
# (libgl1 e libglib2.0-0 são essenciais para o OpenCV funcionar no Docker)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia todo o resto do código do projeto para dentro do container
COPY . .

# Expõe a porta que o Django vai usar
EXPOSE 8000

# Define o comando para rodar o servidor Django
# Em produção, trocaremos isso por um servidor mais robusto
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]