"""
Core Django Application - Main application for handling user interface and orchestration.

Esta aplicação é responsável por:
- Receber requisições HTTP dos usuários
- Orquestrar o fluxo de análise através dos services
- Renderizar templates HTML para exibição dos resultados

Arquitetura:
- Views: Controladores Django (MVT pattern)
- Services: Camada de serviço com regras de negócio
- Forms: Validação de dados de entrada
- Models: Modelos Django para persistência
"""
