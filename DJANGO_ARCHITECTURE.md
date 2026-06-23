"""
Arquitetura Django - Sistema de Análise de Agachamento

ESTRUTURA DE PASTAS PROPOSTA:

squat_analysis_project/           # Raiz do projeto Django
├── manage.py
├── requirements.txt
├── config/                       # Configurações do projeto Django
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── apps/                         # Aplicações Django modulares
│   │
│   ├── core/                     # App principal - Regras de negócio e views
│   │   ├── __init__.py
│   │   ├── models.py             # Modelos Django (sessões de análise)
│   │   ├── views.py              # Views Django (MVT)
│   │   ├── urls.py               # URLs do app core
│   │   ├── forms.py              # Formulários Django
│   │   └── services/             # CAMADA DE SERVIÇO (Business Logic)
│   │       ├── __init__.py
│   │       ├── video_analysis_service.py    # Serviço principal de análise
│   │       ├── squat_analyzer_service.py    # Orquestração da análise
│   │       └── report_generator_service.py  # Geração de relatórios
│   │
│   ├── analysis/                 # App de domínio - Lógica de análise vetorial
│   │   ├── __init__.py
│   │   ├── domain/               # ENTIDADES DE DOMÍNIO
│   │   │   ├── __init__.py
│   │   │   ├── squat_session.py        # Entidade: Sessão de agachamento
│   │   │   ├── repetition.py           # Entidade: Uma repetição
│   │   │   └── body_metrics.py         # Entidade: Métricas corporais
│   │   ├── services/             # SERVIÇOS DE DOMÍNIO
│   │   │   ├── __init__.py
│   │   │   ├── sagittal_analyzer.py      # Análise plano sagital
│   │   │   ├── frontal_analyzer.py       # Análise plano frontal
│   │   │   └── pose_detection_service.py # Detecção de pose
│   │   └── repositories/         # PADRÃO REPOSITORY
│   │       ├── __init__.py
│   │       └── analysis_repository.py    # Repositório de dados de análise
│   │
│   └── reports/                  # App de relatórios - Excel/PDF
│       ├── __init__.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── excel_report_service.py   # Geração de Excel
│       └── generators/                   # GENERATORS (Strategy Pattern)
│           ├── __init__.py
│           ├── base_report_generator.py
│           ├── sagittal_report_generator.py
│           └── frontal_report_generator.py
│
├── shared/                       # MÓDULOS COMPARTILHADOS
│   ├── __init__.py
│   ├── patterns/                 # DESIGN PATTERNS
│   │   ├── __init__.py
│   │   ├── strategy.py           # Base para Strategy Pattern
│   │   └── factory.py            # Factory para criadores de análise
│   ├── utils/                    # Utilitários
│   │   ├── __init__.py
│   │   ├── vector_calculator.py  # Calculadora vetorial (preservada)
│   │   └── feedback_messages.py  # Mensagens de feedback (preservadas)
│   └── exceptions/               # Exceções customizadas
│       └── __init__.py
│
├── static/                       # Arquivos estáticos (CSS, JS)
├── media/                        # Uploads de usuários (vídeos)
├── templates/                    # Templates Django (HTML)
│   ├── base.html
│   ├── core/
│   │   ├── home.html
│   │   ├── analysis_form.html
│   │   └── analysis_result.html
│   └── reports/
│       └── download.html
│
└── tests/                        # Testes unitários e de integração
    ├── __init__.py
    ├── test_services.py
    └── test_domain.py

================================================================================
DESIGN PATTERNS APLICADOS:

1. STRATEGY PATTERN: Para diferentes tipos de análise (Sagital vs Frontal)
   - Cada plano tem sua própria estratégia de análise
   - Facilita adicionar novos planos no futuro

2. REPOSITORY PATTERN: Para abstrair acesso a dados
   - Separa lógica de persistência da lógica de negócio
   - Facilita testes e troca de storage

3. SERVICE LAYER: Para isolar regras de negócio
   - Views Django chamam services, não manipulam lógica direta
   - Services orquestram domínio e repositórios

4. FACTORY PATTERN: Para criação de analisadores
   - Factory decide qual analisador criar baseado no tipo

5. TEMPLATE METHOD: Para geração de relatórios
   - Base define estrutura, subclasses implementam detalhes

PRINCÍPIOS SOLID:

- Single Responsibility: Cada classe tem uma única responsabilidade
- Open/Closed: Classes abertas para extensão, fechadas para modificação
- Liskov Substitution: Subclasses podem substituir classes base
- Interface Segregation: Interfaces específicas por contexto
- Dependency Inversion: Dependência de abstrações, não concretos

================================================================================
"""
