#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pathlib  # <--- ADICIONAR

def main():
    """Run administrative tasks."""
    # Configurar o diretório base do projeto
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    
    # Adicionar a pasta 'src' ao PYTHONPATH para importar 'classes', 'gui', etc.
    SRC_DIR = BASE_DIR / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))  # <--- ADICIONAR

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'squat_analysis_app.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()