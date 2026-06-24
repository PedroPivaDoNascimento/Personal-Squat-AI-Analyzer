import os
import sys
from pathlib import Path

# Adicionar 'src' ao path também no WSGI
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'squat_analysis_app.settings')

application = get_wsgi_application()