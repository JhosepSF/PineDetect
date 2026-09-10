"""Configuración ASGI para el proyecto PineDetect."""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pinedetect_project.settings')

application = get_asgi_application()
