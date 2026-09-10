"""Rutas de la aplicación detector."""

from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/status/', views.status_api_view, name='api_status'),
    path('api/analyze/', views.analyze_api_view, name='api_analyze'),
    path('api/export/csv/', views.export_csv_api_view, name='api_export_csv'),
    path('api/export/json/', views.export_json_api_view, name='api_export_json'),
]
