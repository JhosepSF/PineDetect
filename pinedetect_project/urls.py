"""Enrutamiento URL principal del proyecto PineDetect."""

from django.urls import path, include

urlpatterns = [
    path('', include('detector.urls')),
]
