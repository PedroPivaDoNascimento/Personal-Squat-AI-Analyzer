"""
URLs da aplicação core.

Mapeia as URLs para as views, substituindo a navegação por sessões do Streamlit.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Página inicial (substitui show_selection_page)
    path('', views.index, name='index'),
    
    # Análises Sagitais
    path('sagittal/right/', views.sagittal_right_analysis, name='sagittal_right'),
    path('sagittal/left/', views.sagittal_left_analysis, name='sagittal_left'),
    
    # Análises Frontais
    path('frontal/right/', views.frontal_right_analysis, name='frontal_right'),
    path('frontal/left/', views.frontal_left_analysis, name='frontal_left'),
]
