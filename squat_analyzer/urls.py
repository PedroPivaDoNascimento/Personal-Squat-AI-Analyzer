from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('frontal/<str:side>/', views.frontal_analysis, name='frontal_analysis'),
    path('sagittal/<str:side>/', views.sagittal_analysis, name='sagittal_analysis'),
    path('download/<str:analysis_type>/<str:side>/', views.download_excel, name='download_excel'),
]