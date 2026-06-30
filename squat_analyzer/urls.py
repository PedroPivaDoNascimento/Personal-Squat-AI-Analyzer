from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('frontal/esquerdo/', views.frontal_left_analysis, name='frontal_left_analysis'),
    path('frontal/direito/', views.frontal_right_analysis, name='frontal_right_analysis'),
    path('sagittal/esquerdo/', views.sagittal_left_analysis, name='sagittal_left_analysis'),
    path('sagittal/direito/', views.sagittal_right_analysis, name='sagittal_right_analysis'),
    path('download/<str:analysis_type>/<str:side>/', views.download_excel, name='download_excel'),
]