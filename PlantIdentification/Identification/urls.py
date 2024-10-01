# Identification/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('save-image/', views.save_image, name='save_image'),
    path('capture/', views.cpimage, name='capture_image'),
]
