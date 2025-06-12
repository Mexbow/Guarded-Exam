from django.urls import path
from . import views
app_name = 'main'  # This creates the namespace

urlpatterns = [
    path('', views.home, name='home'),  # URL would be 'main:home'
]