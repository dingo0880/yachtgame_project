"""
URL configuration for yachtgame_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views

urlpatterns = [
    # ... 기존 라우트들

    # ML CPU API
    path("api/ml/keep", views.ml_keep_decision, name="ml_keep_decision"),
    path("api/ml/category", views.ml_category_decision, name="ml_category_decision"),
    path("api/ml/health", views.ml_health, name="ml_health"),
]
