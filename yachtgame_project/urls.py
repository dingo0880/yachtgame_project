# yachtgame_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('yachtgame.urls')),  # ✅ 앱 라우팅만 포함
]
