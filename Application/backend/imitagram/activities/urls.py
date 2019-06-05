from django.urls import path

from . import views

urlpatterns = [
    path('following', views.following),
    path('self', views.self)
]