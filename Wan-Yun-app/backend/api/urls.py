# api/urls.py
from django.urls import include, path, re_path
from django.conf.urls import include, url
from django.contrib import admin

from django.http import HttpResponse

def empty_view(request):
    return HttpResponse('')

urlpatterns = [
               path('users/', include('users.urls')),
               path('rest-auth/', include('rest_auth.urls')),
               path('rest-auth/registration/', include('rest_auth.registration.urls')),
               
               path('password-reset/<uidb64>/<token>/', empty_view, name='password_reset_confirm'),
               
               ]
               
