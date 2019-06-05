from django.urls import path

from . import views

urlpatterns = [
    # path('', views.QustionList.as_view()),
    path('self/feed', views.self_feed),
    path('self/suggest', views.self_suggest),
    path('search', views.search),
    path('self/media/recent', views.self_media_recent),
    path('self', views.self)
]