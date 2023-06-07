from django.urls import path
from . import views

urlpatterns = [path("chat_bot/", views.chat_bot), path("ocr/", views.get_ocr)]
