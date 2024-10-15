from django.urls import path
from . import views

urlpatterns =[
    path('predict/', views.predictor, name='predict'),
    path('', views.predictor, name='predictor'),
]