from django.urls import path
from . import views

urlpatterns = [
    path("", views.enter_exam_code, name="enter_exam_code"),
    path("take-exam/<int:exam_id>/", views.take_exam, name="take_exam"),
    path("exam-results/<int:attempt_id>/",views.exam_results, name="exam_results"),
    path("exam-congrats/<int:attempt_id>/", views.exam_congrats, name="exam_congrats"),
]
