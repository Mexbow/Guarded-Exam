from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.create_exam, name='create_exam'),
    path('view/', views.view_exams, name='view_exams'),  # Make sure this matches your view function
    path('view/<int:exam_id>/', views.view_questions, name='view_questions'),  
    path('exams/delete/<int:exam_id>/', views.delete_exam, name='delete_exam'),  # Delete exam URL
    path('question/edit/<int:question_id>/', views.edit_question, name='edit_question'),
    path('question/delete/<int:question_id>/', views.delete_question, name='delete_question'),
    path('toggle-status/<int:exam_id>/', views.toggle_exam_status, name='toggle_exam_status'),
]
