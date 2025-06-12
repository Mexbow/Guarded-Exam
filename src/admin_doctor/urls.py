from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', views.user_login, name='login'), 
    path('add-doctor/', views.add_doctor, name='add_doctor'),
    path('edit-doctor/<int:doctor_id>/', views.edit_doctor, name='edit_doctor'),
    path('delete-doctor/<int:doctor_id>/', views.delete_doctor, name='delete_doctor'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('doctor-dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('success/', views.success, name='success'),
    path('show-all-doctors/', views.show_all_doctors, name='show_all_doctors'),
    path("doctor-results/", views.doctor_view_results, name="doctor_results"),
    path('student_exam_log/<int:attempt_id>/', views.student_exam_log, name='student_exam_log'),
    path('password-reset/', views.doctor_password_reset_request, name='password_reset_request'),
    path('password-reset/done/', views.doctor_password_reset_done, name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', views.doctor_password_reset_confirm, name='password_reset_confirm'),
    path('password-reset/complete/', views.doctor_password_reset_complete, name='password_reset_complete'),
    path('update_answer_score/<int:answer_id>/', views.update_answer_score, name='update_answer_score'),
]
