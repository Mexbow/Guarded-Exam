from django.contrib import admin
from .models import Student, StudentExamAttempt, StudentAnswer

admin.site.register(Student)
admin.site.register(StudentExamAttempt)
admin.site.register(StudentAnswer)
