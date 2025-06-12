# exams/admin.py

from django.contrib import admin
from .models import Exam, Question  # Import your models

# Register the Exam model to make it visible in the admin interface
admin.site.register(Exam)
admin.site.register(Question)  # Register the Question model if you want to manage questions in the admin
