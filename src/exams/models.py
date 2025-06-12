# exams/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid
from django.core.exceptions import ValidationError


class Exam(models.Model):
    title = models.CharField(max_length=255)
    code = models.CharField(max_length=10, unique=True, editable=False)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(default=30)
    is_active = models.BooleanField(default=True)
    def save(self, *args, **kwargs):
        if not self.code:
            while True:
                random_code = str(uuid.uuid4())[:10] 
                if not Exam.objects.filter(code=random_code).exists():
                    self.code = random_code
                    break
        super().save(*args, **kwargs)


class Question(models.Model):
    QUESTION_TYPES = (
        ('text', 'Text'),
        ('mcq', 'Multiple Choice')
    )

    exam = models.ForeignKey(Exam, related_name='questions', on_delete=models.CASCADE)
    text = models.TextField()
    question_type = models.CharField(max_length=10, choices=QUESTION_TYPES)
    options = models.JSONField(blank=True, null=True) 
    answer = models.CharField(max_length=255, blank=True, null=True)

    def clean(self):
        if self.question_type == 'mcq' and self.answer not in (self.options or []):
            raise ValidationError("MCQ answer must be one of the provided options.")

    def __str__(self):
        return self.text

class ExamSubmission(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE)
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('student', 'exam')