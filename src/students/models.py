from django.db import models
from django.contrib.auth.models import User
from exams.models import Exam, Question
from django.utils.timezone import now

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=255)

    def __str__(self):
        return self.full_name

class StudentExamAttempt(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE)
    score = models.FloatField(default=0)
    submitted_at = models.DateTimeField(auto_now_add=True)
    start_time = models.DateTimeField(default=now)
    end_time = models.DateTimeField(null=True, blank=True)
    
    # New fields for AI analysis tracking
    ai_analysis_completed = models.BooleanField(default=False)
    total_ai_detected = models.IntegerField(default=0)
    total_flagged_answers = models.IntegerField(default=0)

    class Meta:
        unique_together = ('student', 'exam')

    def __str__(self):
        return f"{self.student.full_name} - {self.exam.title} ({self.score})"

class StudentAnswer(models.Model):
    attempt = models.ForeignKey(StudentExamAttempt, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    student_answer = models.TextField()
    is_correct = models.BooleanField(default=False)
    
    ai_detection_result = models.TextField(blank=True, null=True)
    ai_confidence_score = models.FloatField(blank=True, null=True) 
    similarity_score = models.FloatField(blank=True, null=True)    
    is_ai_generated = models.BooleanField(default=False)
    manual_override = models.BooleanField(default=False)
    doctor_notes = models.TextField(blank=True, null=True)
    original_score = models.BooleanField(null=True, blank=True)
    last_modified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    last_modified_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if self.question.question_type == 'mcq':
            self.is_correct = self.student_answer == self.question.answer
        else:
            if self.original_score is None:
                self.original_score = self.is_correct
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.attempt.student.full_name} - Q{self.question.id}"

class AIAnalysisLog(models.Model):
    student_answer = models.ForeignKey(StudentAnswer, on_delete=models.CASCADE, related_name='ai_logs')
    analysis_timestamp = models.DateTimeField(auto_now_add=True)
    ai_detection_raw = models.TextField()  
    similarity_raw = models.TextField()   
    processing_time = models.FloatField(blank=True, null=True)  
    model_version = models.CharField(max_length=100, blank=True) 
    error_log = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"AI Log for {self.student_answer} at {self.analysis_timestamp}"