from django.db import models
from django.contrib.auth.models import User

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15)
    # Add any other fields you need

    def __str__(self):
        return f"Dr. {self.user.first_name} {self.user.last_name}"
