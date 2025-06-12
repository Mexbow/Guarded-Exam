from django.contrib import admin
from .models import Doctor

# Register the Doctor model with the admin site
@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone')  # Adjust the fields to be displayed as needed
    search_fields = ('user__email', 'phone')  # Enable search by email and phone
