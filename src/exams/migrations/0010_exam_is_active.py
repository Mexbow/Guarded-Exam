# Generated by Django 5.1.3 on 2025-06-10 15:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('exams', '0009_exam_duration'),
    ]

    operations = [
        migrations.AddField(
            model_name='exam',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
    ]
