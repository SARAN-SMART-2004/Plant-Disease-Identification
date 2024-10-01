# Generated by Django 5.0.6 on 2024-09-17 07:55

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="UploadedImage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="uploads/")),
                (
                    "upload_time",
                    models.DateTimeField(default=django.utils.timezone.now),
                ),
            ],
        ),
    ]
