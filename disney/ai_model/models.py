from django.db import models

# Create your models here.
class Face_Image(models.Model):
    face = models.ImageField(upload_to="face")