from django.db import models


# Create your models here.
class TablaTotales(models.Model):
    alumno = models.CharField(max_length=50)
