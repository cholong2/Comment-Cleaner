from django.contrib.auth.models import User
from django.db import models


class Filter(models.Model):
    sentence = models.TextField()