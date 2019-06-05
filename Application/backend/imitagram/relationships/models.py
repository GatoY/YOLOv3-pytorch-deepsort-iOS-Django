from django.db import models
from imitagram.users.models import User


class Relationship(models.Model):
    source = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='source')
    sink = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='sink')
    created_at = models.DateTimeField(auto_now_add=True)
