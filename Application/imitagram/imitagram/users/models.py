from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)


def user_profile_picture_path(instance, filename):
    return 'profiles/user_{0}/{1}'.format(instance.id, filename)


class User(AbstractUser):
    full_name = models.CharField(blank=True, max_length=255)
    profile_picture = models.ImageField(blank=True, upload_to=user_profile_picture_path)
    bio = models.CharField(blank=True, max_length=255)
    website = models.CharField(blank=True, max_length=255)
    is_business = models.BooleanField(default=False)

    def __str__(self):
        return self.username

