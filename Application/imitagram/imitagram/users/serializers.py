from rest_framework import serializers
from .models import User
from imitagram.media.models import Media
from imitagram.relationships.models import Relationship


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'full_name', 'profile_picture')


class UserDetailsSerializer(serializers.ModelSerializer):
    media = serializers.SerializerMethodField()
    follows = serializers.SerializerMethodField()
    followed_by = serializers.SerializerMethodField()

    def get_media(self, obj):
        return Media.objects.filter(user=obj).count()

    def get_follows(self, obj):
        return Relationship.objects.filter(source=obj).count()

    def get_followed_by(self, obj):
        return Relationship.objects.filter(sink=obj).count()

    class Meta:
        model = User
        fields = ('id', 'username', 'full_name', 'profile_picture', 'media', 'follows', 'followed_by')

