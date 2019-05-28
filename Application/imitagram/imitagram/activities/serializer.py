from rest_framework import serializers
from imitagram.media.serializers import MediaSerializer
from imitagram.users.serializers import UserSerializer
from .models import Activity

class ActivitySerializer(serializers.ModelSerializer):
    actor = UserSerializer()
    target = UserSerializer()
    obj = MediaSerializer(required=False)
    class Meta:
        model = Activity
        fields = ('id', 'actor', 'target', 'verb', 'obj', 'created_at')