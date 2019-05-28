from rest_framework import serializers
from django.contrib.auth.models import User
from imitagram.users.serializers import UserSerializer
from imitagram.locations.serializers import LocationSerializer
from .models import Image, Media, Comment

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ('low_resolution', 'thumbnail', 'standard_resolution')


class MediaSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    image = ImageSerializer()
    location = LocationSerializer(required=False)
    comments = serializers.SerializerMethodField()
    likes = serializers.SerializerMethodField()

    class Meta:
        model = Media
        fields = ('id', 'user', 'image', 'finished', 'output',
                'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 
                'bus', 'train', 'truck', 'boat', 'traffic_light', 
                'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',
                'bird', 'cat', 'dog', 'comments', 'likes', 'location', 'created_time')

    def get_comments(self, obj):
        return {'count': obj.comments_count}

    def get_likes(self, obj):
        return {'count': obj.likes_count}


class CommentSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    class Meta:
        model = Comment
        fields = ('id', 'user', 'text', 'created_time')
        