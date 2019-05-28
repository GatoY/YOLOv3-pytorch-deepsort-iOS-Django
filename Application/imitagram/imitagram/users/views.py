from imitagram.users.models import User
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from imitagram.media.models import Media
from imitagram.media.serializers import MediaSerializer
from imitagram.relationships.models import Relationship
from .serializers import UserSerializer, UserDetailsSerializer


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def self(request):
    serializer = UserDetailsSerializer(request.user)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def self_media_recent(request):
    posts = Media.objects.filter(user=request.user).order_by('-created_time')[:10]
    serializer = MediaSerializer(posts, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def self_suggest(request):
    followings = Relationship.objects.filter(source=request.user)
    # TODO
    suggests = []
    count = 0
    for f in followings:
        res = Relationship.objects.filter(source=f.sink)
        if res.count() > 0:
            suggests += [r.sink for r in res if r.sink.id is not request.user.id]
    serializer = UserSerializer(suggests, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def search(request):
    query = request.query_params['q']
    users = User.objects.filter(username__icontains=query)[:10]
    serializer = UserSerializer(users, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def self_feed(request):
    count = 100
    if 'count' in request.query_params:
        count = request.query_params['count']

    following = [x.sink for x in Relationship.objects.filter(source=request.user)]

    if 'max_id' in request.query_params:
        max_id = request.query_params['max_id']
        posts = Media.objects.order_by('-created_time').filter(user_id__in=following, pk__lt=max_id)[:count]
    elif 'min_id' in request.query_params:
        min_id = request.query_params['min_id']
        posts = Media.objects.order_by('-created_time').filter(user_id__in=following, pk__gt=min_id)[:count]
    else:
        posts = Media.objects.order_by('-created_time').filter(user_id__in=following)[:count]
    
    
    serializer = MediaSerializer(posts, many=True)
    return Response(serializer.data)
