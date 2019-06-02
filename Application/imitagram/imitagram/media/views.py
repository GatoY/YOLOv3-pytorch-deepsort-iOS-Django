from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser
# from django.core.files.storage import FileSystemStorage
from imitagram.users.serializers import UserSerializer
# from imitagram.locations.models import Location
from .models import Image, Media, Comment, Like
from .serializers import CommentSerializer
import subprocess
import datetime


# PATH ='/Users/liuyu/Desktop/'
PATH = '/home/ubuntu/'



@api_view(['POST'])
@permission_classes((IsAuthenticated,))
@parser_classes((MultiPartParser, JSONParser, ))
def upload(request):
    image = Image(standard_resolution=request.data['file'])
    image.save()
    m = Media(image=image)
    m.user = request.user
    
    m.save()
    id = m.id
    date_time = str(datetime.date.today()).split('-')
    year = date_time[0]
    month = date_time[1]
    day = date_time[2]
    # path = '/home/ubuntu/MovingObjectDetecting/Application/imitagram/upload/media/'+year+'/'+month+'/'+day+'/'

    path = PATH+'MovingObjectDetecting/Application/imitagram/upload/media/'+year+'/'+month+'/'+day+'/'
    dir_name = path + request.data['file'].name
    count_script=PATH+'MovingObjectDetecting/Algorithm/count_light.py'
    sys_cmd = ['python3', count_script, '--id',str(id), '--name', dir_name]
    with open('log.txt', 'w+') as f:
        for i in sys_cmd:
            f.write(i)
    # subprocess.Popen(sys_cmd, env={"PATH":PATH+"MovingObjectDetecting/env/bin/"})
    return Response(status=204)


@api_view(['GET', 'POST'])
@permission_classes((IsAuthenticated,))
def comments(request, id):
    if request.method == 'GET':
        data = Comment.objects.filter(media_id=id)
        serializer = CommentSerializer(data, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        media = Media.objects.get(pk=id)
        c = Comment(media=media)
        c.user = request.user
        c.text = request.data['text']
        c.save()
        # TODO
        # update cache counter
        return Response(status=204)

@api_view(['GET', 'POST'])
@permission_classes((IsAuthenticated,))
def likes(request, id):
    if request.method == 'GET':
        data = Like.objects.select_related('user').filter(media_id=id)
        likes = [x.user for x in data]
        serializer = UserSerializer(likes, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        media = Media.objects.get(pk=id)
        if Like.objects.filter(media=media, user=request.user).count() == 0:
            c = Like(media=media)
            c.user = request.user
            c.save()
            # TODO
            # update cache counter
        return Response(status=204)