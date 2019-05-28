from imitagram.users.models import User
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from imitagram.users.serializers import UserSerializer
from imitagram.relationships.models import Relationship
from .models import Activity
from .serializer import ActivitySerializer



@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def following(request):
    following = [x.sink for x in Relationship.objects.filter(source=request.user)]
    activities = Activity.objects.order_by('-created_at').filter(actor__in=following)[:10]
    serializer = ActivitySerializer(activities, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def self(request):
    activities = Activity.objects.order_by('-created_at').filter(target=request.user)[:10]
    serializer = ActivitySerializer(activities, many=True)
    return Response(serializer.data)
