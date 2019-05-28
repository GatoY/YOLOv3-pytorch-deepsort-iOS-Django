from imitagram.users.models import User
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from .models import Relationship


@api_view(['POST'])
@permission_classes((IsAuthenticated,))
def follow(request):
    edge = Relationship()
    edge.source = request.user
    edge.sink = User.objects.get(pk=request.data['follow'])
    edge.save()
    return Response(status=204)
