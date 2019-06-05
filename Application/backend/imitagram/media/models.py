from django.db import models
from imitagram.users.models import User
from imitagram.locations.models import Location


class Image(models.Model):
    low_resolution = models.ImageField(blank=True, upload_to='media/%Y/%m/%d/')
    thumbnail = models.ImageField(blank=True, upload_to='media/%Y/%m/%d/')
    standard_resolution = models.FileField(upload_to='media/%Y/%m/%d/')


class Media(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.OneToOneField(Image, on_delete=models.CASCADE, null=True)
    location = models.ForeignKey(Location, on_delete=models.PROTECT, null=True)

    finished = models.BooleanField(default=False) #TODO YU
    output = models.CharField(max_length=500, default="") #TODO

    
    person = models.IntegerField(default=0) #TODO
    bicycle = models.IntegerField(default=0) #TODO
    car = models.IntegerField(default=0) #TODO
    motorbike = models.IntegerField(default=0) #TODO
    aeroplane = models.IntegerField(default=0) #TODO
    bus = models.IntegerField(default=0) #TODO
    train = models.IntegerField(default=0) #TODO
    truck = models.IntegerField(default=0) #TODO
    boat = models.IntegerField(default=0) #TODO
    traffic_light = models.IntegerField(default=0) #TODO
    fire_hydrant = models.IntegerField(default=0) #TODO
    stop_sign = models.IntegerField(default=0) #TODO
    parking_meter = models.IntegerField(default=0) #TODO
    bench = models.IntegerField(default=0) #TODO
    bird = models.IntegerField(default=0) #TODO
    cat = models.IntegerField(default=0) #TODO
    dog = models.IntegerField(default=0) #TODO


    comments_count = models.IntegerField(default=0)
    likes_count = models.IntegerField(default=0)
    created_time = models.DateTimeField(auto_now_add=True)


class Comment(models.Model):
    media = models.ForeignKey(Media, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.CharField(max_length=200)
    created_time = models.DateTimeField(auto_now_add=True)


class Like(models.Model):
    media = models.ForeignKey(Media, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_time = models.DateTimeField(auto_now_add=True)

