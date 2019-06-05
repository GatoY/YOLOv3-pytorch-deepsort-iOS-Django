# Multiple Objects Counting

## About

This project combines YOLOv3(pyTorch) and deepsort. An iOS application is implemented. The backend is based on Django RESTful framework and the frontend is implemented on React Native.

## Key Words

##### YOLOv3, pytorch, Deep SORT, iOS, Django RESTful, React Native

## Installation

	git clone https://github.com/GatoY/MovingObjectDetecting
	pip install -r requirements.txt

## How to Use

### RUN Algorithm Only,

In Algorithm/

	python count_light.py --video videofile

### RUN on iOS Application

	brew install node

In Application/backend/,
	
	python manage.py migrate
	python manage.py runserver 
	

## Show Case

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="path/to/poster_image.png">
    <source src="test.mp4" type="video/mp4">
    <source src="path/to/video.ogg" type="video/ogg">
    <source src="path/to/video.webm" type="video/webm">
  </video>
</figure>