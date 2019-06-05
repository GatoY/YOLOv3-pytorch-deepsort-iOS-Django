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

<object width="425" height="350">
  <param name="movie" value="http://www.youtube.com/user/wwwLoveWatercom?v=BTRN1YETpyg" />
  <param name="wmode" value="transparent" />
  <embed src="https://www.youtube.com/watch?v=ERyFKAjGeZM&feature=youtu.be"
         type="application/x-shockwave-flash"
         wmode="transparent" width="425" height="350" />
</object>