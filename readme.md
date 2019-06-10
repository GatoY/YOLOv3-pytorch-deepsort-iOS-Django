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

Under Algorithm/

	python count_light.py --video videofile --debug 1

### RUN on iOS Application

	brew install node

Under Application/backend/,
	
	python manage.py migrate
	python manage.py runserver 
	

## Show Case

![image](demo.gif)