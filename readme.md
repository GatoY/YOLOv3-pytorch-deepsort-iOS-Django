# Multiple Objects Counting

## About

This project combines YOLOv3(pyTorch) and deepsort. An iOS application is implemented. The backend is based on Django RESTful framework and the frontend is implemented on React Native.

## Key Words

##### YOLOv3, pytorch, Deep SORT, iOS, Django RESTful, React Native

## Installation

	git clone https://github.com/GatoY/MovingObjectDetecting
	pip install -r requirements.txt

    brew install node
    brew install watchman
    install yarn
    npm install -g react-native-cli
    yarn add react-navigation
    react-native link react-navigation
    yarn add react-native-vector-icons
    react-native link react-native-vector-icons
    yarn add react-native-image-picker
    react-native link react-native-image-picker
    yarn add react-native-video
    react-native link react-native-video
    react-native link
    
## How to Use

### RUN Algorithm Only,

Under Algorithm/

	python count_light.py --video videofile --debug 1

### RUN on iOS Application (on MAC)
        
Under Application/backend/,
	
	python manage.py migrate
	python manage.py runserver 

Under Application/iOS/project-tracking/:

    react-native run-ios	

## Show Case

![image](demo.gif)

App Demo video: https://youtu.be/R9mIF3EtYI4



