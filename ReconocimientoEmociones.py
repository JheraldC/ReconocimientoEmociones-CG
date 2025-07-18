import cv2
import os
import numpy as np
import json

def get_data_path():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config['dataPath']

def emotionImage(emotion):
	# Emojis
	if emotion == 'Felicidad': image = cv2.imread('emojis/felicidad.jpeg')
	if emotion == 'Enojo': image = cv2.imread('emojis/enojo.jpeg')
	if emotion == 'Sorpresa': image = cv2.imread('emojis/sorpresa.jpeg')
	if emotion == 'Tristeza': image = cv2.imread('emojis/tristeza.jpeg')
	return image

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')
# --------------------------------------------------------------------------------

dataPath = get_data_path()
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:

	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = emotion_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# EigenFaces
		if method == 'EigenFaces':
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			image = emotionImage(imagePaths[result[0]])
			nFrame = cv2.hconcat([frame,image])
		# FisherFace
		if method == 'FisherFaces':
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			image = emotionImage(imagePaths[result[0]])
			nFrame = cv2.hconcat([frame,image])
		# LBPHFace
		if method == 'LBPH':
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			image = emotionImage(imagePaths[result[0]])
			nFrame = cv2.hconcat([frame,image])

	cv2.imshow('nFrame',nFrame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
