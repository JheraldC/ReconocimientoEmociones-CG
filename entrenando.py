import cv2
import os
import numpy as np
import time
import json

def get_data_path():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config['dataPath']

def cargar_datos():
	dataPath = get_data_path()
	emotionsList = os.listdir(dataPath)
	print('Lista de personas: ', emotionsList)

	labels = []
	facesData = []
	label = 0
	for nameDir in emotionsList:
		emotionsPath = os.path.join(dataPath, nameDir)
		for fileName in os.listdir(emotionsPath):
			#print('Rostros: ', nameDir + '/' + fileName)
			labels.append(label)
			facesData.append(cv2.imread(os.path.join(emotionsPath, fileName), 0))
			#image = cv2.imread(emotionsPath+'/'+fileName,0)
			#cv2.imshow('image',image)
			#cv2.waitKey(10)
		label += 1
	return facesData, labels

def obtenerModelo(method, facesData, labels, update_progress=None, set_complete=None):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()

	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+".xml")	

if __name__ == "__main__":
	facesData, labels = cargar_datos()
	obtenerModelo('EigenFaces',facesData,labels)
	obtenerModelo('FisherFaces',facesData,labels)
	obtenerModelo('LBPH',facesData,labels)
	