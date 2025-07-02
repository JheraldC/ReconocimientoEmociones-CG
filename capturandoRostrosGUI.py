from threading import Thread
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import os
import imutils
import json

def get_data_path():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config['dataPath']

def set_data_path(new_path):
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    config['dataPath'] = new_path
    with open('config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

class CapturaRostrosGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Captura de Rostros")

        # Opciones de emociones
        self.emociones = ["Felicidad", "Tristeza", "Sorpresa", "Enojo"]
        self.emocion_seleccionada = tk.StringVar(value=self.emociones[0])

        # Configuración del DataPath
        self.dataPath = tk.StringVar(value=get_data_path())

        self.setup_gui()
        self.capturing = False

    def setup_gui(self):
        self.master.geometry('400x200')  # Ajusta el tamaño de la ventana

        # Configura un layout de cuadrícula para un mejor control del posicionamiento
        row = 0
        tk.Label(self.master, text="Seleccionar Emoción:").grid(row=0, column=0, sticky='w')
        tk.OptionMenu(self.master, self.emocion_seleccionada, *self.emociones).grid(row=0, column=1, padx=5, pady=5)

        row += 1
        tk.Label(self.master, text="DataPath:").grid(row=1, column=0, sticky='w')
        tk.Entry(self.master, textvariable=self.dataPath).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.master, text="Cambiar DataPath", bg='red', fg='white', command=self.cambiar_dataPath).grid(row=1, column=2, padx=5, pady=5)

        self.start_button = tk.Button(self.master, text="Iniciar Captura", bg='red', fg='white', command=self.iniciar_captura)
        self.start_button.grid(row=2, column=0, padx=5, pady=5)
        self.stop_button = tk.Button(self.master, text="Detener Captura", bg='red', fg='white', command=self.detener_captura, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, padx=5, pady=5)

        # Centra los elementos en la ventana
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=1)
        self.master.grid_rowconfigure(3, weight=1)  # Asegúrate de tener al menos un renglón más de los que usas
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

    def cambiar_dataPath(self):
        path = filedialog.askdirectory()
        if path:
            self.dataPath.set(path)
            set_data_path(path)

    def iniciar_captura(self):
        self.capturing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.thread = Thread(target=self.captura_continua, daemon=True)
        self.thread.start()

    def captura_continua(self):
        emotionName = self.emocion_seleccionada.get()
        dataPath = self.dataPath.get()
        emotionsPath = os.path.join(dataPath, emotionName)
        if not os.path.exists(emotionsPath):
            os.makedirs(emotionsPath)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.mostrar_error("No se pudo acceder a la cámara.")
            self.finalizar_captura()
            return

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0

        while self.capturing and count < 200:
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = gray[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(emotionsPath, f'rostro_{count}.jpg'), rostro)
                count += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.finalizar_captura()

    def mostrar_error(self, mensaje):
        # Usamos 'after' para asegurarnos de que la messagebox se ejecute en el hilo principal
        self.master.after(0, messagebox.showerror, "Error", mensaje)

    def finalizar_captura(self):
        # Usamos 'after' para cambiar el estado de los botones en el hilo principal
        self.master.after(0, self.detener_captura)

    def detener_captura(self):
        self.capturing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = CapturaRostrosGUI(root)
    root.mainloop()