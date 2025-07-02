from threading import Thread
import tkinter as tk
from tkinter import ttk
import os
import entrenando
from tkinter import messagebox

def abrir_captura_rostros():
    os.system('python capturandoRostrosGUI.py')

def abrir_entrenamiento():
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=20)

    def update_progress(value):
        def update():
            progress_bar['value'] = value
        root.after(0, update)

    def set_complete():
        update_progress(100)  # Asegúrate de que la barra se llene al 100% al final
        messagebox.showinfo("Completado", "El entrenamiento ha finalizado.")
        progress_bar.pack_forget()

    def iniciar_entrenamiento():
        facesData, labels = entrenando.cargar_datos()
        modelos = ['EigenFaces', 'FisherFaces', 'LBPH']
        for i, method in enumerate(modelos):
            entrenando.obtenerModelo(method, facesData, labels)
            update_progress((i + 1) / len(modelos) * 100)
        set_complete()

    Thread(target=iniciar_entrenamiento, daemon=True).start()

def abrir_reconocimiento_emociones():
    os.system('python ReconocimientoEmociones.py')

root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
root.geometry("600x400")  # Ajusta el tamaño de la ventana aquí, por ejemplo, 600x400

# Configura el estilo de los botones
style = ttk.Style()
style.configure("TButton", font=("Arial", 14), background="red", foreground="black")

# Crea los botones con el estilo definido
btn_capturar = ttk.Button(root, text="Capturar Rostros", command=abrir_captura_rostros, style="TButton")
btn_entrenar = ttk.Button(root, text="Entrenar Modelo", command=abrir_entrenamiento, style="TButton")
btn_reconocer = ttk.Button(root, text="Reconocer Emociones", command=abrir_reconocimiento_emociones, style="TButton")

# Alinea los botones en la ventana
btn_capturar.pack(fill='x', padx=20, pady=10)
btn_entrenar.pack(fill='x', padx=20, pady=10)
btn_reconocer.pack(fill='x', padx=20, pady=10)

root.mainloop()