import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo preentrenado para el reconocimiento de emociones
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Inicializar el clasificador de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) que es el rostro
        face_roi = gray[y:y + h, x:x + w]

        # Preprocesamiento de la imagen para el modelo de emociones
        resized_face = cv2.resize(face_roi, (64, 64))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

        # Hacer la predicción con el modelo de emociones
        emotion_pred = emotion_model.predict(reshaped_face)

        # Obtener la etiqueta de emoción predicha
        emotion_label = np.argmax(emotion_pred)
        emotion_text = "Emocion: "

        # Asignar etiquetas de emociones
        if emotion_label == 0:
            emotion_text += "Enojado"
        elif emotion_label == 1:
            emotion_text += "Disgustado"
        elif emotion_label == 2:
            emotion_text += "Temeroso"
        elif emotion_label == 3:
            emotion_text += "Feliz"
        elif emotion_label == 4:
            emotion_text += "Triste"
        elif emotion_label == 5:
            emotion_text += "Sorprendido"
        else:
            emotion_text += "Neutral"

        # Dibujar un recuadro alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Mostrar el resultado en la ventana de la webcam
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen con los recuadros y la etiqueta de emoción
    cv2.imshow('Reconocimiento de emociones', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
