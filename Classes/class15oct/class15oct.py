import cv2 as cv
import numpy as np
import tf

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se puede recibir el fotograma. Saliendo...")
        break
        
    height, width, _ = frame.shape
    img = cv.resize(frame, (32,32))
    
    # Transformar la imagen a escala de grises
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Tranformar a batch | Añadir una dimensión extra
    img_batch = np.expand_dims(img_array, axis=0) # Añadir una dimensión extra
    
    prediction = model.predict(img_batch) # TODO: Cambiar por el modelo que se desee usar
    # prediction = prediction[0]
    
    class_prediction = class_name[np.argmax(prediction)] # TODO: Cambiar por las clases que se deseen usar
    print(f'Predicción: {class_prediction}')
    
    cv.putText(frame, class_prediction, (width/2, height/2), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), cv.LINE_AA)
    
    cv.imshow('Prediccion en tiempo real (Cifar-10), frame')
        
    cv.imshow('Fabuloso titulo', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()

