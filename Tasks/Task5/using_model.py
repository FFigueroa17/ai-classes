import tensorflow as tf
import numpy as np
import cv2 as cv

#     # Load and preprocess the image
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
# 
#     # Make the prediction
#     prediction = model.predict(img_array)
#     class_names = ['Bottle', 'Shoe']  # Update class names
# 
#     predicted_class = class_names[np.argmax(prediction)]
#     print(f"Prediction: {predicted_class}")
# 
#     # Plot the image with the predicted class
#     plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
#     plt.title(f"Prediction: {predicted_class}")
#     plt.axis('off')
#     plt.show()
# 
# # Example prediction and plot
# predict_and_plot_image(model, 'images/shoe_test.jpeg')

def real_time_prediction(model_path, class_names):
    model = tf.keras.models.load_model(model_path)
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
        img = cv.resize(frame, (64, 64))  # Resize to (64, 64)

        # Transform the image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Add an extra dimension to the batch
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch /= 255.0  # Normalize the image

        prediction = model.predict(img_batch)
        predicted_index = np.argmax(prediction)

        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
        else:
            predicted_class = "objeto desconocido"

        print(f'Predicción: {predicted_class}')

        cv.putText(frame, predicted_class, (width // 2, height // 2), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Prediccion en tiempo real', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
# Example usage
model_path = 'figueroas_model_cifar10.h5'
class_names = ['Bag', 'Bottle' ,'Pamela', 'Shoe']
real_time_prediction(model_path, class_names)