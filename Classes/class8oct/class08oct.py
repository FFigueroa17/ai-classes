import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Cargar el modelo guardado
model = tf.keras.models.load_model('fashion_mnist_cnn.h5')

# Clases de las etiquetas
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Cargar el dataset de Fashion MNIST
fmnist = tf.keras.datasets.fashion_mnist
(_, _), (x_test, y_test) = fmnist.load_data()

# # Normalizar los datos
# x_test = x_test / 255.0
 
# # Seleccionar una imagen del conjunto de prueba
# index = 0  # Puedes cambiar este índice para probar con diferentes imágenes
# image = x_test[index]
# label = y_test[index]
 
# # Preprocesar la imagen para que tenga el mismo formato que las imágenes usadas durante el entrenamiento
# image = image.reshape(1, 28, 28, 1)  # Añadir batch dimension y canal

# Load and preprocess the real image
image_path = 'image_tests/ankle_boot_test.webp'  # Replace with the path to your image
image = Image.open(image_path).convert('L')  # Convert to grayscale
image = image.resize((28, 28))  # Resize to 28x28 pixels
image = np.array(image) / 255.0  # Normalize the image
image = image.reshape(1, 28, 28, 1)  # Add batch dimension and channel

# Predict the class of the image
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

# Display the image and the prediction
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {class_name[predicted_class]} - {predicted_class}')
plt.show()

# # Utilizar el modelo para predecir la clase de la imagen
# predictions = model.predict(image)
# predicted_class = np.argmax(predictions)
# 
# # Mostrar la imagen y la predicción
# plt.imshow(x_test[index], cmap='gray')
# plt.title(f'Predicted: {predicted_class}, Actual: {label}')
# plt.show()