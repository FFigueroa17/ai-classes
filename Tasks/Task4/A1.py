import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Para dividir los datos en entrenamiento y prueba

# Cargamos el dataset de Fashion MNIST
fmnist = tf.keras.datasets.fashion_mnist

# Cargamos los datos de entrenamiento y prueba | Y separarlos en x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = fmnist.load_data()

# Clases de las etiquetas
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Paso4: Normalizamos los datos - En este caso, tenemos datos del 0 al 255, por lo que dividimos entre 255 para tener datos entre 0 y 1
x_train, x_test = x_train / 255, x_test / 255

print(f'x_train: {x_train.shape}')
print(f'x_test: {x_test.shape}')

#Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=42
                                                  )

# print("Finales")
# print(f'Datos de entrenamiento: {x_train.shape}')
# print(f'Datos de validación: {x_val.shape}')
# print(f'Datos de prueba: {x_test.shape}')
# 
# data_size = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
# #Actividad de hoy, que al lado diga el porcentaje de cada uno
# print("Porcentajes finales")
# print(f'Porcentaje de datos de entrenamiento: {np.round(x_train.shape[0]/data_size * 100, 2)} %')
# print(f'Porcentaje de datos de validación: {np.round(x_val.shape[0]/data_size * 100, 2)} %')
# print(f'Porcentaje de datos de prueba: {np.round(x_test.shape[0]/data_size * 100, 2)} %')

# Arquitectura del modelo de la red neuronal

model = tf.keras.models.Sequential([
    # Capa de entrada
    tf.keras.Input(shape=(28, 28)),  # Use Input layer
    tf.keras.layers.Flatten(name='Input_layer'),
    # Capa oculta
    tf.keras.layers.Dense(128, activation='relu', name='Hidden_layer'),

    # Capa oculta
    tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer_1'),

    # Capa de salida
    tf.keras.layers.Dense(10, activation='softmax', name='Output_layer') 
])

# Usamos Adam con una tasa de aprendizaje menor para mejorar los resultados
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

#Compilar el modelo
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Entrenar el modelo
model_history = model.fit(x_train,
                          y_train,
                          epochs=15, # 15 epochs
                          batch_size=64,
                          validation_data=(x_val, y_val)
                          )

#Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Plot the training history
plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()