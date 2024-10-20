import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Para dividir los datos en entrenamiento y prueba
from tensorflow.python.keras.utils.version_utils import callbacks

# Cargamos el dataset de Fashion MNIST
fmnist = tf.keras.datasets.fashion_mnist

# Cargamos los datos de entrenamiento y prueba | Y separarlos en x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = fmnist.load_data()

# Normalizamos los datos
x_train, x_test = x_train / 255.0, x_test / 255.0

# Dividimos el conjunto de entrenamiento en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2, 
    random_state=42
)

# Arquitectura CNN
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),  # Use Input layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='Conv_layer_1'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),  # First MaxPooling layer

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='Conv_layer_2'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),  # Second MaxPooling layer

    tf.keras.layers.Flatten(name='Flatten_layer'),  # Flatten layer

    tf.keras.layers.Dense(128, activation='relu', name='Hidden_layer_1'),  # Dense hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout layer

    tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer_2'),  # Dense hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout layer

    tf.keras.layers.Dense(32, activation='relu', name='Hidden_layer_3'),  # Dense hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout layer

    tf.keras.layers.Dense(10, activation='softmax', name='Output_layer')  # Output layer
])

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Usamos Adam con tasa de aprendizaje estándar
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compilamos el modelo
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Entrenamos el modelo
model_history = model.fit(x_train,
                          y_train,
                          epochs=20,  # Usamos 20 épocas para balancear entre tiempo de entrenamiento y precisión
                          batch_size=64,  # Tamaño de lote estándar
                          validation_data=(x_val, y_val),
                          callbacks=[early_stopping]
                          )

# Evaluamos el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

model.save('fashion_mnist_cnn.h5')

# Graficamos la historia del entrenamiento
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