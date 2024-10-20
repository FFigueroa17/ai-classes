import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Import MNIST dataset
    mnist = tf.keras.datasets.cifar10

    # Separate the dataset into training and testing data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the training data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_model():
    # Create the model | Architecture
    model = tf.keras.models.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='Conv_layer_1', input_shape=(32, 32, 3)), # First Convolutional layer | Only for the first layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),  # First MaxPooling layer

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='Conv_layer_2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),  # Second MaxPooling layer

        # Fully connected layers
        tf.keras.layers.Flatten(name='Flatten_layer'),  # Flatten layer

        tf.keras.layers.Dense(128, activation='relu', name='Hidden_layer_1'),  # Dense hidden layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer

        tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer_2'),  # Dense hidden layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer

        tf.keras.layers.Dense(10, activation='softmax', name='Output_layer')  # Output layer
    ])

    return model

def compile_and_train_model(model, x_train, y_train, x_val, y_val):
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Use Adam with standard learning rate
    optimizer = tf.keras.optimizers.Adam()

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # Train the model
    model_history = model.fit(x_train,
                              y_train,
                              epochs=20,  # Use 20 epochs to balance between training time and accuracy
                              batch_size=64,  # Standard batch size
                              validation_data=(x_val, y_val),
                              callbacks=[early_stopping]
                              )

    return model_history

def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

def plot_training_history(model_history):
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

def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    model = create_model()
    model_history = compile_and_train_model(model, x_train, y_train, x_val, y_val)
    evaluate_model(model, x_test, y_test)
    plot_training_history(model_history)

if __name__ == "__main__":
    main()