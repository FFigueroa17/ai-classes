import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
IMAGE_PATH = 'dataset'

def load_and_preprocess_images(image_path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        shear_range=0.2,  
        zoom_range=0.2,  
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        image_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_directory(
        image_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator

# def create_model():
#     # Create the model | Architecture
#     model = tf.keras.models.Sequential([
#         # Convolutional layers
#         tf.keras.layers.Input(shape=(64, 64, 3)),  # Input layer
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='Conv_layer_1'), # First Convolutional layer | Only for the first layer
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),  # First MaxPooling layer
# 
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='Conv_layer_2'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),  # Second MaxPooling layer
# 
#         # Fully connected layers
#         tf.keras.layers.Flatten(name='Flatten_layer'),  # Flatten layer
# 
#         tf.keras.layers.Dense(128, activation='relu', name='Hidden_layer_1'),  # Dense hidden layer
#         tf.keras.layers.Dropout(0.3),  # Dropout layer
# 
#         tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer_2'),  # Dense hidden layer
#         tf.keras.layers.Dropout(0.3),  # Dropout layer
# 
#         tf.keras.layers.Dense(3, activation='softmax', name='Output_layer')  # Output layer
#     ])
# 
#     return model

def create_model():
    # Load the VGG16 model
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(150, 150, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    # Create the model
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(name='Flatten_layer'),
        tf.keras.layers.Dense(128, activation='relu', name='Hidden_layer_1'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer_2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax', name='Output_layer') # 3 clases (Bottle, Shoe, Pamela)
    ])

    return model

def compile_and_train_model(model, train_generator, validation_generator):
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Use Adam with standard learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Reduced learning rate

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  # loss = 'ca',
                  metrics=['accuracy'])

    model_history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
    
    return model_history


def evaluate_model(model, validation_generator):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(validation_generator)
    print('Validation accuracy:', test_acc)
    print('Validation loss:', test_loss)

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
    train_generator, validation_generator = load_and_preprocess_images(IMAGE_PATH)
    model = create_model()
    model_history = compile_and_train_model(model, train_generator, validation_generator)
    evaluate_model(model, validation_generator)
    plot_training_history(model_history)
    
    model.save('figueroas_model_cifar10.h5')  # Save the model

if __name__ == "__main__":
    main()