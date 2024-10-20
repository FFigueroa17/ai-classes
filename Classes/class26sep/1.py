import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print(y_train[0])

# make a plot to see the data
plt.imshow(x_train[0], cmap='gray')
plt.show()
