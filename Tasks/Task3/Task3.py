import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # Initialize weights and bias with random values
        self.weights = np.random.rand(input_size)  # Random values for each weight
        self.bias = np.random.rand() # Random values for each bias
        self.learning_rate = learning_rate # Learning rate for the model | Which represents the speed of the learning process
        self.epochs = epochs # Number of iterations to train the model "Epocas"

    def activation_function(self, x):
        # Step function (Heaviside function) 
        # Basic function from the perceptron
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Calculate the linear combination of weights and input + bias
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation_function(linear_output)

    def train(self, X, y):
        """
        Training the perceptron with inputs X (features) and y (labels).
        X is a matrix of input vectors, y is the corresponding target outputs.
        """
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Predict the output
                prediction = self.predict(x_i)

                # Calculate the error
                error = y[idx] - prediction

                # Update the weights and bias based on the error
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

        print("Training complete.")
        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias}")

# Example usage:

# Try for AND gate truth table
# X represents inputs and y represents the output (0 or 1)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([0, 0, 0, 1])  # AND gate

# Try for OR gate
y_or = np.array([0, 1, 1, 1])  # OR gate

# Create the Perceptron model
perceptron = Perceptron(input_size=2, learning_rate=0.4, epochs=10)

# Train the perceptron
# perceptron.train(X, y_and)
perceptron.train(X, y_or)

# Test the trained perceptron
print("Predictions:")
for x in X:
    print(f"Input: {x}, Prediction: {perceptron.predict(x)}")