import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data.csv')

data = data.drop(['Unnamed: 32', 'id'], axis=1)

print(data.head()) 
# print(data.describe())

# 1. Transform data
# The diagnosis column is a string, we need to convert it to a number
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# 2. Split data
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 3. Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
# Define the model
model = LogisticRegression(
    max_iter=10000
)

# 4. Fit | Train model
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

print()
# 4. Evaluate the model (Metrics)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# 5. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, cmap='Blues', annot=True, xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])

print()
print('-- Confusion Matrix --')

# Todo: Calculate the accuracy, specificity, sensitivity, negative predictive value, precision
# Calculate the accuracy
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
print(f'Accuracy: {accuracy}')
print()

# Calculate Specificity
specificity = conf_matrix[0, 0] / (conf_matrix[1, 1] + conf_matrix[1, 0])
print(f'Specificity: {specificity}')
print()

# Calculate Sensitivity
sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print(f'Sensitivity: {sensitivity}')
print()

# Calculate Negative Predictive Value
npv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
print(f'Negative Predictive Value: {npv}')
print() 

# Calculate the precision
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
print(f'Precision: {precision}')
print()

# Classification Report
print()
print('-- Classification Report --')
print(classification_report(y_test, y_pred))
print()

# Show the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

# 5. Plot the model
# sns.scatterplot(x=X_test['radius_mean'], y=y_test)
# sns.lineplot(x=X_test['radius_mean'], y=y_pred, color='red')
# plt.xlabel('Kilometers')
# plt.ylabel('Miles')
# plt.title('Kilometers x Miles')
# plt.grid(True)
plt.show()

print(f'Real values: {y_test.values}')
print(f'Predicted values: {y_pred}')

print(data.head())