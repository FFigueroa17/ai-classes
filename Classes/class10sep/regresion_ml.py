from cgi import print_form

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.odr import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('dataset_km_miles.csv')
print(data.head()) 

X = data.drop('miles', axis= 1)
y = data['miles']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.2, 
    random_state=42
)

# 1. Create the model
model = LinearRegression()

# 2. Train the model
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate the model (Metrics)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# 5. Plot the model
sns.scatterplot(x=X_test['km'], y=y_test)
sns.lineplot(x=X_test['km'], y=y_pred, color='red')
# plt.xlabel('Kilometers')
# plt.ylabel('Miles')
# plt.title('Kilometers x Miles')
# plt.grid(True)
plt.show()

print(f'Real values: {y_test.values}')
print(f'Predicted values: {y_pred}')

# To do: Generar un algoritmo de regresión lineal que prediga el precio de un carro
# LabelEncoder - Transformar categorias en números
