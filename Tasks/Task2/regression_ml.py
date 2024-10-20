import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar los datos
data = pd.read_csv('CarPrice_Assignment.csv')

# Seleccionar solo las columnas numéricas
X = data.select_dtypes(include=['number']).drop(['price'], axis=1)
y = data['price']

# Normalizar los datos
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# Graficar los resultados
sns.scatterplot(x=X_test['car_ID'], y=y_test)
sns.lineplot(x=X_test['car_ID'], y=y_pred, color='red')

plt.xlabel('Car ID')
plt.ylabel('Price')

plt.title('Car id 0 vs Price')

plt.show()

print(f'Real values: {y_test.values}')
print(f'Predicted values: {y_pred}')