import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
data = pd.read_csv('CarPrice_Assignment.csv')

# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Seleccionar las columnas categóricas
categorical_columns = data.select_dtypes(include=['object']).columns

# Aplicar LabelEncoder a cada columna categórica
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Seleccionar solo las columnas numéricas
X = data.select_dtypes(include=['number']).drop(['price', 'car_ID'], axis=1)
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
sns.scatterplot(x=X_test['CarName'], y=y_test)
sns.lineplot(x=X_test['CarName'], y=y_pred, color='red')

plt.xlabel('Car Name')
plt.ylabel('Price')

plt.title('Ca rName vs Price')

plt.show()

print(f'Real values: {y_test.values}')
print(f'Predicted values: {y_pred}')