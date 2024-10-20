import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Cargar los datos
data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

# Eliminar las columnas innecesarias    
data_test = data_test.drop(['Name', 'Ticket', 'Embarked'], axis=1)
data_train = data_train.drop(['Name', 'Ticket', 'Embarked'], axis=1)

# Tratamiento de la columna 'Cabin' usando la mediana
data_train['Cabin'] = data_train['Cabin'].fillna('U')
data_test['Cabin'] = data_test['Cabin'].fillna('U')
    
# Tratamiento de la columna 'Fare' usando la mediana
data_train['Fare'] = data_train['Fare'].fillna(data_train['Fare'].median())
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].median())

# Tratamiento de la columna 'Age' usando la mediana
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())

# Transformar la columna de sexo con LabelEncoder
label_encoder = LabelEncoder()
data_test['Sex'] = label_encoder.fit_transform(data_test['Sex'])
data_train['Sex'] = label_encoder.fit_transform(data_train['Sex'])
data_train['Cabin'] = label_encoder.fit_transform(data_train['Cabin'])
data_test['Cabin'] = label_encoder.fit_transform(data_test['Cabin'])

# Dividir los datos en características (X) y variable objetivo (y)
X_train = data_train.drop(['Survived'], axis=1)
y_train = data_train['Survived']

X_test = data_test

# Definir el modelo
# model = LogisticRegression(max_iter=10000)
# model = DecisionTreeClassifier(random_state=42)
model =  RandomForestClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Agregar las predicciones a data_test
data_test['Survived'] = y_pred

# Seleccionar las columnas 'Sex' y 'Survived'
output = data_test[['PassengerId', 'Survived']]

# Guardar el DataFrame resultante en un archivo CSV
output.to_csv('gender_submission.csv', index=False)

# Mostrar las primeras filas del DataFrame resultante
print(output.head())

# Evaluar el modelo
# print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
# print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
# print(f'R2 Score: {r2_score(y_test, y_pred)}')
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
# print(f'Precision: {precision_score(y_test, y_pred)}')
# print(f'Recall: {recall_score(y_test, y_pred)}')
# print(f'F1 Score: {f1_score(y_test, y_pred)}')
# 
# # Matriz de confusión
# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, cmap='Blues', annot=True, xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
# plt.show()
# 
# # Reporte de clasificación
# print(classification_report(y_test, y_pred))
# 
# # Curva ROC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.grid(True)
# plt.show()