import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos aleatorios simulando tarjetas de transporte público
np.random.seed(0)
n_samples = 1000
hora_inicio = np.random.randint(0, 24, n_samples)  # Hora de inicio del viaje (0-23)
dia_semana = np.random.randint(0, 7, n_samples)  # Día de la semana (0-6)
distancia_km = np.random.rand(n_samples) * 20  # Distancia del viaje en kilómetros
tarifa = np.random.randint(1, 6, n_samples) * 10  # Tarifa pagada en pesos

# Crear DataFrame
df = pd.DataFrame({
    'Hora Inicio': hora_inicio,
    'Dia Semana': dia_semana,
    'Distancia (km)': distancia_km,
    'Tarifa (pesos)': tarifa
})
print(df)

# Separar los datos en conjuntos de entrenamiento y prueba
X = df[['Hora Inicio', 'Dia Semana', 'Distancia (km)']]
y = df['Tarifa (pesos)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualizar las predicciones junto con los datos de prueba
predicciones_df = pd.DataFrame({
    'Hora Inicio': X_test['Hora Inicio'],
    'Dia Semana': X_test['Dia Semana'],
    'Distancia (km)': X_test['Distancia (km)'],
    'Tarifa Real': y_test,
    'Tarifa Predicha': y_pred
})

print(predicciones_df)
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')
