import pandas as pd
import numpy as np
from flaml import AutoML
import pickle
import os
from datetime import datetime

df = pd.read_csv('dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)
df['rolling_mean_7'] = df['value'].rolling(window=7).mean().shift(1)
df = df.dropna().reset_index(drop=True)

X = df[['year', 'month', 'day', 'day_of_year', 'lag_1', 'lag_7', 'rolling_mean_7']]
y = df['value']

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

automl = AutoML()
automl_settings = {
    "time_budget": 120,
    "metric": 'rmse',
    "task": 'regression',
    "verbose": 1
}

automl.fit(X_train, y_train, **automl_settings)

os.makedirs('model', exist_ok=True)
with open('model/modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(automl, f)

os.makedirs('requirement', exist_ok=True)
requirements = [
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    'flaml>=2.1.0',
    'scikit-learn>=1.3.0'
]

with open('requirement/requirements.txt', 'w') as f:
    f.write('\n'.join(requirements))

training_df = pd.concat([X_train, y_train], axis=1)
os.makedirs('csv', exist_ok=True)
training_df.to_csv('csv/dataset.csv', index=False)

print(f"Modelo entrenado y guardado")
print(f"RMSE en test: {automl.score(X_test, y_test):.4f}")
print(f"Archivos generados:")
print(f"  - model/modelo_entrenado.pkl")
print(f"  - requirement/requirements.txt")
print(f"  - csv/dataset.csv")

