import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(730)]
trend = np.linspace(100, 200, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonal + noise

df = pd.DataFrame({
    'date': dates,
    'value': values
})

df.to_csv('dataset.csv', index=False)
print(f"Dataset generado: {len(df)} registros")
print(f"Período: {df['date'].min()} a {df['date'].max()}")

