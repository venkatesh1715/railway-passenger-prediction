import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath, seq_length=5):
    df = pd.read_csv(filepath, parse_dates=['Month'], index_col='Month')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i + seq_length])
        y.append(scaled[i + seq_length])

    return np.array(X), np.array(y), scaler, df
