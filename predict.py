import pandas as pd
import torch
import matplotlib.pyplot as plt
from model.lstm_model import LSTMModel
from utils.data_preparation import load_and_preprocess_data
import numpy as np

# Load and preprocess the data
X, y, scaler, df = load_and_preprocess_data("data/railway_passengers.csv", seq_length=5)
X_tensor = torch.Tensor(X)

# Load trained model
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Predict for known data (in-sample)
with torch.no_grad():
    predicted_in_sample = model(X_tensor).numpy()

# Inverse transform predictions and actuals
predicted_in_sample_real = scaler.inverse_transform(predicted_in_sample)
actual_real = scaler.inverse_transform(y)

# ===== 📅 Future prediction for Jan–Dec 2023 =====
last_seq = X[-1]  # last 5 months of 2022
future_preds = []

with torch.no_grad():
    for _ in range(12):  # Predict next 12 months
        input_seq = torch.Tensor(last_seq).unsqueeze(0)  # shape: (1, 5, 1)
        next_pred = model(input_seq).numpy()
        future_preds.append(next_pred[0][0])
        
        # Update sequence: drop first, append new
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1] = next_pred

# Inverse scale future predictions
future_preds_scaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# ===== 🖼️ Plotting =====
# Create x-axis labels
months_actual = df.index[5:]  # since seq_length=5
months_future = pd.date_range(start='2023-01', periods=12, freq='M')

plt.figure(figsize=(12,5))
plt.plot(months_actual, actual_real, label="Actual (2020–2022)")
plt.plot(months_actual, predicted_in_sample_real, label="Predicted (2020–2022)")
plt.plot(months_future, future_preds_scaled, label="Forecasted (2023)", linestyle='--', marker='o')

plt.title("Railway Passenger Prediction: 2020–2023")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()