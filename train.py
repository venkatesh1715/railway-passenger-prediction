import torch
from torch.utils.data import TensorDataset, DataLoader
from model.lstm_model import LSTMModel
from utils.data_preparation import load_and_preprocess_data

# Load data
X, y, scaler, df = load_and_preprocess_data("data/railway_passengers.csv", seq_length=5)
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = LSTMModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
EPOCHS = 100
for epoch in range(EPOCHS):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model trained and saved!")