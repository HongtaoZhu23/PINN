import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from utilities import JitLSTM


# Data loading and preprocessing
infile1 = "./Data/YG_Simulation_train_n0.005.csv"
infile2 = "./Data/YG_Simulation_test_n0.005.csv"

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data, ignoring the first row header
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# Sort in ascending order based on the first column
df_train = df_train.sort_values(by=df_train.columns[0], ascending=True)
df_test = df_test.sort_values(by=df_test.columns[0], ascending=True)


X_train = torch.tensor(df_train.iloc[:, 0:6].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 6].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0:6].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 6].values, dtype=torch.float32)


t_train = X_train[:,0]
t_test = X_test[:,0]

# Move tensors to the defined device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


# Get number of features and batch size
num_features = X_train.shape[1]

# Define model parameters
input_dim = num_features  # Dimensionality of input features
num_hiddens = 1  # Dimensionality of LSTM hidden layer
num_layers = 1  # Number of LSTM layers
output_dim = 1  # Dimensionality of output feature (predicting Y)


# Create LSTM model instance
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first=True)

# Move model to device (CPU or GPU)
model = model.to(device)
learning_rate = 0.01

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
num_epochs = 10000
start_time = time.time()
running_time = 0

X_train.requires_grad = True
X_train = X_train.unsqueeze(0)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total loss

    Y_pred, _ = model(X_train)

    Y_pred = Y_pred.squeeze()

    loss = torch.mean((Y_pred - y_train.to(device).squeeze()) ** 2)

    total_loss += loss.item()  # Accumulate loss
    loss.backward()  # Compute gradients

    optimizer.step()  # Update parameters
    optimizer.zero_grad()  # Clear gradients
    model.zero_grad()  # Clear model gradients

    if epoch % 10 == 0:
        # Check if GPU is available, use GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get device name
        device_name = device.type
        elapsed = time.time() - start_time
        running_time += elapsed / 3600.0
        print(torch.backends.cudnn.enabled)
        print(f'Epoch {epoch}, Total Loss: {total_loss:.3e}, Time: {elapsed:.3f}, RunningTime: {running_time:.3f}')
        start_time = time.time()

torch.save(model.state_dict(), 'YG_LSTM_noise.pth')
model.load_state_dict(torch.load('YG_LSTM_noise.pth'))
model.eval()  # Set model to evaluation mode

x1_preds = []  # Store predictions from all batches

with torch.no_grad():  # No gradient calculation during testing

    Y_pred, _ = model(X_train)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x1_preds.append(Y_pred.cpu().numpy())  # Move to CPU and convert to NumPy array

# Concatenate predictions from all batches into a single array
x1_preds = np.concatenate(x1_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_train = y_train.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_train, Y_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(Y_train, Y_preds)
print(f'R2: {r2}')


x1_preds = []  # Store predictions from all batches
with torch.no_grad():  # No gradient calculation during testing

    X_batch = X_test.unsqueeze(0)  # Add batch dimension
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x1_pred = Y_pred
    x1_preds.append(x1_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
# Concatenate predictions from all batches into a single array
x1_preds = np.concatenate(x1_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_test = y_test.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_train
Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

# Visualize prediction results
plt.scatter(t_test, Y_test, color='red', alpha=0.5, label='True Values')
plt.plot(t_test, Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()
