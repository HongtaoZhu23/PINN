import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from utilities import JitLSTM, compute_higher_order_derivatives


def Penicillin_BB(qp, qp_t, mu, qs, Cs):
    # Parameters
    para = {
        "qs_max": -4.48e-2,  # -4.48×10^-2
        "Ks": 7.8e-6,  # 7.8×10^-6
        "alpha": -0.251,  # α
        "beta": -5.747,  # β
        "mS": -1.5e-3,  # mS = -1.5×10^-3
        "a": 0.0,
        "b": 8e-4,
        "Kp": 8.39e-6,
        "kdE": 1.47e-2,
        "m": 2.0
    }

    e1 = para["qs_max"] * (Cs / (Cs + para["Ks"])) - qs
    e2 = para["alpha"] * mu + para["beta"] * qp + para["mS"] - qs
    e3 = qp_t - (((para["a"] + para["b"] * mu) / (1 + (Cs / para["Kp"]) ** para["m"])) + (para["kdE"] + mu) * qp)

    return e1, e2, e3


def split_batches(data):
    """
    Split data into batches where time steps are monotonically increasing in each batch
    Parameters:
    - data: pandas DataFrame, first column is time, remaining columns are features

    Returns:
    - batches_X: list, each element is a numpy array with shape (time_steps, num_features)
    - batches_y: list, each element is a numpy array with shape (time_steps,)
    """
    batches_X = []
    batches_y = []
    current_batch_X = []
    current_batch_y = []

    for i in range(len(data)):
        if i == 0 or data.iloc[i, 0] > data.iloc[i - 1, 0]:
            current_batch_X.append(data.iloc[i, :-1].to_numpy())
            current_batch_y.append(data.iloc[i, -1])
        else:
            if current_batch_X:
                batches_X.append(np.array(current_batch_X))
                batches_y.append(np.array(current_batch_y))
                current_batch_X = [data.iloc[i, :-1].to_numpy()]
                current_batch_y = [data.iloc[i, -1]]

    if current_batch_X:
        batches_X.append(np.array(current_batch_X))
        batches_y.append(np.array(current_batch_y))

    return batches_X, batches_y


# Data loading and preprocessing
infile1 = "./Data/PEN_Simu_train_n0.1.csv"
infile2 = "./Data/PEN_Simu_test_n0.1.csv"

# Read data, ignoring the first row header
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# Separate features and target variables
X_train = df_train.iloc[:, 0:-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

# Calculate minimum and maximum values
X_min = torch.tensor(X_train.min(axis=0).values, dtype=torch.float32)  # Minimum value for each column
X_max = torch.tensor(X_train.max(axis=0).values, dtype=torch.float32)  # Maximum value for each column
y_min = torch.tensor(y_train.min(), dtype=torch.float32)  # Target minimum value
y_max = torch.tensor(y_train.max(), dtype=torch.float32)  # Target maximum value

# Min-max normalization
X_train_normalized = (torch.tensor(X_train.values, dtype=torch.float32) - X_min) / (X_max - X_min)
y_train_normalized = (torch.tensor(y_train.values, dtype=torch.float32) - y_min) / (y_max - y_min)
X_test_normalized = (torch.tensor(X_test.values, dtype=torch.float32) - X_min) / (X_max - X_min)
y_test_normalized = (torch.tensor(y_test.values, dtype=torch.float32) - y_min) / (y_max - y_min)

# Combine features and target variables
df_train_scaled = pd.DataFrame(X_train_normalized, columns=X_train.columns)
df_train_scaled['y'] = y_train_normalized.flatten()

df_test_scaled = pd.DataFrame(X_test_normalized, columns=X_test.columns)
df_test_scaled['y'] = y_test_normalized.flatten()

# Split training and test data into batches
train_batches_X, train_batches_y = split_batches(df_train_scaled)
test_batches_X, test_batches_y = split_batches(df_test_scaled)

# Convert to PyTorch tensors
train_batches_X = [torch.tensor(batch, dtype=torch.float32) for batch in train_batches_X]
train_batches_y = [torch.tensor(batch, dtype=torch.float32) for batch in train_batches_y]
test_batches_X = [torch.tensor(batch, dtype=torch.float32) for batch in test_batches_X]
test_batches_y = [torch.tensor(batch, dtype=torch.float32) for batch in test_batches_y]

# If needed to combine all batches into a single numpy array
num_batches_train = len(train_batches_X)
num_batches_test = len(test_batches_X)

# Get number of features and batch size
num_features = df_train.shape[1] - 1

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move training data batches to CUDA device
train_batches_X = [tensor.to(device) for tensor in train_batches_X]
train_batches_y = [tensor.to(device) for tensor in train_batches_y]

# Move test data batches to CUDA device
test_batches_X = [tensor.to(device) for tensor in test_batches_X]

X_min = X_min.to(device)
X_max = X_max.to(device)

# Define model parameters
input_dim = num_features  # Dimensionality of input features
num_hiddens = 4  # Dimensionality of LSTM hidden layer
num_layers = 4  # Number of LSTM layers
output_dim = 4  # Dimensionality of output features (predicting Y)

# Create LSTM model instance
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first=True)

# Move model to device (CPU or GPU)
model = model.to(device)
learning_rate = 0.01

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_values_sum = []
loss_values_1_sum = []
loss_values_2_sum = []
loss_values_3_sum = []
loss_values_4_sum = []
loss_values_5_sum = []

# Train model
num_epochs = 10000
start_time = time.time()
running_time = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total loss
    for i in range(num_batches_train):
        # with torch.backends.cudnn.flags(enabled=False):
        X_batch = train_batches_X[i]
        Y_batch = train_batches_y[i]
        X_batch = X_batch.unsqueeze(0)
        Y_batch = Y_batch.unsqueeze(0)
        X_batch.requires_grad = True
        Y_pred, _ = model(X_batch)

        Y_pred = Y_pred.squeeze()
        Y_batch = Y_batch.squeeze()
        X = X_batch.squeeze()

        Y_pred_1 = Y_pred[:, 0]  # Get prediction for first variable

        # Calculate first and second order derivatives
        qp_pred, qpdt_pred = compute_higher_order_derivatives(Y_pred_1, X, order=2)

        [e1_eqns_pred,
         e2_eqns_pred,
         e3_eqns_pred] = Penicillin_BB(qp_pred,
                                       qpdt_pred,
                                       Y_pred[:, 1],
                                       Y_pred[:, 2],
                                       Y_pred[:, 3])

        batch_size = X_batch.shape[1]
        e1_eqns_true = np.zeros(batch_size)
        e2_eqns_true = np.zeros(batch_size)
        e3_eqns_true = np.zeros(batch_size)

        # Convert NumPy arrays to PyTorch tensors
        e1_eqns_true_tensor = torch.from_numpy(e1_eqns_true).float()
        e2_eqns_true_tensor = torch.from_numpy(e2_eqns_true).float()
        e3_eqns_true_tensor = torch.from_numpy(e3_eqns_true).float()

        # Calculate loss for entire batch
        # Move tensors to CUDA device if available
        if torch.cuda.is_available():
            e1_eqns_true = e1_eqns_true_tensor.to('cuda')
            e2_eqns_true = e2_eqns_true_tensor.to('cuda')
            e3_eqns_true = e3_eqns_true_tensor.to('cuda')

        t = X[:, 0].cpu().detach().numpy()
        t = t.flatten()
        t_is_zero = t == 0.0

        loss1 = torch.mean((Y_pred[:, 0] - Y_batch.to(device).squeeze()) ** 2)
        loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
        loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
        loss4 = torch.mean((e3_eqns_true - e3_eqns_pred) ** 2)

        loss = 50000 * loss1 + loss2 + loss3 + loss4
        total_loss += loss.item()  # Accumulate loss
        loss.backward()  # Compute gradients

    optimizer.step()  # Update parameters
    optimizer.zero_grad()  # Clear gradients
    model.zero_grad()  # Clear model gradients

    loss_values_sum.append(loss.detach().cpu().numpy())
    loss_values_1_sum.append(loss1.detach().cpu().numpy())
    loss_values_2_sum.append(loss2.detach().cpu().numpy())
    loss_values_3_sum.append(loss3.detach().cpu().numpy())
    loss_values_4_sum.append(loss4.detach().cpu().numpy())

    if epoch % 10 == 0:
        # Check if GPU is available, use GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get device name
        device_name = device.type
        elapsed = time.time() - start_time
        running_time += elapsed / 3600.0
        print(torch.backends.cudnn.enabled)
        print(
            f'Epoch {epoch}, Total Loss: {total_loss:.3e}, REG Loss: {loss1:.3e}, EQ1 Loss: {loss2:.3e}, EQ2 Loss: {loss3:.3e}, EQ3 Loss: {loss4:.3e}, Time: {elapsed:.3f}, RunningTime: {running_time:.3f}')
        start_time = time.time()

torch.save(model.state_dict(), 'Simu_PINN.pth')
model.load_state_dict(torch.load('Simu_PINN.pth'))
model.eval()  # Set model to evaluation mode

Y_preds = []  # Store predictions from all batches
mu_preds = []
qs_preds = []
with torch.no_grad():  # No gradient calculation during testing
    for i in range(num_batches_train):
        X_batch = torch.tensor(train_batches_X[i], dtype=torch.float32).to(device)  # Move to device
        X_batch = X_batch.unsqueeze(0)  # Add batch dimension
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()  # Remove batch dimension
        qp_pred = Y_pred[:, 0]  # Select first column of predictions
        mu_pred = Y_pred[:, 1]
        qs_pred = Y_pred[:, 2]
        Y_preds.append(qp_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
        mu_preds.append(mu_pred.cpu().numpy())
        qs_preds.append(qs_pred.cpu().numpy())
# Concatenate predictions from all batches into a single array
Y_preds = np.concatenate(Y_preds, axis=0)
mu_preds = np.concatenate(mu_preds, axis=0)
qs_preds = np.concatenate(qs_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_train = torch.cat(train_batches_y, dim=0)
Y_train = Y_train.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(Y_preds) if not isinstance(Y_preds, np.ndarray) else Y_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_train, Y_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(Y_train, Y_preds)
print(f'R2: {r2}')

# Visualize prediction results
plt.scatter(range(len(Y_train)), Y_train, color='red', alpha=0.5, label='True Values')
plt.plot(range(len(Y_preds)), Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

Y_preds = []  # Store predictions from all batches
mu_preds = []
qs_preds = []
with torch.no_grad():  # No gradient calculation during testing
    for i in range(num_batches_test):
        X_batch = torch.tensor(test_batches_X[i], dtype=torch.float32).to(device)  # Move to device
        X_batch = X_batch.unsqueeze(0)  # Add batch dimension
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()  # Remove batch dimension
        qp_pred = Y_pred[:, 0]  # Select first column of predictions
        mu_pred = Y_pred[:, 1]
        qs_pred = Y_pred[:, 2]
        Y_preds.append(qp_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
        mu_preds.append(mu_pred.cpu().numpy())
        qs_preds.append(qs_pred.cpu().numpy())
# Concatenate predictions from all batches into a single array
Y_preds = np.concatenate(Y_preds, axis=0)
mu_preds = np.concatenate(mu_preds, axis=0)
qs_preds = np.concatenate(qs_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_test = torch.cat(test_batches_y, dim=0)
Y_test = Y_test.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
Y_preds = np.array(Y_preds) if not isinstance(Y_preds, np.ndarray) else Y_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

# Visualize prediction results
plt.scatter(range(len(Y_test)), Y_test, color='red', alpha=0.5, label='True Values')
plt.plot(range(len(Y_preds)), Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()