import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from utilities import JitLSTM


def compute_numerical_derivatives(Y_pred, X_train, time_index=0, device=None):
    """
    Calculate numerical derivatives of variables in Y_pred with respect to the time variable in X_train

    Parameters:
    Y_pred: Model prediction output, shape (batch_size, n_variables)
    X_train: Training input containing time variable, shape (batch_size, n_features)
    time_index: Column index of time variable in X_train, default 0
    device: Device information, uses Y_pred's device if None

    Returns:
    derivatives: Numerical derivatives of each variable, same shape as Y_pred
    """
    if device is None:
        device = Y_pred.device

    # Get time variable and convert to NumPy array
    t = X_train[:, time_index].cpu().detach().numpy()

    # Initialize derivative tensor
    derivatives = torch.zeros_like(Y_pred)

    # Calculate numerical derivative for each variable in Y_pred
    for i in range(Y_pred.shape[1]):
        var_pred = Y_pred[:, i].cpu().detach().numpy()
        var_derivative = np.gradient(var_pred, t)
        derivatives[:, i] = torch.tensor(var_derivative, device=device)

    return derivatives


def compute_specific_derivatives(Y_pred, X_train, variable_indices=[0, 1], time_index=0, device=None):
    """
    Calculate numerical derivatives for specified variables in Y_pred

    Parameters:
    Y_pred: Model prediction output
    X_train: Training input
    variable_indices: List of variable indices to calculate derivatives for
    time_index: Column index of time variable in X_train
    device: Device information

    Returns:
    Tuple of derivatives for specified variables
    """
    all_derivatives = compute_numerical_derivatives(Y_pred, X_train, time_index, device)
    return tuple(all_derivatives[:, i] for i in variable_indices)


def Xt_eqution(x1, dx1_dt, x2, dx2_dt):
    a1, a2, a3 = 0.004, 0.07, 0.04
    b1, b2 = 0.82, 1855
    e1 = a1 + (a2 * x1 ** 2) / (a3 + x1 ** 2) - x1 / (1 + x1 + x2) - dx1_dt
    e2 = b1 / (1 + b2 * x1 ** 5) - x2 / (1 + x1 + x2) - dx2_dt

    return e1, e2


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
infile1 = "./Data/Bs_traindata_n0.3.csv"
infile2 = "./Data/Bs_testdata.csv"

# Read data, ignoring the first row header
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Separate features and target variables, convert to PyTorch tensors
X_train = torch.tensor(df_train.iloc[:, 0].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 1].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 1].values, dtype=torch.float32)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

t_train = X_train
t_test = X_test

# Move tensors to defined device
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

x2_train = df_train.iloc[:, 2]

x2_test = df_test.iloc[:, 2]

# Get number of features and batch size
num_features = X_train.shape[1]

# Define model parameters
input_dim = num_features  # Dimensionality of input features
num_hiddens = 2  # Dimensionality of LSTM hidden layer
num_layers = 1  # Number of LSTM layers
output_dim = 2  # Dimensionality of output features (predicting Y)

# Create LSTM model instance
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first=True)

# Move model to device (CPU or GPU)
model = model.to(device)
learning_rate = 0.1

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
num_epochs = 10000
start_time = time.time()
running_time = 0

loss_values_sum = []
loss_values_1_sum = []
loss_values_2_sum = []
loss_values_3_sum = []
loss_values_4_sum = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total loss

    X_train.requires_grad = True

    X_batch = X_train.unsqueeze(0)

    Y_pred, _ = model(X_batch)

    Y_pred = Y_pred.squeeze()
    X_batch = X_batch.squeeze()

    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]

    # Create all-ones tensor with same shape as Y_pred_1 for grad_outputs
    grad_outputs_1 = torch.ones_like(x1_pred)
    grad_outputs_2 = torch.ones_like(x2_pred)

    # Calculate derivatives of x2_pred with respect to X_batch
    x1dt_pred = torch.autograd.grad(x1_pred, X_train, grad_outputs=grad_outputs_1, create_graph=True)[0]
    x1dt_pred = x1dt_pred[:, 0]

    x2dt_pred = torch.autograd.grad(x2_pred, X_train, grad_outputs=grad_outputs_2, create_graph=True)[0]
    x2dt_pred = x2dt_pred[:, 0]

    [e1_eqns_pred,
     e2_eqns_pred] = Xt_eqution(x1_pred,
                                x1dt_pred,
                                x2_pred,
                                x2dt_pred)

    batch_size = X_train.shape[0]

    e1_eqns_true = np.zeros(batch_size)
    e2_eqns_true = np.zeros(batch_size)

    # Convert NumPy arrays to PyTorch tensors
    e1_eqns_true_tensor = torch.from_numpy(e1_eqns_true).float()
    e2_eqns_true_tensor = torch.from_numpy(e2_eqns_true).float()

    # Calculate loss for entire batch
    # Move tensors to CUDA device if available
    if torch.cuda.is_available():
        e1_eqns_true = e1_eqns_true_tensor.to('cuda')
        e2_eqns_true = e2_eqns_true_tensor.to('cuda')

    # Assume t_batch is a PyTorch tensor
    t_batch = X_train[:, 0]

    # Check if each element in t_batch equals 0.0
    t_is_zero = t_batch == 0.0

    loss1 = torch.mean((x1_pred - y_train) ** 2)
    loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
    loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
    loss4 = torch.mean((x1_pred[t_is_zero] - 2.5) ** 2) + torch.mean(x2_pred[t_is_zero] ** 2)
    loss = loss1 + loss2 + loss3 + loss4

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
            f'Epoch {epoch}, Total Loss: {total_loss:.3e}, REG Loss: {loss1:.3e}, EQ1 Loss: {loss2:.3e}, EQ2 Loss: {loss3:.3e}, Initial Loss: {loss4:.3e}, Time: {elapsed:.3f}, RunningTime: {running_time:.3f}')
        start_time = time.time()

plt.plot(loss_values_sum)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

plt.plot(loss_values_1_sum)
plt.xlabel('Iteration')
plt.ylabel('REG_Loss')
plt.title('REG Loss Curve')
plt.show()

plt.plot(loss_values_2_sum)
plt.xlabel('Iteration')
plt.ylabel('EQ1_Loss')
plt.title('EQ1 Loss Curve')
plt.show()

plt.plot(loss_values_3_sum)
plt.xlabel('Iteration')
plt.ylabel('EQ2_Loss')
plt.title('EQ2 Loss Curve')
plt.show()

plt.plot(loss_values_4_sum)
plt.xlabel('Iteration')
plt.ylabel('EQ2_Loss')
plt.title('EQ3 Loss Curve')
plt.show()

torch.save(model.state_dict(), 'BS_PINN.pth')
model.load_state_dict(torch.load('BS_PINN.pth'))
model.eval()  # Set model to evaluation mode

x1_preds = []  # Store predictions from all batches
x2_preds = []

with torch.no_grad():  # No gradient calculation during testing

    X_batch = X_train.unsqueeze(0)  # Add batch dimension
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]
    x1_preds.append(x1_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
    x2_preds.append(x2_pred.cpu().numpy())

# Concatenate predictions from all batches into a single array
x1_preds = np.concatenate(x1_preds, axis=0)
x2_preds = np.concatenate(x2_preds, axis=0)

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

# Visualize prediction results
plt.scatter(t_train, Y_train, color='red', alpha=0.5, label='True Values')
plt.plot(t_train, Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(t_train, x2_train, color='red', alpha=0.5, label='True Values')
plt.plot(t_train, x2_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

x2_preds = []  # Store predictions from all batches
x1_preds = []

with torch.no_grad():  # No gradient calculation during testing

    X_batch = X_test.unsqueeze(0)  # Add batch dimension
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]
    x2_preds.append(x2_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
    x1_preds.append(x1_pred.cpu().numpy())

# Concatenate predictions from all batches into a single array
x1_preds = np.concatenate(x1_preds, axis=0)
x2_preds = np.concatenate(x2_preds, axis=0)

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

plt.scatter(t_test, x2_test, color='red', alpha=0.5, label='True Values')
plt.plot(t_test, x2_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()