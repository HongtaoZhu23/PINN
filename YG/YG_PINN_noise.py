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


def Xt_eqution(x1, dx1_dt, x2, dx2_dt, x3, dx3_dt, x4, dx4_dt, x5, dx5_dt, x6, dx6_dt, x7, dx7_dt):
    c1, c2, c3 = 2.5, -100, 13.6769
    d1, d2, d3, d4 = 200, 13.6769, -6.0, -6.0
    e1, e2, e3, e4 = 6.0, -64.0, 6.0, 16.0
    f1, f2, f3, f4, f5 = 64.0, -13.0, 3.0, -16.0, -100.0
    g1, g2 = 1.3, -3.1
    h1, h2, h3, h4, h5 = -200.0, 13.6769, 128.0, -1.28, -32.0
    j1, j2, j3 = 6.0, -18.0, -100.0

    eq1 = c1 + (c2 * x1 * x6) / (1 + c3 * x6 ** 4) - dx1_dt
    eq2 = (d1 * x1 * x6) / (1 + d2 * x6 ** 4) + d3 * x2 - d4 * x2 * x7 - dx2_dt
    eq3 = e1 * x2 + e2 * x3 + e3 * x2 * x7 + e4 * x3 * x6 - dx3_dt
    eq4 = f1 * x3 + f2 * x4 + f3 * x5 + f4 * x3 * x6 + f5 * x4 * x7 - dx4_dt
    eq5 = g1 * x4 + g2 * x5 - dx5_dt
    eq6 = (h1 * x1 * x6) / (1 + h2 * x6 ** 4) + h3 * x3 + h5 * x6 + h4 * x3 * x7 - dx6_dt
    eq7 = j1 * x2 + j2 * x2 * x7 + j3 * x4 * x7 - dx7_dt

    return eq4, eq5


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
infile1 = "./Data/YG_Simulation_train_n0.01.csv"
infile2 = "./Data/YG_Simulation_test_n0.01.csv"

# Read data, ignoring the first row header
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Separate features and target variables, convert to PyTorch tensors
X_train = torch.tensor(df_train.iloc[:, 0:6].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 6].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0:6].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 6].values, dtype=torch.float32)

t_train = X_train[:, 0]
x1 = X_train[:, 1]
x2 = X_train[:, 2]
x3 = X_train[:, 3]
x6 = X_train[:, 4]
x7 = X_train[:, 5]

t_test = X_test[:, 0]

# Move tensors to defined device
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

x4_train = df_train.iloc[:, 7]

x4_test = df_test.iloc[:, 7]

x1dt = np.gradient(x1, t_train)
x2dt = np.gradient(x2, t_train)
x3dt = np.gradient(x3, t_train)
x6dt = np.gradient(x6, t_train)
x7dt = np.gradient(x7, t_train)

# dx5/dt
x1dt = torch.tensor(x1dt, dtype=torch.float32)
x2dt = torch.tensor(x2dt, dtype=torch.float32)
x3dt = torch.tensor(x3dt, dtype=torch.float32)
x6dt = torch.tensor(x6dt, dtype=torch.float32)
x7dt = torch.tensor(x7dt, dtype=torch.float32)

x1dt = x1dt.to(device)
x2dt = x2dt.to(device)
x3dt = x3dt.to(device)
x6dt = x6dt.to(device)
x7dt = x7dt.to(device)

# Get number of features and batch size
num_features = X_train.shape[1]

# Define model parameters
input_dim = num_features  # Dimensionality of input features
num_hiddens = 3  # Dimensionality of LSTM hidden layer
num_layers = 1  # Number of LSTM layers
output_dim = 2  # Dimensionality of output features (predicting Y)

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

loss_values_sum = []
loss_values_1_sum = []
loss_values_2_sum = []
loss_values_3_sum = []
loss_values_4_sum = []
loss_values_5_sum = []
loss_values_6_sum = []
loss_values_7_sum = []
loss_values_8_sum = []
loss_values_9_sum = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total loss

    X_train.requires_grad = True

    X_batch = X_train.unsqueeze(0)

    Y_pred, _ = model(X_batch)

    Y_pred = Y_pred.squeeze()
    X_batch = X_batch.squeeze()

    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]

    x5dt_pred, x4dt_pred = compute_specific_derivatives(Y_pred, X_train, [0, 1])

    t = X_batch[:, 0]
    x1 = X_batch[:, 1]
    x2 = X_batch[:, 2]
    x3 = X_batch[:, 3]
    x6 = X_batch[:, 4]
    x7 = X_batch[:, 5]

    [e1_eqns_pred,
     e2_eqns_pred] = Xt_eqution(x1, x1dt,
                                x2, x2dt,
                                x3, x3dt,
                                x4_pred, x4dt_pred,
                                x5_pred, x5dt_pred,
                                x6, x6dt,
                                x7, x7dt)

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

    # Check if each element in t_batch equals 0.0
    t_is_zero = t == 0.0

    loss1 = torch.mean((x5_pred - y_train) ** 2)
    loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
    loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
    loss4 = torch.mean((x5_pred[t_is_zero] - 0.077) ** 2) + torch.mean((x4_pred[t_is_zero] - 0.115) ** 2)
    loss = 500 * loss1 + loss2 + loss3 + loss4

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

torch.save(model.state_dict(), 'YG_PINN_noise.pth')
model.load_state_dict(torch.load('YG_PINN_noise.pth'))
model.eval()  # Set model to evaluation mode

x5_preds = []  # Store predictions from all batches
x4_preds = []

with torch.no_grad():  # No gradient calculation during testing

    X_batch = X_train.unsqueeze(0)  # Add batch dimension
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]
    x5_preds.append(x5_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
    x4_preds.append(x4_pred.cpu().numpy())

# Concatenate predictions from all batches into a single array
x5_preds = np.concatenate(x5_preds, axis=0)
x4_preds = np.concatenate(x4_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_train = y_train.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(x5_preds) if not isinstance(x5_preds, np.ndarray) else x5_preds

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

plt.scatter(t_train, x4_train, color='red', alpha=0.5, label='True Values')
plt.plot(t_train, x4_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

x5_preds = []  # Store predictions from all batches
x4_preds = []

with torch.no_grad():  # No gradient calculation during testing

    X_batch = X_test.unsqueeze(0)  # Add batch dimension
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # Remove batch dimension
    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]
    x5_preds.append(x5_pred.cpu().numpy())  # Move to CPU and convert to NumPy array
    x4_preds.append(x4_pred.cpu().numpy())

# Concatenate predictions from all batches into a single array
x5_preds = np.concatenate(x5_preds, axis=0)
x4_preds = np.concatenate(x4_preds, axis=0)

# Ensure Y_test is also a 1D array
Y_test = y_test.reshape(-1, 1)

# Calculate RMSE
# Ensure both Y_test and Y_preds are on CPU and are NumPy arrays
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
Y_preds = np.array(x5_preds) if not isinstance(x5_preds, np.ndarray) else x5_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

x4_test = x4_test.cpu().numpy() if isinstance(x4_test, torch.Tensor) else x4_test
x4_preds = np.array(x4_preds) if not isinstance(x4_preds, np.ndarray) else x4_preds

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(x4_test, x4_preds))
print(f'RMSE: {rmse}')

# Calculate R2
r2 = r2_score(x4_test, x4_preds)
print(f'R2: {r2}')

# Calculate Mean Relative Error (MRE) for x5 and x4
mre_x5 = np.mean(np.abs((Y_test.squeeze() - x5_preds) / Y_test.squeeze()))
mre_x4 = np.mean(np.abs((x4_test.squeeze() - x4_preds) / x4_test.squeeze()))

print(f'Mean Relative Error (x5): {mre_x5:.3%}')
print(f'Mean Relative Error (x4): {mre_x4:.3%}')

# Visualize prediction results
plt.scatter(t_test, Y_test, color='red', alpha=0.5, label='True Values')
plt.plot(t_test, Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(t_test, x4_test, color='red', alpha=0.5, label='True Values')
plt.plot(t_test, x4_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()