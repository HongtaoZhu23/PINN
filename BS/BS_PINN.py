import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from utilities import JitLSTM
import torch.nn.functional as F

def compute_numerical_derivatives(Y_pred, X_train, time_index=0, device=None):
    """
    计算Y_pred中各变量关于X_train中时间变量的数值导数

    参数:
    Y_pred: 模型预测输出，形状为(batch_size, n_variables)
    X_train: 训练输入，包含时间变量，形状为(batch_size, n_features)
    time_index: 时间变量在X_train中的列索引，默认为0
    device: 设备信息，如果为None则使用Y_pred的设备

    返回:
    derivatives: 各变量的数值导数，形状与Y_pred相同
    """
    if device is None:
        device = Y_pred.device

    # 获取时间变量并转换为NumPy数组
    t = X_train[:, time_index].cpu().detach().numpy()

    # 初始化导数张量
    derivatives = torch.zeros_like(Y_pred)

    # 对Y_pred中的每个变量计算数值导数
    for i in range(Y_pred.shape[1]):
        var_pred = Y_pred[:, i].cpu().detach().numpy()
        var_derivative = np.gradient(var_pred, t)
        derivatives[:, i] = torch.tensor(var_derivative, device=device)

    return derivatives

def compute_specific_derivatives(Y_pred, X_train, variable_indices=[0, 1], time_index=0, device=None):
    """
    计算Y_pred中指定变量的数值导数

    参数:
    Y_pred: 模型预测输出
    X_train: 训练输入
    variable_indices: 需要计算导数的变量索引列表
    time_index: 时间变量在X_train中的列索引
    device: 设备信息

    返回:
    各指定变量的导数元组
    """
    all_derivatives = compute_numerical_derivatives(Y_pred, X_train, time_index, device)
    return tuple(all_derivatives[:, i] for i in variable_indices)

def Xt_eqution(x1,dx1_dt,x2,dx2_dt):

    a1, a2, a3 = 0.004, 0.07, 0.04
    b1, b2 = 0.82, 1855
    e1 = a1 + (a2 * x1 ** 2) / (a3 + x1 ** 2) - x1 / (1 + x1 + x2) - dx1_dt
    e2 = b1 / (1 + b2 * x1**5) - x2 / (1 + x1 + x2) - dx2_dt

    return e1, e2



def split_batches(data):
    """
    将数据划分为若干批次，每个批次的时间步单调递增。
    参数：
    - data: pandas DataFrame，每行数据的第一列为时间，其余列为特征。

    返回：
    - batches_X: list，每个元素为 numpy 数组，形状为 (time_steps, num_features)。
    - batches_y: list，每个元素为 numpy 数组，形状为 (time_steps,)
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


# 数据加载和预处理
infile1 = "./Data/Bs_traindata_n0.3.csv"
infile2 = "./Data/Bs_testdata.csv"

# 读取数据，忽略第一行表头
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# 添加噪声
# noise_std = 0.00  # 噪声的标准差
# df_train.iloc[:, 1] = df_train.iloc[:, 1] + noise_std * np.random.randn(*df_train.iloc[:, 1].shape)
# df_test.iloc[:, 1] = df_test.iloc[:, 1] + noise_std * np.random.randn(*df_test.iloc[:, 1].shape)

# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分离特征和目标变量，并转换为 PyTorch 张量
X_train = torch.tensor(df_train.iloc[:, 0].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 3].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 1].values, dtype=torch.float32)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

t_train = X_train
t_test = X_test

# 将张量移动到定义的设备上
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

x2_train = df_train.iloc[:, 2]

x2_test = df_test.iloc[:, 2]

# 获取特征数量和批次数量
num_features = X_train.shape[1]

# 定义模型参数
input_dim = num_features  # 输入特征的维度
num_hiddens = 2  # LSTM 隐藏层的维度
num_layers = 1  # LSTM 的层数
output_dim = 2  # 输出特征的维度（预测Y）

# 创建 LSTM 模型实例
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first=True)

# # 创建 LSTM 模型实例
# model = LSTMModel(input_dim, num_hiddens, num_layers, batch_first= True)

# 将模型移动到设备（CPU 或 GPU）
model = model.to(device)
learning_rate = 0.1

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
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
    total_loss = 0  # 初始化总损失

    X_train.requires_grad = True

    X_batch = X_train.unsqueeze(0)

    Y_pred, _ = model(X_batch)

    Y_pred = Y_pred.squeeze()
    X_batch = X_batch.squeeze()

    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]

    # # 创建与Y_pred_1形状相同的全1张量作为grad_outputs
    # grad_outputs_1 = torch.ones_like(x1_pred)
    # grad_outputs_2 = torch.ones_like(x2_pred)
    #
    # # 计算x2_pred对X_batch的导数
    # x1dt_pred = torch.autograd.grad(x1_pred, X_train, grad_outputs=grad_outputs_1, create_graph=True)[0]
    # x1dt_pred = x1dt_pred[:, 0]
    #
    # x2dt_pred = torch.autograd.grad(x2_pred, X_train, grad_outputs=grad_outputs_2, create_graph=True)[0]
    # x2dt_pred = x2dt_pred[:, 0]

    # x1dt_pred, x2dt_pred = compute_specific_derivatives(Y_pred, X_train, [0, 1])
    # 创建与Y_pred_1形状相同的全1张量作为grad_outputs
    grad_outputs_1 = torch.ones_like(x1_pred)
    grad_outputs_2 = torch.ones_like(x2_pred)

    # 计算x2_pred对X_batch的导数
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

    # 将 NumPy 数组转换为 PyTorch 张量
    e1_eqns_true_tensor = torch.from_numpy(e1_eqns_true).float()
    e2_eqns_true_tensor = torch.from_numpy(e2_eqns_true).float()

    # 计算整个批次的损失
    # 将张量移动到 CUDA 设备上
    if torch.cuda.is_available():
        e1_eqns_true = e1_eqns_true_tensor.to('cuda')
        e2_eqns_true = e2_eqns_true_tensor.to('cuda')

    # 假设 t_batch 是一个 PyTorch 张量
    t_batch = X_train[:,0]

    # 比较 t_batch 中的每个元素是否等于 0.0
    t_is_zero = t_batch == 0.0

    loss1 = torch.mean((x1_pred - y_train) ** 2)
    loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
    loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
    loss4 = torch.mean((x1_pred[t_is_zero] - 2.5) ** 2) + torch.mean(x2_pred[t_is_zero] ** 2)
    loss = loss1 + loss2 + loss3 + loss4

    total_loss += loss.item()  # 累积损失
    loss.backward()  # 计算梯度

    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度
    model.zero_grad()  # 清空模型梯度

    loss_values_sum.append(loss.detach().cpu().numpy())
    loss_values_1_sum.append(loss1.detach().cpu().numpy())
    loss_values_2_sum.append(loss2.detach().cpu().numpy())
    loss_values_3_sum.append(loss3.detach().cpu().numpy())
    loss_values_4_sum.append(loss4.detach().cpu().numpy())

    if epoch % 10 == 0:
        # 检查是否有可用的GPU，如果有，则使用GPU，否则使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取设备名称
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
model.eval()  # 将模型设置为评估模式

x1_preds = []  # 用于存储所有批次的预测结果
x2_preds = []

with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_train.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]
    x1_preds.append(x1_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组
    x2_preds.append(x2_pred.cpu().numpy())

# 将所有批次的预测结果拼接成一个数组
x1_preds = np.concatenate(x1_preds, axis=0)
x2_preds = np.concatenate(x2_preds, axis=0)



# 确保Y_test也是一维数组
Y_train = y_train.reshape(-1, 1)

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(Y_train, Y_preds))
print(f'RMSE: {rmse}')

# 计算R2
r2 = r2_score(Y_train, Y_preds)
print(f'R2: {r2}')

# 可视化预测结果
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

x2_preds = []  # 用于存储所有批次的预测结果
x1_preds = []

with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_test.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x1_pred = Y_pred[:, 0]
    x2_pred = Y_pred[:, 1]
    x2_preds.append(x2_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组
    x1_preds.append(x1_pred.cpu().numpy())

# 将所有批次的预测结果拼接成一个数组
x1_preds = np.concatenate(x1_preds, axis=0)
x2_preds = np.concatenate(x2_preds, axis=0)

# 确保Y_test也是一维数组
Y_test = y_test.reshape(-1, 1)

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_train
Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# 计算R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

# 可视化预测结果
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