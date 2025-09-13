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



def Xt_eqution(x1,dx1_dt,x2,dx2_dt,x3,dx3_dt,x4,dx4_dt,x5,dx5_dt,x6,dx6_dt,x7,dx7_dt):

    c1, c2, c3 = 2.5, -100, 13.6769
    d1, d2, d3, d4 = 200, 13.6769, -6.0, -6.0
    e1, e2, e3, e4 = 6.0, -64.0, 6.0, 16.0
    f1, f2, f3, f4, f5 = 64.0, -13.0, 3.0, -16.0, -100.0
    g1, g2 = 1.3, -3.1
    h1, h2, h3, h4, h5 = -200.0, 13.6769, 128.0, -1.28, -32.0
    j1, j2, j3 = 6.0, -18.0, -100.0


    eq1 = c1 + (c2 * x1 * x6) / (1 + c3 * x6**4) - dx1_dt
    eq2 = (d1 * x1 * x6) / (1 + d2 * x6**4) + d3 * x2 - d4 * x2 * x7 - dx2_dt
    eq3 = e1 * x2 + e2 * x3 + e3 * x2 * x7 + e4 * x3 * x6 - dx3_dt
    eq4 = f1 * x3 + f2 * x4 + f3 * x5 + f4 * x3 * x6 + f5 * x4 * x7 - dx4_dt
    eq5 = g1 * x4 + g2 * x5 - dx5_dt
    eq6 = (h1 * x1 * x6) / (1 + h2 * x6**4) + h3 * x3 + h5 * x6 + h4 * x3 * x7 - dx6_dt
    eq7 = j1 * x2 + j2 * x2 * x7 + j3 * x4 * x7 - dx7_dt

    return  eq4, eq5



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
infile1 = "./Data/YG_Simulation_train.csv"
infile2 = "./Data/YG_Simulation_test_n0.01.csv"

# 读取数据，忽略第一行表头
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分离特征和目标变量，并转换为 PyTorch 张量
X_train = torch.tensor(df_train.iloc[:, 0:6].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 6].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0:6].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 6].values, dtype=torch.float32)

t_train = X_train[:,0]
x1 = X_train[:, 1]
x2 = X_train[:, 2]
x3 = X_train[:, 3]
x6 = X_train[:, 4]
x7 = X_train[:, 5]

t_test = X_test[:,0]

# 将张量移动到定义的设备上
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
learning_rate = 0.01

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
num_epochs = 20000
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
    total_loss = 0  # 初始化总损失

    X_train.requires_grad = True

    X_batch = X_train.unsqueeze(0)

    Y_pred, _ = model(X_batch)

    Y_pred = Y_pred.squeeze()
    X_batch = X_batch.squeeze()

    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]

    # # 假设 t 是 X_train 中的第一个变量
    # t = X_train[:, 0].cpu().detach().numpy()  # 转换为 NumPy 数组
    #
    # # 将预测值转换为 NumPy 数组
    # x5_pred_np = x5_pred.cpu().detach().numpy()
    # x4_pred_np = x4_pred.cpu().detach().numpy()
    #
    # # 使用 np.gradient 计算导数
    # x5dt_pred = np.gradient(x5_pred_np, t)
    # x4dt_pred = np.gradient(x4_pred_np, t)
    #
    # x5dt_pred = torch.tensor(x5dt_pred, device=device)
    # x4dt_pred = torch.tensor(x4dt_pred, device=device)

    x5dt_pred, x4dt_pred = compute_specific_derivatives(Y_pred, X_train, [0, 1])

    t = X_batch[:, 0]
    x1 = X_batch[:, 1]
    x2 = X_batch[:, 2]
    x3 = X_batch[:, 3]
    x6 = X_batch[:, 4]
    x7 = X_batch[:, 5]

    # t_col = X_train[:, 0:1].clone()  # shape [B, 1]
    # t_col.requires_grad_(True)
    #
    # # 2. 组装输入
    # X_in = X_train.clone()
    # X_in[:, 0:1] = t_col  # 替换时间列为带梯度的张量
    #
    # # 3. 前向
    # Y_pred, _ = model(X_in.unsqueeze(0))  # [1, B, 2]
    # Y_pred = Y_pred.squeeze()  # [B, 2]
    # x5_pred = Y_pred[:, 0]  # [B]
    # x4_pred = Y_pred[:, 1]  # [B]
    #
    # # 4. 自动求导：dx/dt
    # ones = torch.ones_like(x5_pred)
    # x5dt_pred = torch.autograd.grad(
    #     outputs=x5_pred, inputs=t_col,
    #     grad_outputs=ones, create_graph=True, retain_graph=True)[0].squeeze(-1)
    #
    # x4dt_pred = torch.autograd.grad(
    #     outputs=x4_pred, inputs=t_col,
    #     grad_outputs=ones, create_graph=True, retain_graph=True)[0].squeeze(-1)
    #
    # # 5. 其余变量
    # t = X_in[:, 0]  # 已带梯度
    # x1 = X_in[:, 1]
    # x2 = X_in[:, 2]
    # x3 = X_in[:, 3]
    # x6 = X_in[:, 4]
    # x7 = X_in[:, 5]

    [e1_eqns_pred,
     e2_eqns_pred] = Xt_eqution(x1,x1dt,
                                x2,x2dt,
                                x3,x3dt,
                                x4_pred,x4dt_pred,
                                x5_pred,x5dt_pred,
                                x6,x6dt,
                                x7,x7dt)

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

    # 比较 t_batch 中的每个元素是否等于 0.0
    t_is_zero = t == 0.0

    loss1 = torch.mean((x5_pred - y_train) ** 2)
    loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
    loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
    loss4 = torch.mean((x5_pred[t_is_zero] - 0.077) ** 2) + torch.mean((x4_pred[t_is_zero] - 0.115) ** 2)
    loss = 50000 * loss1 + loss2 + 1000 * loss3 + 1000 * loss4

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

# plt.plot(loss_values_sum)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.show()
#
# plt.plot(loss_values_6_sum)
# plt.xlabel('Iteration')
# plt.ylabel('EQ5_Loss')
# plt.title('EQ5 Loss Curve')
# plt.show()
#
# plt.plot(loss_values_8_sum)
# plt.xlabel('Iteration')
# plt.ylabel('EQ7_Loss')
# plt.title('EQ7 Loss Curve')
# plt.show()
#
# plt.plot(loss_values_9_sum)
# plt.xlabel('Iteration')
# plt.ylabel('EQ8_Loss')
# plt.title('Initial Condition Loss Curve')
# plt.show()

torch.save(model.state_dict(), 'PI-YG.pth')
model.load_state_dict(torch.load('PI-YG.pth'))
model.eval()  # 将模型设置为评估模式

x5_preds = []  # 用于存储所有批次的预测结果
x4_preds = []

with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_train.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]
    x5_preds.append(x5_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组
    x4_preds.append(x4_pred.cpu().numpy())

# 将所有批次的预测结果拼接成一个数组
x5_preds = np.concatenate(x5_preds, axis=0)
x4_preds = np.concatenate(x4_preds, axis=0)


# 确保Y_test也是一维数组
Y_train = y_train.reshape(-1, 1)

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(x5_preds) if not isinstance(x5_preds, np.ndarray) else x5_preds

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

plt.scatter(t_train, x4_train, color='red', alpha=0.5, label='True Values')
plt.plot(t_train, x4_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()

x5_preds = []  # 用于存储所有批次的预测结果
x4_preds = []

with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_test.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x5_pred = Y_pred[:, 0]
    x4_pred = Y_pred[:, 1]
    x5_preds.append(x5_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组
    x4_preds.append(x4_pred.cpu().numpy())

# 将所有批次的预测结果拼接成一个数组
x5_preds = np.concatenate(x5_preds, axis=0)
x4_preds = np.concatenate(x4_preds, axis=0)

# 确保Y_test也是一维数组
Y_test = y_test.reshape(-1, 1)

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
Y_preds = np.array(x5_preds) if not isinstance(x5_preds, np.ndarray) else x5_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# 计算R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

x4_test = x4_test.cpu().numpy() if isinstance(x4_test, torch.Tensor) else x4_test
x4_preds = np.array(x4_preds) if not isinstance(x4_preds, np.ndarray) else x4_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(x4_test, x4_preds))
print(f'RMSE: {rmse}')

# 计算R2
r2 = r2_score(x4_test, x4_preds)
print(f'R2: {r2}')

# 计算 x5 与 x4 的平均相对误差（Mean Relative Error, MRE）
mre_x5 = np.mean(np.abs((Y_test.squeeze() - x5_preds) / Y_test.squeeze()))
mre_x4 = np.mean(np.abs((x4_test.squeeze() - x4_preds) / x4_test.squeeze()))

print(f'Mean Relative Error (x5): {mre_x5:.3%}')
print(f'Mean Relative Error (x4): {mre_x4:.3%}')

# # 可视化预测结果
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
