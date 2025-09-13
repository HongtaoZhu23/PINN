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


def Xt_eqution(x1, dx1_dt, x2, dx2_dt):
    k1 = 1.0497
    k2 = 0.9766
    e1 = dx1_dt + k1 * x1
    e2 = dx2_dt - k1 * x1 + k2 * x2

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
infile1 = "./Data/CR_train.csv"
infile2 = "./Data/CR_test.csv"

# 读取数据，忽略第一行表头
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分离特征和目标变量，并转换为 PyTorch 张量
X_train = torch.tensor(df_train.iloc[:, 0:2].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 2].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0:2].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 2].values, dtype=torch.float32)

t_train = X_train[:, 0]
t_test = X_test[:, 0]

dx3_dt = torch.tensor(df_train.iloc[:, 2].values, dtype=torch.float32)
dx3_dt = dx3_dt.to(device)

# 将张量移动到定义的设备上
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

x2_train = torch.tensor(df_train.iloc[:, -1].values, dtype=torch.float32)
x2_train = x2_train.to(device)

x2_test = df_test.iloc[:, -1]

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
num_epochs = 10000
start_time = time.time()
running_time = 0

loss_values_sum = []
loss_values_1_sum = []
loss_values_2_sum = []
loss_values_3_sum = []
loss_values_4_sum = []
loss_values_5_sum = []

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

    # 假设 t 是 X_train 中的第一个变量
    t = X_train[:, 0].cpu().detach().numpy()  # 转换为 NumPy 数组

    # 将预测值转换为 NumPy 数组
    x1_pred_np = x1_pred.cpu().detach().numpy()
    x2_pred_np = x2_pred.cpu().detach().numpy()

    # 使用 np.gradient 计算导数
    x1dt_pred = np.gradient(x1_pred_np, t)
    x2dt_pred = np.gradient(x2_pred_np, t)

    x1dt_pred = torch.tensor(x1dt_pred, device=device)
    x2dt_pred = torch.tensor(x2dt_pred, device=device)

    # 创建与Y_pred_1形状相同的全1张量作为grad_outputs
    grad_outputs_1 = torch.ones_like(x1_pred)
    grad_outputs_2 = torch.ones_like(x2_pred)


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
    t_batch = X_train[:, 0]  # 您的张量

    # 比较 t_batch 中的每个元素是否等于 0.0
    t_is_zero = t_batch == 0.0

    loss1 = torch.mean((x2_pred - x2_train) ** 2)
    loss2 = torch.mean((e1_eqns_true - e1_eqns_pred) ** 2)
    loss3 = torch.mean((e2_eqns_true - e2_eqns_pred) ** 2)
    loss5 = torch.mean((x1_pred[t_is_zero] - 1) ** 2) + \
            torch.mean(x2_pred[t_is_zero] ** 2)
    loss = 500 * loss1 + loss2 + loss3 + loss5

    total_loss += loss.item()  # 累积损失
    loss.backward()  # 计算梯度

    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度
    model.zero_grad()  # 清空模型梯度

    loss_values_sum.append(loss.detach().cpu().numpy())
    loss_values_1_sum.append(loss1.detach().cpu().numpy())
    loss_values_2_sum.append(loss2.detach().cpu().numpy())
    loss_values_3_sum.append(loss3.detach().cpu().numpy())

    loss_values_5_sum.append(loss5.detach().cpu().numpy())

    if epoch % 10 == 0:
        # 检查是否有可用的GPU，如果有，则使用GPU，否则使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取设备名称
        device_name = device.type
        elapsed = time.time() - start_time
        running_time += elapsed / 3600.0
        print(torch.backends.cudnn.enabled)
        print(
            f'Epoch {epoch}, Total Loss: {total_loss:.3e}, REG Loss: {loss1:.3e}, EQ1 Loss: {loss2:.3e}, EQ2 Loss: {loss3:.3e}, Time: {elapsed:.3f}, RunningTime: {running_time:.3f}')
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

# plt.plot(loss_values_4_sum)
# plt.xlabel('Iteration')
# plt.ylabel('EQ2_Loss')
# plt.title('EQ3 Loss Curve')
# plt.show()

plt.plot(loss_values_5_sum)
plt.xlabel('Iteration')
plt.ylabel('Initial_Loss')
plt.title('Initial Condition Loss Curve')
plt.show()

torch.save(model.state_dict(), 'CR_PINN.pth')
model.load_state_dict(torch.load('CR_PINN.pth'))
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

# # 确保Y_test也是一维数组
# Y_train = y_train.reshape(-1, 1)
#
# # 计算RMSE
# # 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
# Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
# Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# 确保Y_test也是一维数组
Y_train = x2_train.reshape(-1, 1)

# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
Y_preds = np.array(x2_preds) if not isinstance(x2_preds, np.ndarray) else x2_preds

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

# x2_train = x2_train.cpu().numpy() if isinstance(x2_train, torch.Tensor) else x2_train
#
# plt.scatter(t_train, x2_train, color='red', alpha=0.5, label='True Values')
# plt.plot(t_train, x2_preds, color='blue', label='Predictions')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Predictions vs True Values')
# plt.legend()
# plt.grid(True)
# plt.show()

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
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
Y_preds = np.array(x1_preds) if not isinstance(x1_preds, np.ndarray) else x1_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'x1_RMSE: {rmse}')

# 计算R2
r2 = r2_score(Y_test, Y_preds)
print(f'x1_R2: {r2}')

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
x2_test = x2_test.cpu().numpy() if isinstance(x2_test, torch.Tensor) else x2_test
x2_preds = np.array(x2_preds) if not isinstance(x2_preds, np.ndarray) else x2_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(x2_test, x2_preds))
print(f'x2_RMSE: {rmse}')

# 计算R2
r2 = r2_score(x2_test, x2_preds)
print(f'x2_R2: {r2}')

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

# 拉平成 1-D
y1_true = Y_test.ravel()
y2_true = x2_test.ravel()
y1_pred = x1_preds.ravel()
y2_pred = x2_preds.ravel()

# 构造掩码，排除 0 值
mask1 = y1_true != 0
mask2 = y2_true != 0

# 计算 MRE，如果 mask 为空则返回 NaN
mre_x1 = np.mean(np.abs((y1_true[mask1] - y1_pred[mask1]) / y1_true[mask1])) if mask1.any() else np.nan
mre_x2 = np.mean(np.abs((y2_true[mask2] - y2_pred[mask2]) / y2_true[mask2])) if mask2.any() else np.nan

print(f'x1_MRE: {mre_x1:.3%}')
print(f'x2_MRE: {mre_x2:.3%}')