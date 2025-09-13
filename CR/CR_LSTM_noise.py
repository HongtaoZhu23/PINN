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


# 数据加载和预处理
infile1 = "./Data/CR_train_noise.csv"
infile2 = "./Data/CR_test_noise.csv"

# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据，忽略第一行表头
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)

# 按第一列升序排序
df_train = df_train.sort_values(by=df_train.columns[0], ascending=True)
df_test = df_test.sort_values(by=df_test.columns[0], ascending=True)

# 分离特征和目标变量，并转换为 PyTorch 张量
X_train = torch.tensor(df_train.iloc[:, 0:2].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, 4].values, dtype=torch.float32)

X_test = torch.tensor(df_test.iloc[:, 0:2].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, 3].values, dtype=torch.float32)

t_train = X_train[:,0]
t_test = X_test[:,0]
# 将张量移动到定义的设备上
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)


# 获取特征数量和批次数量
num_features = 2

# 定义模型参数
input_dim = num_features  # 输入特征的维度
num_hiddens = 1 # LSTM 隐藏层的维度
num_layers = 1  # LSTM 的层数
output_dim = 1  # 输出特征的维度（预测Y）


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

X_train.requires_grad = True


for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # 初始化总损失

    X_train = X_train.unsqueeze(0)

    Y_pred, _ = model(X_train)

    Y_pred = Y_pred.squeeze()
    X_train = X_train.squeeze()

    loss = torch.mean((Y_pred - y_train.to(device).squeeze()) ** 2)

    total_loss += loss.item()  # 累积损失
    loss.backward()  # 计算梯度

    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度
    model.zero_grad()  # 清空模型梯度

    if epoch % 10 == 0:
        # 检查是否有可用的GPU，如果有，则使用GPU，否则使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 获取设备名称
        device_name = device.type
        elapsed = time.time() - start_time
        running_time += elapsed / 3600.0
        print(torch.backends.cudnn.enabled)
        print(f'Epoch {epoch}, Total Loss: {total_loss:.3e}, Time: {elapsed:.3f}, RunningTime: {running_time:.3f}')
        start_time = time.time()

torch.save(model.state_dict(), 'CR_LSTM_noise.pth')
model.load_state_dict(torch.load('CR_LSTM_noise.pth'))
model.eval()  # 将模型设置为评估模式

x1_preds = []  # 用于存储所有批次的预测结果

with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_train.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x1_preds.append(Y_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组

# 将所有批次的预测结果拼接成一个数组
x1_preds = np.concatenate(x1_preds, axis=0)

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


x1_preds = []  # 用于存储所有批次的预测结果
with torch.no_grad():  # 在测试阶段不计算梯度

    X_batch = X_test.unsqueeze(0)  # 添加批次维度
    Y_pred, _ = model(X_batch)
    Y_pred = Y_pred.squeeze()  # 移除批次维度
    x1_pred = Y_pred
    x1_preds.append(x1_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组
# 将所有批次的预测结果拼接成一个数组
x1_preds = np.concatenate(x1_preds, axis=0)

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
