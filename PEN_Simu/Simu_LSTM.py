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

def Penicillin_BB(qp, qp_t, mu, qs, Cs):

    # para
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
infile1 = "./Data/PEN_Simu_train_SNR50.csv"
infile2 = "./Data/PEN_Simu_test_SNR50.csv"


# 读取数据，忽略第一行表头
df_train = pd.read_csv(infile1, header=0)
df_test = pd.read_csv(infile2, header=0)


# 分离特征和目标变量
X_train = df_train.iloc[:, 0:-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

# 计算最大值和最小值
X_min = torch.tensor(X_train.min(axis=0).values, dtype=torch.float32)  # 每列最小值
X_max = torch.tensor(X_train.max(axis=0).values, dtype=torch.float32)  # 每列最大值
y_min = torch.tensor(y_train.min(), dtype=torch.float32)  # 目标最小值
y_max = torch.tensor(y_train.max(), dtype=torch.float32)  # 目标最大值

# 最大最小归一化
X_train_normalized = (torch.tensor(X_train.values, dtype=torch.float32) - X_min) / (X_max - X_min)
y_train_normalized = (torch.tensor(y_train.values, dtype=torch.float32) - y_min) / (y_max - y_min)
X_test_normalized = (torch.tensor(X_test.values, dtype=torch.float32) - X_min) / (X_max - X_min)
y_test_normalized = (torch.tensor(y_test.values, dtype=torch.float32) - y_min) / (y_max - y_min)


# 将特征和目标变量合并
df_train_scaled = pd.DataFrame(X_train_normalized, columns=X_train.columns)
df_train_scaled['y'] = y_train_normalized.flatten()

df_test_scaled = pd.DataFrame(X_test_normalized, columns=X_test.columns)
df_test_scaled['y'] = y_test_normalized.flatten()

# 划分训练和测试数据的批次
train_batches_X, train_batches_y = split_batches(df_train)
test_batches_X, test_batches_y = split_batches(df_test)

# 转换为 PyTorch 张量
train_batches_X = [torch.tensor(batch, dtype=torch.float32) for batch in train_batches_X]
train_batches_y = [torch.tensor(batch, dtype=torch.float32) for batch in train_batches_y]
test_batches_X = [torch.tensor(batch, dtype=torch.float32) for batch in test_batches_X]
test_batches_y = [torch.tensor(batch, dtype=torch.float32) for batch in test_batches_y]

# 如果需要将所有批次整合成一个 numpy 数组
num_batches_train = len(train_batches_X)
num_batches_test = len(test_batches_X)

# 获取特征数量和批次数量
num_features = df_train.shape[1] - 1


# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将训练数据批次移动到 CUDA 设备
train_batches_X = [tensor.to(device) for tensor in train_batches_X]
train_batches_y = [tensor.to(device) for tensor in train_batches_y]

# 将测试数据批次移动到 CUDA 设备
test_batches_X = [tensor.to(device) for tensor in test_batches_X]

X_min = X_min.to(device)
X_max = X_max.to(device)

# 定义模型参数
input_dim = num_features  # 输入特征的维度
num_hiddens = 1  # LSTM 隐藏层的维度
num_layers = 1  # LSTM 的层数
output_dim = 1  # 输出特征的维度（预测Y）



# 创建 LSTM 模型实例
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first= True)

# # 创建 LSTM 模型实例
# model = LSTMModel(input_dim, num_hiddens, num_layers, batch_first= True)

# 将模型移动到设备（CPU 或 GPU）
model = model.to(device)
learning_rate = 0.001

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
num_epochs = 10000
start_time = time.time()
running_time = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # 初始化总损失
    for i in range(num_batches_train):
        # with torch.backends.cudnn.flags(enabled=False):
        X_batch = train_batches_X[i]
        Y_batch = train_batches_y[i]
        X_batch = X_batch.unsqueeze(0)
        Y_batch = Y_batch.unsqueeze(0)
        X_batch.requires_grad = True
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()

        loss = torch.mean((Y_pred - Y_batch.to(device).squeeze()) ** 2)


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

torch.save(model.state_dict(),'LSTM_Simu_SNR50.pth')
model.load_state_dict(torch.load('LSTM_Simu_SNR50.pth'))
model.eval()  # 将模型设置为评估模式

Y_preds = []  # 用于存储所有批次的预测结果
mu_preds = []
qs_preds = []
with torch.no_grad():  # 在测试阶段不计算梯度
    for i in range(num_batches_test):
        X_batch = torch.tensor(test_batches_X[i], dtype=torch.float32).to(device)  # 移动到设备
        X_batch = X_batch.unsqueeze(0)  # 添加批次维度
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()  # 移除批次维度
        qp_pred = Y_pred # 选择预测的第一列
        Y_preds.append(qp_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组

# 将所有批次的预测结果拼接成一个数组
Y_preds = np.concatenate(Y_preds, axis=0)


# 确保Y_test也是一维数组
Y_test = torch.cat(test_batches_y, dim=0)
Y_test = Y_test.reshape(-1, 1)

# 计算RMSE
# 确保 Y_test 和 Y_preds 都在 CPU 上，并且是 NumPy 数组
Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
Y_preds = np.array(Y_preds) if not isinstance(Y_preds, np.ndarray) else Y_preds

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_preds))
print(f'RMSE: {rmse}')

# 计算R2
r2 = r2_score(Y_test, Y_preds)
print(f'R2: {r2}')

# # 可视化预测结果
# plt.scatter(range(len(Y_test)), Y_test, color='red', alpha=0.5, label='True Values')
# plt.plot(range(len(Y_preds)), Y_preds, color='blue', label='Predictions')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Predictions vs True Values')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.scatter(range(len(mu_data)), mu_data, color='red', alpha=0.5, label='True Values')
# plt.plot(range(len(mu_preds)), mu_preds, color='blue', label='Predictions')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Predictions vs True Values')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.scatter(range(len(qs_data)), qs_data, color='red', alpha=0.5, label='True Values')
# plt.plot(range(len(qs_preds)), qs_preds, color='blue', label='Predictions')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Predictions vs True Values')
# plt.legend()
# plt.grid(True)
# plt.show()