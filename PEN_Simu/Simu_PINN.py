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

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if hidden is not None:
            h0, c0 = hidden
        else:
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out, (hn, cn)

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

# 数据加载和预处理
infile1 = "./Data/PEN_Simulation_3Batch_train.csv"
infile2 = "./Data/PEN_Simulation_test.csv"

# 读取数据
df_train = pd.read_csv(infile1)
df_test = pd.read_csv(infile2)

# 获取特征数量和批次数量
num_features = df_train.shape[1] - 1
num_batches_train = df_train.shape[0] // 150 # 每个批次400个时间步
num_batches_test = df_test.shape[0] // 150  # 每个批次400个时间步

# 训练数据
X_train_np = (pd.read_csv(infile1)).to_numpy()[:, :-1]
X_train = torch.from_numpy(X_train_np).reshape(num_batches_train, 150, num_features).float()  # 转换为float类型
Y_train_np = pd.read_csv(infile1).to_numpy()[:, -1]
Y_train = torch.from_numpy(Y_train_np).reshape(num_batches_train, 150, 1).float()  # 转换为float类型

# 测试数据
X_test_np = (pd.read_csv(infile2)).to_numpy()[:, :-1]
X_test = torch.from_numpy(X_test_np).reshape(num_batches_test, 150, num_features).float()  # 转换为float类型
Y_test_np = pd.read_csv(infile2).to_numpy()[:, -1]
Y_test = torch.from_numpy(Y_test_np).reshape(num_batches_test, 150, 1).float()  # 转换为float类型

# 将数据移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

# 定义模型参数
input_dim = num_features  # 输入特征的维度
num_hiddens = 1  # LSTM 隐藏层的维度
num_layers = 2  # LSTM 的层数
output_dim = 4  # 输出特征的维度（预测Y）
batch_size = 150


# 创建 LSTM 模型实例
model = JitLSTM(input_dim, num_hiddens, num_layers, output_dim, batch_first= True)

# # 创建 LSTM 模型实例
# model = LSTMModel(input_dim, num_hiddens, num_layers, batch_first= True)

# 将模型移动到设备（CPU 或 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
learning_rate = 0.005

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

e1_eqns_true = np.zeros(batch_size)
e2_eqns_true = np.zeros(batch_size)
e3_eqns_true = np.zeros(batch_size)

# 将 NumPy 数组转换为 PyTorch 张量
e1_eqns_true_tensor = torch.from_numpy(e1_eqns_true).float()
e2_eqns_true_tensor = torch.from_numpy(e2_eqns_true).float()
e3_eqns_true_tensor = torch.from_numpy(e3_eqns_true).float()

# 训练模型
num_epochs = 10000
start_time = time.time()
running_time = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # 初始化总损失
    for i in range(num_batches_train):
        # with torch.backends.cudnn.flags(enabled=False):
        X_batch = X_train[i]
        Y_batch = Y_train[i]
        X_batch = X_batch.unsqueeze(0)
        Y_batch = Y_batch.unsqueeze(0)
        X_batch.requires_grad = True
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()

        Y_pred_1 = Y_pred[:, 0]

        # 创建与Y_pred_1形状相同的全1张量作为grad_outputs
        grad_outputs = torch.ones_like(Y_pred_1)

        # 创建与Y_pred_1形状相同的全1张量作为grad_outputs
        grad_outputs = torch.ones_like(Y_pred_1)



        # 计算Y_pred_1对X_batch的导数
        qp_pred = torch.autograd.grad(Y_pred_1, X_batch, grad_outputs=grad_outputs, create_graph=True)[0]

        qp_pred = qp_pred.squeeze(0)
        qp_pred_1st_feature = qp_pred[:, 0]
        qp_pred = qp_pred[:, 0]

        # 计算qp_pred对X_batch第一特征的导数
        qp_pred_1st_feature_grad = torch.autograd.grad(qp_pred_1st_feature.sum(), X_batch, create_graph=True)[0]
        qp_pred_1st_feature_grad = qp_pred_1st_feature_grad.squeeze(0)
        qp_t_pred = qp_pred_1st_feature_grad[:, 0]



        [e1_eqns_pred,
         e2_eqns_pred,
         e3_eqns_pred] = Penicillin_BB(qp_pred,
                                       qp_t_pred,
                                       Y_pred[:, 1],
                                       Y_pred[:, 2],
                                       Y_pred[:, 3])


        # 计算整个批次的损失
        # 将张量移动到 CUDA 设备上
        if torch.cuda.is_available():
            e1_eqns_true = e1_eqns_true_tensor.to('cuda')
            e2_eqns_true = e2_eqns_true_tensor.to('cuda')
            e3_eqns_true = e3_eqns_true_tensor.to('cuda')

        loss = loss_fn(Y_pred_1, Y_batch.to(device).squeeze())+ \
               torch.mean((e1_eqns_true - e1_eqns_pred) ** 2) + \
               torch.mean((e2_eqns_true - e2_eqns_pred) ** 2) + \
               torch.mean((e3_eqns_true - e3_eqns_pred) ** 2)

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

torch.save(model.state_dict(),'PI-LSTM_SNR75.pth')
model.load_state_dict(torch.load('PI-LSTM_SNR75.pth'))
model.eval()  # 将模型设置为评估模式

Y_preds = []  # 用于存储所有批次的预测结果
with torch.no_grad():  # 在测试阶段不计算梯度
    for i in range(num_batches_test):
        X_batch = torch.tensor(X_test[i], dtype=torch.float32).to(device)  # 移动到设备
        X_batch = X_batch.unsqueeze(0)  # 添加批次维度
        Y_pred, _ = model(X_batch)
        Y_pred = Y_pred.squeeze()  # 移除批次维度
        Y_pred = Y_pred[:, 0]  # 选择预测的第一列
        Y_preds.append(Y_pred.cpu().numpy())  # 移动到CPU并转换为NumPy数组

# 将所有批次的预测结果拼接成一个数组
Y_preds = np.concatenate(Y_preds, axis=0)

# 确保Y_test也是一维数组
Y_test = Y_test.squeeze()
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

# 可视化预测结果
plt.scatter(range(len(Y_test)), Y_test, color='red', alpha=0.5, label='True Values')
plt.plot(range(len(Y_preds)), Y_preds, color='blue', label='Predictions')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()