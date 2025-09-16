import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple, Optional
from torch import Tensor
import math
import numpy as np

def compute_higher_order_derivatives(Y_pred_var, X_data, time_index=0, order=2, device=None):
    """
    计算高阶数值导数

    参数:
    Y_pred_var: 单个变量的预测值，形状为(batch_size,)
    X_data: 包含时间变量的输入数据，形状为(batch_size, n_features)
    time_index: 时间变量在X_data中的列索引，默认为0
    order: 导数的阶数，默认为2（计算一阶和二阶导数）
    device: 设备信息，如果为None则使用Y_pred_var的设备

    返回:
    包含各阶导数的元组，顺序为从一阶到指定阶数
    """
    if device is None:
        device = Y_pred_var.device

    # 获取时间变量并转换为NumPy数组
    t = X_data[:, time_index].cpu().detach().numpy().flatten()

    # 将预测值转换为NumPy数组
    var_np = Y_pred_var.cpu().detach().numpy().flatten()

    # 计算各阶导数
    derivatives = []
    current_derivative = var_np

    for i in range(order):
        current_derivative = np.gradient(current_derivative, t)
        derivatives.append(torch.tensor(current_derivative, device=device))

    return tuple(derivatives)


# 或者更通用的版本，可以处理多个变量
def compute_multi_variable_higher_order_derivatives(Y_pred, X_data, variable_indices=None,
                                                    time_index=0, order=2, device=None):
    """
    计算多个变量的高阶数值导数

    参数:
    Y_pred: 模型预测输出，形状为(batch_size, n_variables)
    X_data: 包含时间变量的输入数据，形状为(batch_size, n_features)
    variable_indices: 需要计算导数的变量索引列表，如果为None则计算所有变量
    time_index: 时间变量在X_data中的列索引，默认为0
    order: 导数的阶数，默认为2（计算一阶和二阶导数）
    device: 设备信息，如果为None则使用Y_pred的设备

    返回:
    字典，键为变量索引，值为包含各阶导数的元组
    """
    if device is None:
        device = Y_pred.device

    if variable_indices is None:
        variable_indices = range(Y_pred.shape[1])

    # 获取时间变量并转换为NumPy数组
    t = X_data[:, time_index].cpu().detach().numpy().flatten()

    results = {}

    for var_idx in variable_indices:
        # 获取当前变量的预测值
        var_pred = Y_pred[:, var_idx]
        var_np = var_pred.cpu().detach().numpy().flatten()

        # 计算各阶导数
        derivatives = []
        current_derivative = var_np

        for i in range(order):
            current_derivative = np.gradient(current_derivative, t)
            derivatives.append(torch.tensor(current_derivative, device=device))

        results[var_idx] = tuple(derivatives)

    return results

class JitLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

        i, f, g, o = gates.chunk(4, 1)

        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        cell_gate = torch.tanh(g)
        output_gate = torch.sigmoid(o)

        cy = (forget_gate * cx) + (input_gate * cell_gate)
        hy = output_gate * torch.tanh(cy)

        return hy, cy

class JitLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        hx, cx = hidden

        for i in range(len(inputs)):
            hx, cx = self.cell(inputs[i], (hx, cx))
            outputs += [hx]

        return torch.stack(outputs), (hx, cx)

class JitLSTM(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=False, bias=True):
        super(JitLSTM, self).__init__()
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitLSTMLayer(JitLSTMCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitLSTMLayer(JitLSTMCell, input_size, hidden_size)] +
                                        [JitLSTMLayer(JitLSTMCell, hidden_size, hidden_size) for _ in range(num_layers - 1)])

        self.fc = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()  # 确保输出大于0

    @jit.script_method
    def forward(self, x: Tensor, h: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = (torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device),
                 torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device))

        output = x
        hn, cn = [], []

        for i, layer in enumerate(self.layers):
            output, (hn_i, cn_i) = layer(output, (h[0][i], h[1][i]))
            hn.append(hn_i)
            cn.append(cn_i)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        output = output.view(-1, self.hidden_size)
        output = self.fc(output)
        output = self.softplus(output)  # 确保输出大于0

        output = output.view(-1, x.size(1), self.fc.out_features)

        return output, (torch.stack(hn), torch.stack(cn))