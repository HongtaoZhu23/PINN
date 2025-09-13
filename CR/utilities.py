import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple, Optional
from torch import Tensor
import math

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