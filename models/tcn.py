"""
TCN 模型定义 (修正因果填充)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

# --- 辅助函数：复现Keras的 'TruncatedNormal' 初始化 ---
def truncated_normal_(tensor, mean=0, std=0.01):
    """
    用截断正态分布填充张量。
    复现 Keras/TensorFlow 的初始化方式。
    """
    a, b = -2 * std, 2 * std # 截断范围

    # 从 (0, 1) 的均匀分布开始
    uniform_tensor = torch.rand_like(tensor)

    # 转换为标准正态分布 (逆CDF)
    norm_tensor = torch.erfinv(2 * uniform_tensor - 1) * math.sqrt(2)

    # 截断
    torch.clamp(norm_tensor, a, b, out=norm_tensor)

    # 缩放并平移
    norm_tensor.mul_(std).add_(mean)

    with torch.no_grad():
        tensor.copy_(norm_tensor)
    return tensor

class CausalConv1dLayer(nn.Module):
    """
    一个包含 因果卷积 + BN + 激活 的TCN层。
    复现 Keras (padding='causal')
    """
    # --- 修改：移除了未使用的 l2_reg 参数 ---
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn, initializer_stddev):
        super(CausalConv1dLayer, self).__init__()

        # 计算因果填充量 (只在左侧)
        self.padding_left = (kernel_size - 1)

        # 卷积层 (padding=0, 因为我们手动填充)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0, # 我们将使用 F.pad() 手动进行因果填充
            dilation=1 # 原项目未使用空洞卷积
        )

        # 批量归一化 (Batch Norm)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation_fn

        # --- 初始化 ---
        # 复现 Keras Conv1D 初始化
        truncated_normal_(self.conv.weight, std=initializer_stddev)
        # 第一层使用0.1偏置 (Keras默认为0)
        # PyTorch Conv1d 默认初始化偏置，我们重新设置它
        if in_channels == 1278: # 假设一个标识（或者传入is_first_layer=True）
            nn.init.constant_(self.conv.bias, 0.1)
        else:
            nn.init.constant_(self.conv.bias, 0.0) # Keras Conv1D 默认

    def forward(self, x):
        """
        x: (B, C, T)
        """
        # 手动在左侧（时间维度）填充 (0, 0) 表示不在C维度填充
        x_padded = F.pad(x, (self.padding_left, 0))
        x_conv = self.conv(x_padded)
        x_bn = self.bn(x_conv)
        x_act = self.activation(x_bn)
        return x_act

class TCN(nn.Module):
    """
    复现 selfishgene/neuron_as_deep_net TCN 模型。

    这是一个没有残差连接、空洞因子为1的简单 Conv1d 栈。
    Keras (B, T, C) -> PyTorch (B, C, T)
    """
    def __init__(self, input_channels, num_dvt_channels, cfg):
        """
        Args:
            input_channels (int): 输入通道数 (例如 1278 个突触)
            num_dvt_channels (int): 树突输出通道数 (PCA成分数, 可以为0)
            cfg (module): 配置文件 (config.py)
        """
        super(TCN, self).__init__()

        self.input_channels = input_channels
        self.num_dvt_channels = num_dvt_channels
        self.cfg = cfg

        layers = []
        in_channels = input_channels

        filter_sizes = cfg.MODEL_CONFIG["FILTER_SIZES"]
        num_filters = cfg.MODEL_CONFIG["NUM_FILTERS"]
        depth = cfg.MODEL_CONFIG["NETWORK_DEPTH"]

        if len(filter_sizes) != depth:
            raise ValueError("FILTER_SIZES 列表的长度必须等于 NETWORK_DEPTH")

        for i in range(depth):
            kernel_size = filter_sizes[i]
            out_channels = num_filters

            if cfg.MODEL_CONFIG["ACTIVATION_FUNCTION"] == 'relu':
                activation_fn = nn.ReLU()
            else:
                activation_fn = nn.Identity() # 线性

            layers.append(
                CausalConv1dLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation_fn,
                    initializer_stddev=cfg.MODEL_CONFIG["INITIALIZER_STDDEV"],
                    l2_reg=cfg.MODEL_CONFIG["L2_REGULARIZATION"] # L2正则化稍后添加到优化器
                )
            )
            in_channels = out_channels

        # 核心卷积块
        self.network = nn.Sequential(*layers)

        # --- 3. 输出头 (Output Heads) ---
        # 复现原项目的三个独立1x1卷积头

        # 3a. 脉冲预测头 (Spike Head)
        self.head_spike = nn.Conv1d(in_channels, 1, kernel_size=1)

        # 3b. Soma 电压头 (Soma Head)
        self.head_soma = nn.Conv1d(in_channels, 1, kernel_size=1)

        # 3c. 树突电压头 (Dendritic Head) - 可选
        if self.num_dvt_channels > 0:
            self.head_dvt = nn.Conv1d(in_channels, self.num_dvt_channels, kernel_size=1)

        # --- 4. 初始化输出头权重 ---
        self.init_head_weights()

    def init_head_weights(self):
        """
        根据 config.py 复现输出头的特定初始化
        """
        # 脉冲头
        truncated_normal_(self.head_spike.weight, std=self.cfg.MODEL_CONFIG["OUTPUT_SPIKE_INIT_STDDEV"])
        nn.init.constant_(self.head_spike.bias, self.cfg.MODEL_CONFIG["OUTPUT_SPIKE_INIT_BIAS"])

        # Soma 头
        truncated_normal_(self.head_soma.weight, std=self.cfg.MODEL_CONFIG["OUTPUT_SOMA_INIT_STDDEV"])
        nn.init.constant_(self.head_soma.bias, 0)

        # 树突头
        if self.num_dvt_channels > 0:
            truncated_normal_(self.head_dvt.weight, std=self.cfg.MODEL_CONFIG["OUTPUT_DEND_INIT_STDDEV"])
            nn.init.constant_(self.head_dvt.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入 (B, C_in, T_in)

        Returns:
            (spike_logits, soma_voltage, dvt_voltage)
            dvt_voltage (Tensor or None): 如果 num_dvt_channels 为 0，则为 None。
        """
        # 1. 通过核心网络
        # x: (B, C_in, T_in) -> (B, C_out, T_in)
        features = self.network(x)

        # 2. 通过输出头
        # (B, C_out, T_in) -> (B, 1, T_in)
        spike_logits = self.head_spike(features)
        soma_voltage = self.head_soma(features)

        dvt_voltage = None
        if self.num_dvt_channels > 0:
            # (B, C_out, T_in) -> (B, N_dvt, T_in)
            dvt_voltage = self.head_dvt(features)

        return spike_logits, soma_voltage, dvt_voltage