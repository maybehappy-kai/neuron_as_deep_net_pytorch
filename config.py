"""
项目配置文件
"""
import torch

# --- 1. 全局设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 结果将保存在 "results/RUN_NAME" 中
RUN_NAME = "TCN_replication_run_1"

# --- 2. 数据集参数 ---
DATA_CONFIG = {
    # 路径
    "INPUT_FILE_PATH": "data/spike_matrix_0.02.npy", # 您的输入数据路径
    "TARGET_FILE_PATH": "data/V_soma_0.02.npy",    # 您的目标数据路径

    # 数据划分比例 (训练集, 验证集, 测试集)
    "DATA_SPLIT_RATIO": [0.6, 0.1, 0.3],

    # 目标生成 (脉冲)
    # 原项目使用-25mV作为阈值
    "SPIKE_THRESHOLD_MV": -25.0,

    # 预处理 (Soma电压)
    # 原项目在评估时使用-55mV截断
    "SOMA_VOLTAGE_THRESHOLD_MV": -55.0,
    # 原项目使用的Soma电压偏差
    "Y_SOMA_BIAS": -67.7,

    # 预处理 (树突电压)
    # 注意：如果您的 V_soma.npy 中没有树突通道，请将此设置为 0
    "DVT_PCA_COMPONENTS": 20, # 20, # 原项目NMDA使用20
    "DVT_THRESHOLD": 3.0, # PCA成分的截断阈值
}

# --- 3. 模型参数 ---
# 复现自 fit_CNN.py
# 重要：所有时间相关的参数（..._SIZE）现在都以“时间步”为单位，而不是ms
MODEL_CONFIG = {
    "MODEL_NAME": "tcn", # 用于主脚本动态加载模型

    # TCN 架构
    # 请根据您的时间步长（例如0.02ms）手动缩放这些值以匹配原项目的真实时间感受野
    # 示例：原项目 400ms / (0.02ms/步) = 20000 步
    "INPUT_WINDOW_SIZE": 400, # 原项目为 400 (步)

    # 示例：原项目 [54, 24, 24]ms -> [2700, 1200, 1200] 步 (在0.02ms步长下)
    "FILTER_SIZES": [54, 24, 24], # 原项目为 [54, 24, 24] (步)
    "NUM_FILTERS": 64,             # 原项目为 64

    # 其他TCN参数
    "NETWORK_DEPTH": 3,            # 必须等于 len(FILTER_SIZES)
    "L2_REGULARIZATION": 1e-8,     #
    "ACTIVATION_FUNCTION": "relu", #
    "INITIALIZER_STDDEV": 0.002,   #

    # 输出层初始化参数
    "OUTPUT_SPIKE_INIT_STDDEV": 0.001,
    "OUTPUT_SPIKE_INIT_BIAS": -2.0,
    "OUTPUT_SOMA_INIT_STDDEV": 0.03,
    "OUTPUT_DEND_INIT_STDDEV": 0.05,
}

# --- 4. 训练参数 ---
# 严格复现原项目的学习调度
NUM_EPOCHS = 1 # "外层" epoch 数量 (现在可以灵活设置)
NUM_STEPS_MULTIPLIER = 10 # 每个 "外层" epoch 包含的 "子-epoch" 数量

# --- 学习率调度 (新) ---
# 使用 PyTorch MultiStepLR 策略
LR_POLICY = {
    "INITIAL_LR": 0.0001,
    # 原 Keras 衰减点: 40, 80, 120, 160
    "MILESTONES": [40, 80, 120, 160],
    # 原 Keras 衰减值: 0.00003, 0.00001, 0.000003, 0.000001
    # 我们可以用 0.3 的 gamma 来近似 (0.3, 0.3, 0.3, 0.3)
    "GAMMA": 0.3
}

# --- 批次大小和损失权重的动态调度 (新) ---
# 定义索引对应的具体值
DVT_LOSS_MULT_FACTOR = 0.1
LOSS_WEIGHT_VALUES = [
    # 索引 0 (Epochs 0-39)
    [1.0, 0.0200, DVT_LOSS_MULT_FACTOR * 0.00005],
    # 索引 1 (Epochs 40-79)
    [2.0, 0.0100, DVT_LOSS_MULT_FACTOR * 0.00003],
    # 索引 2 (Epochs 80-119)
    [4.0, 0.0100, DVT_LOSS_MULT_FACTOR * 0.00001],
    # 索引 3 (Epochs 120-159)
    [8.0, 0.0100, DVT_LOSS_MULT_FACTOR * 0.0000001],
    # 索引 4 (Epochs 160+)
    [9.0, 0.0030, DVT_LOSS_MULT_FACTOR * 0.00000001]
]

BATCH_SIZE_VALUES = [
    # 索引 0 (默认)
    512
]

# 定义策略：(Epoch 开始节点, 损失权重索引, 批次大小索引)
DYNAMIC_SCHEDULE_POLICY = [
    (0,   0, 0),
    (40,  1, 0),
    (80,  2, 0),
    (120, 3, 0),
    (160, 4, 0)
]

TRAIN_CONFIG = {
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_STEPS_MULTIPLIER": NUM_STEPS_MULTIPLIER,

    # --- 新增：将 "魔术数字" 100 移到此处 ---
    "STEPS_PER_SUB_EPOCH": 100,

    # 窗口采样
    # 原项目在采样时忽略前500ms
    # 请根据您的时间步长手动缩放（例如 500ms / 0.02ms/步 = 25000 步）
    "IGNORE_TIME_FROM_START": 500, # (单位：时间步)

    # --- 学习调度 (新) ---
    "LR_POLICY": LR_POLICY,
    "LOSS_WEIGHT_VALUES": LOSS_WEIGHT_VALUES,
    "BATCH_SIZE_VALUES": BATCH_SIZE_VALUES,
    "DYNAMIC_SCHEDULE_POLICY": DYNAMIC_SCHEDULE_POLICY,
}

# --- 5. 评估参数 ---
EVAL_CONFIG = {
    "BATCH_SIZE": 32, # 评估时使用稍大的批次大小以加快速度
    "VISUALIZATION_SLICE_SECONDS": 6.0, # 6秒的可视化切片
}