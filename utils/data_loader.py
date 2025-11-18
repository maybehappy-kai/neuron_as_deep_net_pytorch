"""
数据加载和预处理模块
"""
import torch
import numpy as np
import joblib
import os
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 导入我们在 config.py 中定义的配置
import config

def find_local_maxima_with_threshold(v_soma, threshold):
    """
    精确复现原项目的脉冲检测逻辑。
    查找所有高于'threshold' (-25mV)的局部最大点。
    """
    print(f"  > 正在检测脉冲 (阈值={threshold}mV)...")

    # 检查是否高于阈值
    above_threshold = v_soma > threshold

    # 检查是否为局部最大点 (比前后邻居都大)
    # 1. (v[t] > v[t-1])
    rising_before = np.hstack((True, v_soma[1:] > v_soma[:-1]))
    # 2. (v[t] > v[t+1])
    falling_after = np.hstack((v_soma[:-1] > v_soma[1:], True))

    local_maxima = np.logical_and(rising_before, falling_after)

    # 最终的脉冲是两者交集
    spikes_binary = np.logical_and(above_threshold, local_maxima)

    print(f"  > 检测到 {np.sum(spikes_binary)} 个脉冲。")
    return spikes_binary.astype(np.float32)


class NeuronRandomWindowDataset(Dataset):
    """
    PyTorch Dataset，用于复现原项目的随机窗口采样逻辑。

    在每次 `__getitem__` 调用时，它会忽略索引，
    并从 [ignore_steps_start, total_length - window_size] 范围内
    随机选择一个起始点来返回一个窗口。
    """
    def __init__(self, X, Y_dict, window_size, steps_per_epoch, ignore_steps_start=0):
        """
        Args:
            X (np.array): 输入数据 (T, C_in)
            Y_dict (dict): 包含目标张量的字典 {'spike': (T, 1), 'soma': (T, 1), ...}
            window_size (int): 要采样的时间窗口大小 (单位：时间步)
            steps_per_epoch (int): 每个epoch要生成的批次总数 (用于定义 __len__)
            ignore_steps_start (int): 从数据开头要忽略的时间步数
        """
        self.X = torch.from_numpy(X).float()
        self.Y_spike = torch.from_numpy(Y_dict['spike']).float()
        self.Y_soma = torch.from_numpy(Y_dict['soma']).float()

        self.has_dvt = 'dvt_pca' in Y_dict
        if self.has_dvt:
            self.Y_dvt = torch.from_numpy(Y_dict['dvt_pca']).float()

        self.window_size = window_size
        self.steps_per_epoch = steps_per_epoch

        total_length = X.shape[0]

        # 定义可供采样的有效起始索引范围
        # 复现原项目的 ignore_time_at_start_ms
        self.min_start_index = ignore_steps_start
        self.max_start_index = total_length - window_size - 1

        if self.min_start_index >= self.max_start_index:
            raise ValueError("数据长度不足以在'ignore_steps_start'后采样一个完整窗口。")

        print(f"  > 随机窗口采样器已创建：")
        print(f"    - 窗口大小: {window_size} 步")
        print(f"    - 有效采样范围: [{self.min_start_index}, {self.max_start_index}] 步")

    def __len__(self):
        # 长度定义为每个epoch的步数
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # 忽略 'idx'，总是返回一个随机窗口

        # 1. 随机选择一个起始点
        start_idx = torch.randint(self.min_start_index, self.max_start_index + 1, (1,)).item()
        end_idx = start_idx + self.window_size

        # 2. 切片
        X_window = self.X[start_idx:end_idx]
        Y_spike_window = self.Y_spike[start_idx:end_idx]
        Y_soma_window = self.Y_soma[start_idx:end_idx]

        # 3. 组装目标
        # (B, T, C) -> (B, C, T) 以匹配Conv1D的 (N, C_in, L_in)
        # PyTorch Conv1D 默认需要 (批次大小, 通道数, 时间步)
        X_window = X_window.permute(1, 0)
        Y_spike_window = Y_spike_window.permute(1, 0)
        Y_soma_window = Y_soma_window.permute(1, 0)

        targets = {
            'spike': Y_spike_window,
            'soma': Y_soma_window
        }

        if self.has_dvt:
            Y_dvt_window = self.Y_dvt[start_idx:end_idx]
            Y_dvt_window = Y_dvt_window.permute(1, 0)
            targets['dvt'] = Y_dvt_window

        return X_window, targets

def get_data_loaders(cfg, output_dir):
    """
    主数据加载函数。
    加载、预处理、划分数据，并返回PyTorch DataLoaders和评估数据。
    """
    print("--- 步骤 1: 加载原始数据 ---")

    # --- 使用 config 中的路径加载 ---
    input_path = cfg.DATA_CONFIG["INPUT_FILE_PATH"]
    target_path = cfg.DATA_CONFIG["TARGET_FILE_PATH"]

    if not os.path.exists(input_path) or not os.path.exists(target_path):
        print(f"[错误] 数据文件未找到。")
        print(f"  - 尝试加载: {input_path}")
        print(f"  - 尝试加载: {target_path}")
        raise FileNotFoundError("请确保 config.py 中的路径正确，或者通过 --data_subdir 提供了正确的子目录。")

    X_raw = np.load(input_path)
    Y_raw = np.load(target_path)

    # --- 加载通道类型信息 ---
    data_dir = os.path.dirname(input_path)
    channel_info_path = os.path.join(data_dir, "channel_info.json")
    exc_indices = None
    inh_indices = None

    try:
        with open(channel_info_path, 'r') as f:
            channel_info = json.load(f)
            exc_indices = channel_info.get('exc_indices', [])
            inh_indices = channel_info.get('inh_indices', [])
            print(f"  > 成功加载 {channel_info_path}。")
            print(f"    - 找到 {len(exc_indices)} 个兴奋性通道。")
            print(f"    - 找到 {len(inh_indices)} 个抑制性通道。")
    except FileNotFoundError:
        print(f"  > [警告] 未找到 'channel_info.json' (路径: {channel_info_path})。")
        print(f"  > TCN 卷积核可视化将显示所有通道，不区分E/I。")
    except Exception as e:
        print(f"  > [错误] 加载 {channel_info_path} 失败: {e}")


    # 检查shape
    if X_raw.shape[0] != Y_raw.shape[0]:
        raise ValueError("输入和目标文件的时间步数量不匹配!")

    total_steps = X_raw.shape[0]

    # --- 提取时间步长 (仅用于打印) ---
    timestamps = X_raw[:, 0]
    time_step_ms = np.mean(np.diff(timestamps))
    print(f"  > 检测到 {total_steps} 个时间步。")
    print(f"  > 平均时间步长: {time_step_ms:.4f} ms。")

    # --- 分离特征和目标 ---
    # 输入: 通道 1 及之后 (通道 0 是时间戳)
    X_data = X_raw[:, 1:].astype(np.float32)

    # 目标:
    # 通道 1 是 Soma 电压
    Y_soma_raw = Y_raw[:, 1].astype(np.float32)

    # --- 1. 修改：DVT 自动化检测与自适应策略 ---
    Y_dvt_raw = None
    # 从 config 中获取用户 *期望* 的成分数量 (通常为 20)
    requested_pca_components = cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"]

    if Y_raw.shape[1] > 2:
        # 1. 自动检测到数据 (Y_raw[:, 0]=time, [:, 1]=soma, [:, 2+]=dvt)
        num_dvt_actual = Y_raw.shape[1] - 2
        print(f"  > 自动检测到 {num_dvt_actual} 个树突电压通道。")
        Y_dvt_raw = Y_raw[:, 2:].astype(np.float32)

        if requested_pca_components <= 0:
            # 2. 用户在 config.py 中设置了 0，但数据存在
            print(f"  > [警告] 检测到树突数据，但 config.py 中的 DVT_PCA_COMPONENTS = 0。")
            print(f"  > 将忽略树突数据。")
            cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] = 0 # 确保为 0
            Y_dvt_raw = None # 丢弃数据
        else:
            # 3. 用户设置了 > 0，数据也存在
            # --- 关键修改：不足 20 (requested) 时，自动下调维度，但仍进行 PCA ---
            if num_dvt_actual < requested_pca_components:
                print(f"  > [自动调整] 实际通道数 ({num_dvt_actual}) 小于配置的 PCA 维度 ({requested_pca_components})。")
                print(f"  > 将 PCA 维度调整为 {num_dvt_actual} (以便执行全秩 PCA + 白化)。")
                # 更新配置，确保后续 PCA 逻辑使用正确的维度
                cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] = num_dvt_actual
            else:
                print(f"  > 实际通道数 ({num_dvt_actual}) 充足。将执行标准 PCA 降维至 {requested_pca_components}。")
            # --- 修改结束 ---

    else:
        # 4. 未检测到数据
        if requested_pca_components > 0:
            # 5. 用户设置了 > 0，但数据不存在
            print(f"  > [警告] config.py 中 DVT_PCA_COMPONENTS = {requested_pca_components}，但未在数据中找到树突通道。")

        # 自动将 *有效* 成分数量设为 0
        cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] = 0

    del X_raw, Y_raw # 释放内存

    # --- 生成脉冲目标 ---
    Y_spike = find_local_maxima_with_threshold(
        Y_soma_raw, cfg.DATA_CONFIG["SPIKE_THRESHOLD_MV"]
    )
    # 调整shape为 (T, 1)
    Y_spike = Y_spike.reshape(-1, 1)

    # --- 数据划分 (6:1:3) ---
    print("--- 步骤 2: 划分数据 (6:1:3) ---")
    split_ratio = cfg.DATA_CONFIG["DATA_SPLIT_RATIO"]
    train_size = int(total_steps * split_ratio[0])
    val_size = int(total_steps * split_ratio[1])

    # (T, C)
    X_train, X_temp = X_data[:train_size], X_data[train_size:]
    Y_soma_train, Y_soma_temp = Y_soma_raw[:train_size], Y_soma_raw[train_size:]
    Y_spike_train, Y_spike_temp = Y_spike[:train_size], Y_spike[train_size:]

    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    Y_soma_val, Y_soma_test = Y_soma_temp[:val_size], Y_soma_temp[val_size:]
    Y_spike_val, Y_spike_test = Y_spike_temp[:val_size], Y_spike_temp[val_size:]

    # --- 2. 修改：此处的 Y_dvt_raw 已经是 None 或者 真实数据 ---
    if Y_dvt_raw is not None:
        Y_dvt_train, Y_dvt_temp = Y_dvt_raw[:train_size], Y_dvt_raw[train_size:]
        Y_dvt_val, Y_dvt_test = Y_dvt_temp[:val_size], Y_dvt_temp[val_size:]

    print(f"  > 训练集: {X_train.shape[0]} 步")
    print(f"  > 验证集: {X_val.shape[0]} 步")
    print(f"  > 测试集: {X_test.shape[0]} 步")

    # --- 拟合与应用预处理 ---
    # 严格复现原项目的预处理

    print("--- 步骤 3: 拟合与应用预处理 ---")

    # 6a. Soma 电压: 截断 + 减去偏置
    print(f"  > 应用Soma电压截断 (阈值={cfg.DATA_CONFIG['SOMA_VOLTAGE_THRESHOLD_MV']}mV)...")
    np.clip(Y_soma_train, None, cfg.DATA_CONFIG['SOMA_VOLTAGE_THRESHOLD_MV'], out=Y_soma_train)
    np.clip(Y_soma_val, None, cfg.DATA_CONFIG['SOMA_VOLTAGE_THRESHOLD_MV'], out=Y_soma_val)
    np.clip(Y_soma_test, None, cfg.DATA_CONFIG['SOMA_VOLTAGE_THRESHOLD_MV'], out=Y_soma_test)

    soma_bias = cfg.DATA_CONFIG["Y_SOMA_BIAS"]
    if soma_bias == 'auto':
        soma_bias = np.mean(Y_soma_train)
        print(f"  > 自动计算Soma偏置: {soma_bias:.4f}")
    else:
        print(f"  > 使用固定Soma偏置: {soma_bias}")

    Y_soma_train -= soma_bias
    Y_soma_val -= soma_bias
    Y_soma_test -= soma_bias

    # 调整shape为 (T, 1)
    Y_soma_train = Y_soma_train.reshape(-1, 1)
    Y_soma_val = Y_soma_val.reshape(-1, 1)
    Y_soma_test -= soma_bias

    # 调整shape为 (T, 1)
    Y_soma_train = Y_soma_train.reshape(-1, 1)
    Y_soma_val = Y_soma_val.reshape(-1, 1)
    Y_soma_test = Y_soma_test.reshape(-1, 1)

    # --- 3. 修改：只要 PCA 维度 > 0 且数据存在，始终应用 PCA ---
    pca_model = None

    # 这里的判断条件修改了：不再检查 should_apply_pca
    if cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] > 0 and Y_dvt_raw is not None:
        print(f"  > 拟合PCA (n={cfg.DATA_CONFIG['DVT_PCA_COMPONENTS']})，仅使用训练集...")
        pca_model = PCA(n_components=cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"], whiten=True)
        # 仅在训练集上拟合
        pca_model.fit(Y_dvt_train)

        # 在所有数据集上转换
        Y_dvt_train = pca_model.transform(Y_dvt_train)
        Y_dvt_val = pca_model.transform(Y_dvt_val)
        Y_dvt_test = pca_model.transform(Y_dvt_test)

        # 应用截断
        dvt_thresh = cfg.DATA_CONFIG["DVT_THRESHOLD"]
        print(f"  > 应用PCA分量截断 (阈值= +/- {dvt_thresh})...")
        np.clip(Y_dvt_train, -dvt_thresh, dvt_thresh, out=Y_dvt_train)
        np.clip(Y_dvt_val, -dvt_thresh, dvt_thresh, out=Y_dvt_val)
        np.clip(Y_dvt_test, -dvt_thresh, dvt_thresh, out=Y_dvt_test)

    # --- 保存标量和PCA模型 ---
    scalers = {'soma_bias': soma_bias}
    joblib.dump(scalers, os.path.join(output_dir, "data_scalers.pkl"))
    if pca_model:
        joblib.dump(pca_model, os.path.join(output_dir, "pca_model.pkl"))

    # --- 创建 DataLoaders ---
    print("--- 步骤 4: 创建 DataLoaders ---")

    # 训练集: 使用随机窗口采样
    train_targets = {'spike': Y_spike_train, 'soma': Y_soma_train}
    # --- 4. 修改： 检查 Y_dvt_raw 是否存在 ---
    if Y_dvt_raw is not None:
        train_targets['dvt_pca'] = Y_dvt_train

    # --- 修复第 1 步：将批次大小的定义移到 Dataset 创建之前 ---
    # (原始位置在第 220 行)
    # (注意: 我们在问题1中已将调度改为策略，但 data_loader 仍然
    #  只在初始化时读取一次批次大小，这是符合预期的。)
    batch_size = cfg.TRAIN_CONFIG["BATCH_SIZE_VALUES"][0]

    # --- 原始 STEPS_PER_SUB_EPOCH 定义 (保持不变) ---
    # (移除了硬编码的 steps_per_sub_epoch = 100)
    steps_per_sub_epoch = cfg.TRAIN_CONFIG["STEPS_PER_SUB_EPOCH"]

    train_dataset = NeuronRandomWindowDataset(
        X_train,
        train_targets,
        window_size=cfg.MODEL_CONFIG["INPUT_WINDOW_SIZE"],

        # --- 修复第 2 步：修正此行逻辑 ---
        # 原始错误逻辑: steps_per_epoch=steps_per_sub_epoch * cfg.TRAIN_CONFIG["NUM_STEPS_MULTIPLIER"],
        # 修正后的正确逻辑:
        steps_per_epoch=steps_per_sub_epoch * batch_size,
        # ---------------------------------

        ignore_steps_start=cfg.TRAIN_CONFIG["IGNORE_TIME_FROM_START"]
    )

    # 批次大小从调度中获取第一个值
    # (原始第 220 行的 batch_size 定义现在已移至上方)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, # (现在使用上面定义的 batch_size)
        shuffle=False, # shuffle=False 因为我们的Dataset.__getitem__已经实现了随机采样
        num_workers=4,
        pin_memory=True
    )

    # 验证集和测试集: 返回完整数据以供评估函数处理

    # 组装验证集
    val_targets = {'spike': Y_spike_val, 'soma': Y_soma_val}
    if Y_dvt_raw is not None:
        val_targets['dvt_pca'] = Y_dvt_val

    # 组装测试集
    test_targets = {'spike': Y_spike_test, 'soma': Y_soma_test}
    if Y_dvt_raw is not None:
        test_targets['dvt_pca'] = Y_dvt_test

    # 将数据转换为张量以便评估
    X_val_tensor = torch.from_numpy(X_val).float().to(cfg.DEVICE)
    X_test_tensor = torch.from_numpy(X_test).float().to(cfg.DEVICE)

    Y_val_tensor = {k: torch.from_numpy(v).float().to(cfg.DEVICE) for k, v in val_targets.items()}
    Y_test_tensor = {k: torch.from_numpy(v).float().to(cfg.DEVICE) for k, v in test_targets.items()}

    # --- 打包数据信息 ---
    data_info = {
        "time_step_ms": time_step_ms,
        "input_channels": X_train.shape[1],
        # --- 5. 修改：这里现在是“有效”通道数 (0 或 用户请求的值) ---
        "dvt_channels": cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"],
        "scalers": scalers,
        "pca_model": pca_model,
        "exc_indices": exc_indices,
        "inh_indices": inh_indices
    }

    print("--- 数据加载完成 ---")

    return train_loader, (X_val_tensor, Y_val_tensor), (X_test_tensor, Y_test_tensor), data_info