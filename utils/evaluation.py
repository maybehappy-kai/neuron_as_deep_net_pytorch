"""
评估模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, explained_variance_score
from tqdm import tqdm
import config

def get_optimal_threshold(y_true, y_pred_prob):
    """
    计算ROC曲线并找到最佳阈值（最大化 TPR - FPR）。
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Youden's J-statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, (fpr, tpr, thresholds)

@torch.no_grad() # 确保在评估期间不计算梯度
def predict_full_sequence(model, X_full, cfg):
    """
    在完整的长序列上执行 "Overlap-Splice" 风格的预测。
    (已修改：根据用户反馈，实现“先写获胜”逻辑，而非“平均”)

    Args:
        model (nn.Module): TCN模型。
        X_full (Tensor): (T, C_in) 完整的输入张量（在GPU上）。
        cfg (module): 配置文件。

    Returns:
        dict: 包含完整预测序列的字典 {'spike': (T, 1), 'soma': (T, 1), ...}
    """
    model.eval() # 设置为评估模式

    window_size = cfg.MODEL_CONFIG["INPUT_WINDOW_SIZE"]
    batch_size = cfg.EVAL_CONFIG["BATCH_SIZE"]
    device = cfg.DEVICE

    # 切换维度 (T, C) -> (C, T) 以便切片
    X_full_c_t = X_full.permute(1, 0)

    total_len = X_full.shape[0]
    input_channels = X_full.shape[1]

    # 初始化输出张量（在GPU上）
    pred_spike_logits = torch.zeros((total_len, 1), device=device)
    pred_soma = torch.zeros((total_len, 1), device=device)

    has_dvt = cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] > 0
    if has_dvt:
        pred_dvt = torch.zeros((total_len, cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"]), device=device)

    # --- 修改 ---
    # overlap_counter 现在用作“写入锁” (0 = 可写, 1 = 已写入)
    overlap_counter = torch.zeros((total_len, 1), device=device)

    # --- 开始替换 ---
    # 从 config 中获取模型架构
    filter_sizes = np.array(cfg.MODEL_CONFIG["FILTER_SIZES"])
    input_window_size = cfg.MODEL_CONFIG["INPUT_WINDOW_SIZE"]

    # 1. 计算 Keras 版本中的 "time_window_T" (模型的感受野)
    #    (来自 main_figure_replication.py)
    time_window_T = (filter_sizes - 1).sum() + 1

    # 2. 计算 Keras 版本中的 "overlap_size"
    #    (来自 main_figure_replication.py)
    overlap_size = min(max(time_window_T + 1, min(150, input_window_size - 50)), 250)

    # 3. Keras 版本的有效步长 (stride) 是窗口大小减去重叠部分
    stride = input_window_size - overlap_size

    print(f"  > [动态步长] time_window_T: {time_window_T}, overlap_size: {overlap_size}, stride: {stride}")
    # --- 结束替换 ---

    # 创建批次起始点
    start_indices = list(range(0, total_len - window_size + 1, stride))
    # 确保最后一个窗口被包含
    if start_indices[-1] + window_size < total_len:
        start_indices.append(total_len - window_size)

    for i in tqdm(range(0, len(start_indices), batch_size), desc="  > 评估中", leave=False):
        batch_starts = start_indices[i:i+batch_size]

        # (B, C, T)
        batch_X = torch.stack(
            [X_full_c_t[:, s:s+window_size] for s in batch_starts]
        )

        # 模型推理
        spike_logits, soma, dvt = model(batch_X)

        # --- 修改：实现 "Overlap-Splice" (先写获胜) 逻辑 ---
        for j, start_idx in enumerate(batch_starts):
            end_idx = start_idx + window_size

            # 1. 确定哪些位置是 "可写" 的 (即 counter == 0)
            # mask: (T_window, 1)
            mask = (overlap_counter[start_idx:end_idx] == 0).float()

            # 2. 获取当前窗口的预测
            # (1, T) -> (T, 1)
            spike_logits_j = spike_logits[j].permute(1, 0)
            soma_j = soma[j].permute(1, 0)

            # 3. 仅在 "可写" 位置上写入
            # (因为 pred_* 初始值为 0, 使用加法 + mask 等效于选择性写入)
            pred_spike_logits[start_idx:end_idx] += spike_logits_j * mask
            pred_soma[start_idx:end_idx] += soma_j * mask

            if has_dvt:
                # (DVT_C, T) -> (T, DVT_C)
                dvt_j = dvt[j].permute(1, 0)
                # (T_window, DVT_C) * (T_window, 1) -> 广播
                pred_dvt[start_idx:end_idx] += dvt_j * mask

            # 4. 标记已写入的位置
            # (注意: 我们不能使用 'overlap_counter[start_idx:end_idx] = mask'
            #  因为 mask 只在重叠区的前半部分为 0, 后半部分为 1,
            #  我们需要的是将新写入的区域标记为 1)
            overlap_counter[start_idx:end_idx] += mask # (0+1=1, 1+0=1)

    # --- 移除：不再需要平均步骤 ---
    # pred_spike_logits /= overlap_counter (已删除)
    # pred_soma /= overlap_counter (已删除)
    # if has_dvt: ... (已删除)

    predictions = {
        'spike_logits': pred_spike_logits,
        'soma': pred_soma,
        'dvt_pca': pred_dvt if has_dvt else None
    }

    return predictions


def evaluate_metrics(predictions, targets, loss_weights, cfg):
    """
    计算所有评估指标。

    Args:
        predictions (dict): predict_full_sequence 的输出。
        targets (dict): 包含 'spike', 'soma', 'dvt_pca' (T, C) 张量的字典。
        loss_weights (list): [w_spike, w_soma, w_dvt]
        cfg (module): 配置文件。

    Returns:
        dict: 包含所有指标的字典。
    """

    # --- 1. 准备数据 (转移到 CPU/numpy) ---
    # logits (未激活)
    pred_spike_logits_cpu = predictions['spike_logits'].cpu().numpy().ravel()
    # 概率 (已激活)
    pred_spike_prob_cpu = torch.sigmoid(predictions['spike_logits']).cpu().numpy().ravel()

    pred_soma_cpu = predictions['soma'].cpu().numpy().ravel()

    true_spike_cpu = targets['spike'].cpu().numpy().ravel()
    true_soma_cpu = targets['soma'].cpu().numpy().ravel()

    results = {}

    # --- 2. 计算损失 (在 GPU 上) ---
    w_spike, w_soma, w_dvt = loss_weights

    # 脉冲损失 (BCEWithLogitsLoss)
    loss_spike = F.binary_cross_entropy_with_logits(
        predictions['spike_logits'], targets['spike']
    )

    # Soma 损失 (MSELoss)
    loss_soma = F.mse_loss(predictions['soma'], targets['soma'])

    total_loss = w_spike * loss_spike + w_soma * loss_soma

    results['loss_total'] = total_loss.item()
    results['loss_spike'] = loss_spike.item()
    results['loss_soma'] = loss_soma.item()

    # 树突损失 (如果存在)
    if cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] > 0:
        loss_dvt = F.mse_loss(predictions['dvt_pca'], targets['dvt_pca'])
        results['loss_dvt'] = loss_dvt.item()
        total_loss += w_dvt * loss_dvt
        results['loss_total'] = total_loss.item() # 更新总损失


    # --- 3. 计算Soma电压指标 (Numpy) ---
    # 复现原项目的 explained_variance_score
    results['soma_VE'] = explained_variance_score(true_soma_cpu, pred_soma_cpu)


    # --- 4. 计算脉冲指标 (Numpy) ---

    # 4a. AUC
    # 复现原项目的 roc_auc_score
    try:
        results['spike_AUC'] = roc_auc_score(true_spike_cpu, pred_spike_prob_cpu)
    except ValueError:
        results['spike_AUC'] = 0.5 # 以防万一数据中没有脉冲

    # 4b. 最佳阈值, P, R, F1
    optimal_threshold, roc_data_tuple = get_optimal_threshold(true_spike_cpu, pred_spike_prob_cpu)

    results['spike_optimal_threshold'] = optimal_threshold

    # 应用阈值
    pred_spike_binary = (pred_spike_prob_cpu >= optimal_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_spike_cpu,
        pred_spike_binary,
        average='binary',
        zero_division=0
    )

    results['spike_Precision'] = precision
    results['spike_Recall'] = recall
    results['spike_F1'] = f1

    # 保存ROC曲线数据以供绘图
    results['roc_curve_data'] = roc_data_tuple

    return results