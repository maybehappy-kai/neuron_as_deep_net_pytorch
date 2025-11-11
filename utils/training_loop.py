"""
复现原项目的复杂训练循环逻辑
"""
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR # 导入调度器

# 导入我们自己的模块
from utils.evaluation import predict_full_sequence, evaluate_metrics
import config

def get_current_schedule(policy, values, current_epoch, index_in_policy):
    """
    根据当前 epoch 从调度策略中动态获取值。

    Args:
        policy (list): 形如 [(epoch_milestone, value_index), ...] 的策略
        values (list): 索引对应的值
        current_epoch (int): 当前的外层 epoch (从 0 开始)

    Returns:
        对应的 value
    """
    current_value_index = 0
    # 遍历策略，找到最后一个小于等于当前 epoch 的节点
    for policy_entry in policy:
        epoch_milestone = policy_entry[0]
        value_index = policy_entry[index_in_policy] # 从 (epoch, loss_idx, batch_idx) 中选择

        if current_epoch >= epoch_milestone:
            current_value_index = value_index
        else:
            # 已经超过了当前 epoch，停止
            break

    return values[current_value_index]


def train_one_epoch(model, loader, optimizer, loss_fn_spike, loss_fn_soma, loss_fn_dvt,
                    loss_weights, device, sub_epoch_idx):
    """
    执行一个“子-epoch”（即遍历一次 train_loader）。
    这对应原 Keras fit_generator 的 1 epoch
    """
    model.train() # 设置为训练模式

    total_loss = 0.0
    total_spike_loss = 0.0
    total_soma_loss = 0.0
    total_dvt_loss = 0.0

    # Keras fit_generator(steps_per_epoch=100) -> 对应 100 步
    # 我们的 loader 长度被 data_loader.py 设置为 100
    progress_bar = tqdm(loader, desc=f"  > Sub-Epoch {sub_epoch_idx+1}/{config.TRAIN_CONFIG['NUM_STEPS_MULTIPLIER']}", leave=False)

    for batch_X, batch_Y_dict in progress_bar:
        # B, C, T
        batch_X = batch_X.to(device)
        # (B, 1, T) 或 (B, C_dvt, T)
        true_spike = batch_Y_dict['spike'].to(device)
        true_soma = batch_Y_dict['soma'].to(device)

        has_dvt = 'dvt' in batch_Y_dict
        if has_dvt:
            true_dvt = batch_Y_dict['dvt'].to(device)

        # 1. 前向传播
        optimizer.zero_grad()
        spike_logits, soma_pred, dvt_pred = model(batch_X)

        # 2. 计算损失
        loss_spike = loss_fn_spike(spike_logits, true_spike)
        loss_soma = loss_fn_soma(soma_pred, true_soma)

        w_spike, w_soma, w_dvt = loss_weights
        batch_total_loss = w_spike * loss_spike + w_soma * loss_soma

        if has_dvt:
            loss_dvt = loss_fn_dvt(dvt_pred, true_dvt)
            batch_total_loss += w_dvt * loss_dvt
            total_dvt_loss += loss_dvt.item()
        else:
            loss_dvt = 0.0

        # 3. 反向传播
        batch_total_loss.backward()
        optimizer.step()

        # 4. 记录损失
        total_loss += batch_total_loss.item()
        total_spike_loss += loss_spike.item()
        total_soma_loss += loss_soma.item()

    # 返回平均训练损失
    avg_loss = total_loss / len(loader)
    avg_spike_loss = total_spike_loss / len(loader)
    avg_soma_loss = total_soma_loss / len(loader)
    avg_dvt_loss = (total_dvt_loss / len(loader)) if has_dvt else 0.0

    return avg_loss, avg_spike_loss, avg_soma_loss, avg_dvt_loss


def run_training_loop(model, optimizer, train_loader, validation_data, cfg, output_dir):
    """
    运行复现的完整训练循环 (250个“外层epoch”)。
    """
    print("--- 步骤 5: 开始复现训练循环 ---")

    # --- 从 config.py 中获取新策略 ---
    num_outer_epochs = cfg.TRAIN_CONFIG["NUM_EPOCHS"]
    num_sub_epochs = cfg.TRAIN_CONFIG["NUM_STEPS_MULTIPLIER"]

    # 获取策略定义
    lr_policy = cfg.TRAIN_CONFIG["LR_POLICY"]
    dynamic_policy = cfg.TRAIN_CONFIG["DYNAMIC_SCHEDULE_POLICY"]
    loss_weight_values = cfg.TRAIN_CONFIG["LOSS_WEIGHT_VALUES"]
    batch_size_values = cfg.TRAIN_CONFIG["BATCH_SIZE_VALUES"] # (注意: data_loader 仍使用第一个值)

    # 验证数据 (已在GPU上)
    val_X_full, val_Y_dict_full = validation_data

    # 损失函数
    loss_fn_spike = nn.BCEWithLogitsLoss()
    loss_fn_soma = nn.MSELoss()
    loss_fn_dvt = nn.MSELoss() if cfg.DATA_CONFIG["DVT_PCA_COMPONENTS"] > 0 else None

    # --- 初始化学习率调度器 ---
    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=lr_policy["MILESTONES"],
        gamma=lr_policy["GAMMA"]
    )

    # 检查: 确保初始 LR 与优化器一致
    # (AdamW 在 main_run.py 中已使用 INITIAL_LR 初始化)

    log_data = [] # 收集日志
    best_val_loss = float('inf')

    # --- 外层循环 (现在可以灵活设置次数) ---
    # 对应原项目的 learning_schedule 循环
    for outer_epoch in range(num_outer_epochs):

        print(f"\n--- 外层 Epoch {outer_epoch+1}/{num_outer_epochs} ---")

        # 1. 动态获取当前Epoch的参数

        # (注意: 批次大小在 data_loader 创建时已固定,
        #  我们这里只是读取它以供未来使用或记录)
        current_batch_size = get_current_schedule(
            dynamic_policy, batch_size_values, outer_epoch, index_in_policy=2
        )

        current_loss_weights = get_current_schedule(
            dynamic_policy, loss_weight_values, outer_epoch, index_in_policy=1
        )

        # 从调度器获取当前 LR (用于记录)
        current_lr = lr_scheduler.get_last_lr()[0]

        # (不再需要手动设置优化器 LR)
        print(f"  > 当前 LR = {current_lr}")
        print(f"  > 当前 Loss Weights (Spike, Soma, DVT) = {current_loss_weights}")
        # print(f"  > (当前批次大小: {current_batch_size})") # (可选)

        # 累计子-epoch的训练损失
        train_losses = []
        train_spike_losses = []
        train_soma_losses = []
        train_dvt_losses = []

        # --- 内层循环 (10次) ---
        # 对应原 Keras fit_generator(epochs=num_steps_multiplier)
        for sub_epoch in range(num_sub_epochs):
            avg_loss, avg_spike, avg_soma, avg_dvt = train_one_epoch(
                model, train_loader, optimizer,
                loss_fn_spike, loss_fn_soma, loss_fn_dvt,
                current_loss_weights, cfg.DEVICE, sub_epoch
            )
            train_losses.append(avg_loss)
            train_spike_losses.append(avg_spike)
            train_soma_losses.append(avg_soma)
            train_dvt_losses.append(avg_dvt)

        # 计算10个子-epoch的平均训练损失
        avg_train_loss_total = np.mean(train_losses)
        avg_train_loss_spike = np.mean(train_spike_losses)
        avg_train_loss_soma = np.mean(train_soma_losses)
        avg_train_loss_dvt = np.mean(train_dvt_losses)

        print(f"  > 外层 Epoch {outer_epoch+1} 训练完成。")
        print(f"    > 平均训练损失 (Total): {avg_train_loss_total:.6f}")

        # --- 3. 运行完整验证 (在10个子-epoch后) ---
        print(f"  > 正在对 {val_X_full.shape[0]} 步的完整验证集进行评估...")

        # 3a. 在完整验证集上运行重叠预测
        val_predictions = predict_full_sequence(model, val_X_full, cfg)

        # 3b. 计算所有指标
        val_results = evaluate_metrics(val_predictions, val_Y_dict_full, current_loss_weights, cfg)

        print(f"  > 验证集评估完成:")
        print(f"    > Val Loss: {val_results['loss_total']:.6f} | Val AUC: {val_results['spike_AUC']:.4f} | Val VE: {val_results['soma_VE']:.4f}")
        print(f"    > Val F1: {val_results['spike_F1']:.4f} (P: {val_results['spike_Precision']:.4f}, R: {val_results['spike_Recall']:.4f} at Thresh: {val_results['spike_optimal_threshold']:.4f})")

        # 4. 记录日志
        log_entry = {
            "epoch": outer_epoch + 1,
            "lr": current_lr,
            "train_loss_total": avg_train_loss_total,
            "train_loss_spike": avg_train_loss_spike,
            "train_loss_soma": avg_train_loss_soma,
            "train_loss_dvt": avg_train_loss_dvt,
            "val_loss_total": val_results['loss_total'],
            "val_loss_spike": val_results['loss_spike'],
            "val_loss_soma": val_results['loss_soma'],
            "val_loss_dvt": val_results.get('loss_dvt', 0.0), # .get() 确保DVT=0时也能工作
            "val_VE_soma": val_results['soma_VE'],
            "val_AUC_spike": val_results['spike_AUC'],
            "val_Precision_spike": val_results['spike_Precision'],
            "val_Recall_spike": val_results['spike_Recall'],
            "val_F1_spike": val_results['spike_F1'],
            "val_optimal_threshold": val_results['spike_optimal_threshold']
        }
        log_data.append(log_entry)

        # 5. 保存最佳模型 (基于验证集总损失)
        if val_results['loss_total'] < best_val_loss:
            best_val_loss = val_results['loss_total']
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  > 新的最佳模型（Val Loss: {best_val_loss:.6f}）。已保存到 {save_path}")

        # --- 6. 更新学习率调度器 ---
        # 在 epoch 循环的末尾调用 .step()
        lr_scheduler.step()

    print("--- 训练循环已完成 ---")

    # 将日志列表转换为 DataFrame
    log_df = pd.DataFrame(log_data)

    return log_df