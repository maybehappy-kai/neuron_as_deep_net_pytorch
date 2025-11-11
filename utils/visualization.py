"""
可视化模块
包含所有绘图函数 (ROC, 轨迹, 权重等)
"""
import matplotlib
from sklearn.metrics import auc

matplotlib.use('Agg') # 使用非交互式后端，防止在服务器上出错
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy import signal

# --- 1. 训练过程可视化 ---

def plot_learning_curves(log_df, save_path):
    """
    绘制训练和验证的损失及指标随 Epoch 变化的曲线。
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("Training and Validation Metrics", fontsize=20)

    # 总损失
    axes[0, 0].plot(log_df['epoch'], log_df['train_loss_total'], label='Train Loss')
    axes[0, 0].plot(log_df['epoch'], log_df['val_loss_total'], label='Validation Loss')
    axes[0, 0].set_title("Total Loss vs. Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (Log Scale)")
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()

    # Soma 电压 Explained Variance
    axes[0, 1].plot(log_df['epoch'], log_df['val_VE_soma'], label='Validation VE')
    axes[0, 1].set_title("Soma Voltage Explained Variance vs. Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Explained Variance")
    axes[0, 1].legend()

    # 脉冲 AUC
    axes[1, 0].plot(log_df['epoch'], log_df['val_AUC_spike'], label='Validation AUC')
    axes[1, 0].set_title("Spike AUC vs. Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].legend()

    # 脉冲 F1-Score
    axes[1, 1].plot(log_df['epoch'], log_df['val_F1_spike'], label='Validation F1-Score')
    axes[1, 1].set_title("Spike F1-Score (at Optimal Thresh) vs. Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

# --- 2. 评估结果可视化 ---

def plot_prediction_trace(y_true_soma, y_pred_soma, y_true_spike, y_pred_spike_binary, time_axis_s, title, save_path, zoom_range_s=None, time_step_ms=None):
    """
    绘制Soma电压的真实值与预测值对比图。
    如果提供了 zoom_range_s (单位：秒)，则会创建一个带缩放框的图。
    """

    # 将脉冲可视化为 40mV
    y_true_soma_spikes = y_true_soma.copy()
    y_pred_soma_spikes = y_pred_soma.copy()
    y_true_soma_spikes[y_true_spike == 1] = 40
    y_pred_soma_spikes[y_pred_spike_binary == 1] = 40

    if zoom_range_s is None:
        # --- 绘制单张完整轨迹图 ---
        fig, ax = plt.subplots(figsize=(25, 8))
        ax.plot(time_axis_s, y_true_soma_spikes, 'c', label='Ground Truth')
        ax.plot(time_axis_s, y_pred_soma_spikes, 'm:', label='ANN Prediction')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel("Voltage (mV)", fontsize=16)
        ax.legend()
        ax.set_xlim(time_axis_s[0], time_axis_s[-1])
    else:
        # --- 绘制带缩放框的图 (复现 Fig 2C) ---
        fig = plt.figure(figsize=(25, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        ax_full = plt.subplot(gs[0])
        ax_zoom = plt.subplot(gs[1])

        # 完整轨迹
        ax_full.plot(time_axis_s, y_true_soma_spikes, 'c', label='Ground Truth')
        ax_full.plot(time_axis_s, y_pred_soma_spikes, 'm:', label='ANN Prediction')
        ax_full.set_title(title, fontsize=20)
        ax_full.set_ylabel("Voltage (mV)", fontsize=16)
        ax_full.legend(loc='upper right')
        ax_full.set_xlim(time_axis_s[0], time_axis_s[-1])

        # 缩放轨迹
        ax_zoom.plot(time_axis_s, y_true_soma_spikes, 'c')
        ax_zoom.plot(time_axis_s, y_pred_soma_spikes, 'm:')
        ax_zoom.set_xlabel("Time (s)", fontsize=16)
        ax_zoom.set_ylabel("Voltage (mV)", fontsize=16)

        # 计算缩放范围
        zoom_start_s, zoom_end_s = zoom_range_s
        ax_zoom.set_xlim(zoom_start_s, zoom_end_s)

        # 计算Y轴范围
        # 将绝对时间 (zoom_start_s) 转换回相对于传入数组 (time_axis_s) 的索引
        start_idx = int((zoom_start_s - time_axis_s[0]) * 1000 / time_step_ms)
        end_idx = int((zoom_end_s - time_axis_s[0]) * 1000 / time_step_ms)
        min_val = min(y_true_soma[start_idx:end_idx].min(), y_pred_soma[start_idx:end_idx].min())
        ax_zoom.set_ylim(min_val - 5, -45) # 稍微低于最小值，最高到-45mV

        # 在完整图上绘制缩放框
        rect = mpatches.Rectangle(
            (zoom_start_s, min_val - 5),
            zoom_end_s - zoom_start_s,
            -45 - (min_val - 5),
            linewidth=1, edgecolor='k', linestyle='--', facecolor='none'
        )
        ax_full.add_patch(rect)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

def _plot_roc_curve(ax, y_true, y_pred_prob, optimal_threshold, roc_data):
    """
    辅助函数：绘制ROC曲线及最佳阈值放大图。
    """
    fpr, tpr, thresholds = roc_data
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, 'k', label=f'AUC = {roc_auc:.4f}')

    # 找到最佳阈值对应的点
    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.plot(fpr[opt_idx], tpr[opt_idx], 'r*', markersize=15, label=f'Best Threshold ({optimal_threshold:.2f})')

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("Spike Prediction ROC Curve", fontsize=16)
    ax.legend(loc="lower right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 放大图（Inset） ---
    # 复现 Fig 2D inset
    # 根据您的要求，在最佳阈值点附近放大
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='center right')
    ax_inset.plot(fpr, tpr, 'k')
    ax_inset.plot(fpr[opt_idx], tpr[opt_idx], 'r*', markersize=10)

    # 设置放大范围
    x_lim_min = max(0, fpr[opt_idx] - 0.01)
    x_lim_max = min(1.0, fpr[opt_idx] + 0.01)
    y_lim_min = max(0, tpr[opt_idx] - 0.1)
    y_lim_max = min(1.0, tpr[opt_idx] + 0.1)

    ax_inset.set_xlim(x_lim_min, x_lim_max)
    ax_inset.set_ylim(y_lim_min, y_lim_max)
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)

def _plot_voltage_scatter(ax, y_true_soma, y_pred_soma):
    """
    辅助函数：绘制Soma电压的真实值 vs 预测值散点图。
    """
    # 随机采样点以避免过密
    num_points = min(len(y_true_soma), 50000)
    indices = np.random.choice(len(y_true_soma), num_points, replace=False)

    ax.scatter(y_true_soma[indices], y_pred_soma[indices], s=1, alpha=0.5)
    ax.plot([y_true_soma.min(), y_true_soma.max()], [y_true_soma.min(), y_true_soma.max()], 'k--')
    ax.set_xlabel("Ground Truth Voltage (mV)", fontsize=14)
    ax.set_ylabel("ANN Prediction (mV)", fontsize=14)
    ax.set_title("Soma Voltage Prediction", fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _plot_spike_cross_correlation(ax, y_true_spike, y_pred_spike_binary, time_step_ms):
    """
    辅助函数：绘制脉冲交叉相关图。
    复现 Fig 2F
    """
    half_window_ms = 50
    half_window_steps = int(half_window_ms / time_step_ms)

    # 填充
    zero_padding = np.zeros(half_window_steps)
    true_padded = np.hstack((zero_padding, y_true_spike, zero_padding))
    pred_padded = np.hstack((zero_padding, y_pred_spike_binary, zero_padding))

    recall_curve = np.zeros(1 + 2 * half_window_steps)

    # 找到所有真实脉冲的索引
    trace_inds, spike_inds = np.nonzero(true_padded[np.newaxis, :]) # 增加一个维度以匹配原代码

    for spike_ind in spike_inds:
        window = pred_padded[spike_ind - half_window_steps : 1 + spike_ind + half_window_steps]
        recall_curve += window

    if recall_curve.sum() > 0:
        recall_curve /= recall_curve.sum()

    # 转换为Hz
    spike_rate_hz = recall_curve * (1000.0 / time_step_ms)

    time_axis_ms = np.arange(-half_window_steps, half_window_steps + 1) * time_step_ms

    ax.plot(time_axis_ms, spike_rate_hz, 'k')
    ax.set_xlabel("Delta t (ms)", fontsize=14)
    ax.set_ylabel("Spike Rate (Hz)", fontsize=14)
    ax.set_title("P(Prediction | GT Spike)", fontsize=16)
    ax.set_xticks([-50, 0, 50])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_evaluation_metrics(y_true_soma, y_pred_soma, y_true_spike, y_pred_prob_spike,
                            optimal_threshold, roc_data, time_step_ms, save_path):
    """
    创建并保存一个包含ROC、散点图和交叉相关的 1x3 评估图。
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. ROC 曲线
    _plot_roc_curve(axes[0], y_true_spike, y_pred_prob_spike, optimal_threshold, roc_data)

    # 2. 电压散点图
    _plot_voltage_scatter(axes[1], y_true_soma, y_pred_soma)

    # 3. 交叉相关图
    y_pred_spike_binary = (y_pred_prob_spike >= optimal_threshold).astype(int)
    _plot_spike_cross_correlation(axes[2], y_true_spike, y_pred_spike_binary, time_step_ms)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# --- 3. 模型可解释性 (TCN特定) ---

# --- 1. 修改函数签名 ---
def plot_tcn_kernels(model, time_step_ms, save_path, exc_indices=None, inh_indices=None):
    """
    可视化TCN第一层卷积核的权重。
    如果提供了 exc_indices 和 inh_indices，将分组绘图。
    否则，将所有通道绘制在一起。
    """
    print("  > 正在生成TCN卷积核可视化图...")
    try:
        # 1. 提取权重 (保持不变)
        weights = model.network[0].conv.weight.data.cpu().numpy()
        kernel_data = weights[0, :, :]
        kernel_data_flipped = np.fliplr(kernel_data)
        kernel_size = kernel_data.shape[1]
        time_axis_ms = -np.arange(kernel_size) * time_step_ms

        # 3. 确定颜色范围 (保持不变)
        vmax = np.percentile(np.abs(kernel_data), 99)
        vmin = -vmax

        # --- 2. 检查是否提供了E/I分组 ---
        use_grouping = (exc_indices is not None and inh_indices is not None and
                        len(exc_indices) > 0 and len(inh_indices) > 0)

        if use_grouping:
            # --- 4a. E/I 分组绘图 (原始逻辑) ---
            print("    > 使用 E/I 分组进行绘图。")
            exc_kernels = kernel_data_flipped[exc_indices, :]
            inh_kernels = kernel_data_flipped[inh_indices, :]

            fig = plt.figure(figsize=(15, 12))
            gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

            ax_map_exc = plt.subplot(gs[0, 0])
            ax_map_inh = plt.subplot(gs[0, 1])
            ax_line_exc = plt.subplot(gs[1, 0])
            ax_line_inh = plt.subplot(gs[1, 1])

            # 热图
            ax_map_exc.imshow(exc_kernels, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            ax_map_exc.set_title("Excitatory Kernels (Filter 0)", fontsize=16)
            ax_map_exc.set_ylabel("Synaptic Channel Index", fontsize=14)
            ax_map_exc.set_xticklabels([])

            im = ax_map_inh.imshow(inh_kernels, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            ax_map_inh.set_title("Inhibitory Kernels (Filter 0)", fontsize=16)
            ax_map_inh.set_yticklabels([])
            ax_map_inh.set_xticklabels([])

            # 添加颜色条
            fig.colorbar(im, ax=[ax_map_exc, ax_map_inh], orientation='horizontal', pad=0.05, label="Weight (A.U.)")

            # 时间横截面图
            ax_line_exc.plot(time_axis_ms, exc_kernels.T, 'r', alpha=0.05)
            ax_line_exc.plot(time_axis_ms, exc_kernels.mean(axis=0), 'r', linewidth=2, label='Mean Excitation')
            ax_line_exc.set_xlabel("Time before t0 (ms)", fontsize=14)
            ax_line_exc.set_ylabel("Weight (A.U.)", fontsize=14)
            ax_line_exc.legend()
            ax_line_exc.set_xlim(time_axis_ms.min(), time_axis_ms.max())
            ax_line_exc.spines['top'].set_visible(False)
            ax_line_exc.spines['right'].set_visible(False)

            ax_line_inh.plot(time_axis_ms, inh_kernels.T, 'b', alpha=0.05)
            ax_line_inh.plot(time_axis_ms, inh_kernels.mean(axis=0), 'b', linewidth=2, label='Mean Inhibition')
            ax_line_inh.set_xlabel("Time before t0 (ms)", fontsize=14)
            ax_line_inh.set_yticklabels([])
            ax_line_inh.set_xlim(time_axis_ms.min(), time_axis_ms.max())
            ax_line_inh.legend()
            ax_line_inh.spines['top'].set_visible(False)
            ax_line_inh.spines['right'].set_visible(False)
            ax_line_inh.spines['left'].set_visible(False)

        else:
            # --- 4b. 不分组绘图 (新逻辑) ---
            print("    > 未提供 E/I 分组，将所有通道绘制在一起。")
            all_kernels = kernel_data_flipped

            fig, (ax_map_all, ax_line_all) = plt.subplots(1, 2, figsize=(15, 6))

            # 热图
            im = ax_map_all.imshow(all_kernels, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            ax_map_all.set_title("All Kernels (Filter 0)", fontsize=16)
            ax_map_all.set_ylabel("Synaptic Channel Index", fontsize=14)
            ax_map_all.set_xlabel("Time before t0 (ms)", fontsize=14)

            # 添加颜色条
            fig.colorbar(im, ax=ax_map_all, orientation='vertical', pad=0.03, label="Weight (A.U.)")

            # 时间横截面图
            ax_line_all.plot(time_axis_ms, all_kernels.T, 'k', alpha=0.05)
            ax_line_all.plot(time_axis_ms, all_kernels.mean(axis=0), 'k', linewidth=2, label='Mean (All Channels)')
            ax_line_all.set_xlabel("Time before t0 (ms)", fontsize=14)
            ax_line_all.set_ylabel("Weight (A.U.)", fontsize=14)
            ax_line_all.legend()
            ax_line_all.set_xlim(time_axis_ms.min(), time_axis_ms.max())
            ax_line_all.spines['top'].set_visible(False)
            ax_line_all.spines['right'].set_visible(False)

        # --- 5. 保存 ---
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"  > [警告] 无法生成TCN卷积核可视化图: {e}")
        print("  > (这可能是因为模型不是TCN，或者第一层不是Conv1d)")