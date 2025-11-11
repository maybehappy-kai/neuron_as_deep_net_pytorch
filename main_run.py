"""
项目主运行脚本
运行此文件将自动执行 训练 -> 验证 -> 测试 -> 可视化 的完整流程。
"""
import torch
import torch.optim as optim
import os
import time
import datetime
import json
import numpy as np
import pandas as pd
import shutil
import argparse

# 导入我们的模块
import config # 加载配置
import models # 从 models/__init__.py 导入
from utils import data_loader, training_loop, evaluation, visualization

# --- 函数：解析命令行参数 ---
def parse_arguments():
    """
    解析命令行参数以覆盖 config.py 中的默认值
    """
    parser = argparse.ArgumentParser(description="运行神经元TCN模型训练")

    # --- 1. 数据集路径 ---
    parser.add_argument(
        '--data_subdir',
        type=str,
        default=None,
        help="指定 'data/' 文件夹下的数据集子目录 (例如: 'dataset_A')")

    # --- 2. 模型名称 ---
    parser.add_argument(
        '--model_name',
        type=str,
        default=config.MODEL_CONFIG["MODEL_NAME"],
        help="要使用的模型名称 (来自 models 文件夹)")

    # --- 3. TCN 架构参数 ---
    parser.add_argument(
        '--tcn_depth',
        type=int,
        default=config.MODEL_CONFIG["NETWORK_DEPTH"],
        help="TCN 层数 (NETWORK_DEPTH)")

    parser.add_argument(
        '--tcn_width',
        type=int,
        default=config.MODEL_CONFIG["NUM_FILTERS"],
        help="TCN 滤波器数量 (NUM_FILTERS)")

    # 从 config.py 中获取默认的卷积核大小
    default_kernel_first = config.MODEL_CONFIG["FILTER_SIZES"][0] if len(config.MODEL_CONFIG["FILTER_SIZES"]) > 0 else 54
    default_kernel_rest = config.MODEL_CONFIG["FILTER_SIZES"][1] if len(config.MODEL_CONFIG["FILTER_SIZES"]) > 1 else 24

    parser.add_argument(
        '--tcn_kernel_first',
        type=int,
        default=default_kernel_first,
        help="TCN 第一层卷积核大小")

    parser.add_argument(
        '--tcn_kernel_rest',
        type=int,
        default=default_kernel_rest,
        help="TCN 后续层卷积核大小")

    return parser.parse_args()

# --- 函数：用参数更新配置 ---
def update_config_with_args(cfg_module, args):
    """
    使用解析的 args 更新 cfg_module (config) 模块的属性
    """

    # 1. 更新数据集路径
    if args.data_subdir:
        print(f"  > [配置覆盖] 使用数据集子目录: {args.data_subdir}")
        input_filename = "spike_matrix.npy"
        target_filename = "V_soma.npy"

        # 构建新路径
        cfg_module.DATA_CONFIG["INPUT_FILE_PATH"] = os.path.join("data", args.data_subdir, input_filename)
        cfg_module.DATA_CONFIG["TARGET_FILE_PATH"] = os.path.join("data", args.data_subdir, target_filename)
        print(f"    - 输入路径: {cfg_module.DATA_CONFIG['INPUT_FILE_PATH']}")
        print(f"    - 目标路径: {cfg_module.DATA_CONFIG['TARGET_FILE_PATH']}")

    # 2. 更新模型名称
    if args.model_name != cfg_module.MODEL_CONFIG["MODEL_NAME"]:
        print(f"  > [配置覆盖] 使用模型: {args.model_name}")
        cfg_module.MODEL_CONFIG["MODEL_NAME"] = args.model_name

    # 3. 更新TCN架构
    # 检查是否有任何TCN参数被修改
    tcn_params_changed = (
            args.tcn_depth != cfg_module.MODEL_CONFIG["NETWORK_DEPTH"] or
            args.tcn_width != cfg_module.MODEL_CONFIG["NUM_FILTERS"] or
            args.tcn_kernel_first != (cfg_module.MODEL_CONFIG["FILTER_SIZES"][0] if len(cfg_module.MODEL_CONFIG["FILTER_SIZES"]) > 0 else -1) or
            args.tcn_kernel_rest != (cfg_module.MODEL_CONFIG["FILTER_SIZES"][1] if len(cfg_module.MODEL_CONFIG["FILTER_SIZES"]) > 1 else -1)
    )

    if tcn_params_changed:
        print(f"  > [配置覆盖] TCN 架构更新:")
        cfg_module.MODEL_CONFIG["NETWORK_DEPTH"] = args.tcn_depth
        cfg_module.MODEL_CONFIG["NUM_FILTERS"] = args.tcn_width

        # 动态构建 FILTER_SIZES 列表
        if args.tcn_depth > 0:
            new_filter_sizes = [args.tcn_kernel_first]
            if args.tcn_depth > 1:
                new_filter_sizes.extend([args.tcn_kernel_rest] * (args.tcn_depth - 1))
            cfg_module.MODEL_CONFIG["FILTER_SIZES"] = new_filter_sizes

            print(f"    - 深度 (NETWORK_DEPTH): {args.tcn_depth}")
            print(f"    - 宽度 (NUM_FILTERS): {args.tcn_width}")
            print(f"    - 滤波器大小 (FILTER_SIZES): {new_filter_sizes}")
        else:
            print("  > [警告] TCN 深度必须 > 0。FILTER_SIZES 未被修改。")
            cfg_module.MODEL_CONFIG["FILTER_SIZES"] = [] # 或者保持原样

        # 确保 NETWORK_DEPTH 和 FILTER_SIZES 长度一致
        if len(cfg_module.MODEL_CONFIG["FILTER_SIZES"]) != cfg_module.MODEL_CONFIG["NETWORK_DEPTH"]:
            print(f"  > [警告] 深度 ({cfg_module.MODEL_CONFIG['NETWORK_DEPTH']}) 和 滤波器列表长度 ({len(cfg_module.MODEL_CONFIG['FILTER_SIZES'])}) 不匹配。")
            # 以 args.tcn_depth (即 NETWORK_DEPTH) 为准
            cfg_module.MODEL_CONFIG["NETWORK_DEPTH"] = len(cfg_module.MODEL_CONFIG["FILTER_SIZES"])
            print(f"    - 自动修正 NETWORK_DEPTH 为 {cfg_module.MODEL_CONFIG['NETWORK_DEPTH']}")


# --- 函数：设置实验目录 (已更新命名规则) ---
def setup_experiment_directory(cfg):
    """
    创建本次运行的唯一输出目录，并复制配置文件。
    """
    # 1. 创建基于时间和配置的唯一运行名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从配置中提取关键参数用于命名
    model_name = cfg.MODEL_CONFIG["MODEL_NAME"]
    lr_initial = cfg.TRAIN_CONFIG["LR_POLICY"]["INITIAL_LR"]
    window_size = cfg.MODEL_CONFIG["INPUT_WINDOW_SIZE"]

    # 提取数据子目录
    data_subdir = os.path.basename(os.path.dirname(cfg.DATA_CONFIG["INPUT_FILE_PATH"]))
    if data_subdir == "data": # 如果没有子目录
        data_subdir = "default"

    # --- 核心修改：根据模型名称构建后缀 ---
    run_name_parts = [
        timestamp,
        f"data-{data_subdir}",
        f"model-{model_name}",
        f"lr-{lr_initial}",
        f"win-{window_size}"
    ]

    if model_name == "tcn":
        # 如果是 TCN，添加 TCN 特定的架构参数
        depth = cfg.MODEL_CONFIG['NETWORK_DEPTH']
        width = cfg.MODEL_CONFIG['NUM_FILTERS']

        # 从 FILTER_SIZES 列表中安全地获取 k1 和 kr
        filter_sizes = cfg.MODEL_CONFIG.get("FILTER_SIZES", [])
        k1 = filter_sizes[0] if len(filter_sizes) > 0 else "NA"
        # 如果只有1层(depth=1)，后续核大小(kr)就等于第一层(k1)
        kr = filter_sizes[1] if len(filter_sizes) > 1 else k1

        run_name_parts.append(f"d-{depth}")
        run_name_parts.append(f"w-{width}")
        run_name_parts.append(f"k1-{k1}")
        run_name_parts.append(f"kr-{kr}")
    else:
        if "NETWORK_DEPTH" in cfg.MODEL_CONFIG:
            run_name_parts.append(f"d-{cfg.MODEL_CONFIG['NETWORK_DEPTH']}")
        if "NUM_FILTERS" in cfg.MODEL_CONFIG:
            run_name_parts.append(f"w-{cfg.MODEL_CONFIG['NUM_FILTERS']}")

    run_name = "_".join(run_name_parts)
    # --- 修改结束 ---

    output_dir = os.path.join("results", run_name)
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    # 2. 复制 config.py 快照到该目录
    shutil.copyfile("config.py", os.path.join(output_dir, "config_snapshot.py"))

    # 3. 保存一份包含所有覆盖参数的json
    all_configs = {
        "DATA_CONFIG": cfg.DATA_CONFIG,
        "MODEL_CONFIG": cfg.MODEL_CONFIG,
        "TRAIN_CONFIG": cfg.TRAIN_CONFIG,
        "EVAL_CONFIG": cfg.EVAL_CONFIG,
    }
    with open(os.path.join(output_dir, "effective_config.json"), 'w') as f:
        json.dump(all_configs, f, indent=4)


    print(f"--- 实验已创建 ---")
    print(f"  > 运行名称: {run_name}")
    print(f"  > 输出目录: {output_dir}")

    return output_dir, plots_dir

def get_model(data_info, cfg):
    """
    根据 config.py 动态实例化模型。
    """
    model_name = cfg.MODEL_CONFIG["MODEL_NAME"]

    if model_name == "tcn":
        model = models.TCN(
            input_channels=data_info["input_channels"],
            num_dvt_channels=data_info["dvt_channels"],
            cfg=cfg
        )
    # elif model_name == "transformer":
    #     model = models.TransformerModel(...)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model

def get_optimizer(model, cfg):
    """
    设置优化器，并在此处应用L2正则化 (weight_decay)。

    --- 修改 ---
    根据最佳实践，将参数分为两组：
    1. 应用 Weight Decay (L2 正则化) 的参数 (如 Conv 层的 weights)
    2. 不应用 Weight Decay 的参数 (如 bias 和 BatchNorm 层的参数)
    """
    lr = cfg.TRAIN_CONFIG["LR_POLICY"]["INITIAL_LR"] # 从新的 LR_POLICY 获取
    weight_decay = cfg.MODEL_CONFIG["L2_REGULARIZATION"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 检查是否为 Conv 层的 weight
        # (在我们的模型中，主要是 'network.*.conv.weight' 和 'head_*.weight')
        if 'conv.weight' in name or 'head' in name and 'weight' in name:
            # 检查是否为 nn.Conv1d 的权重 (以防未来添加 nn.Linear)
            if param.dim() > 1: # Conv1d 的 weight 是 3D, Linear 的 weight 是 2D
                decay_params.append(param)
            else:
                # (可能是 Linear 层的 weight，但在此 TCN 模型中主要是 Conv1d)
                decay_params.append(param)

        # 检查是否为 bias 或 BatchNorm 参数
        elif 'bias' in name:
            no_decay_params.append(param)
        elif 'bn' in name: # 对应 CausalConv1dLayer 中的 self.bn
            no_decay_params.append(param)
        else:
            # 其他参数（以防万一），默认不衰减
            no_decay_params.append(param)

    # 创建两个参数组
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    print("  > 优化器已设置：")
    print(f"    - {len(decay_params)} 个参数组应用 L2 (wd={weight_decay})")
    print(f"    - {len(no_decay_params)} 个参数组不应用 L2 (wd=0.0)")

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def save_test_results(results, output_dir):
    """
    将最终测试指标保存为 JSON 文件。
    """
    results_to_save = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            results_to_save[key] = value
        elif isinstance(value, np.ndarray):
            results_to_save[key] = value.tolist()

    if 'roc_curve_data' in results_to_save:
        del results_to_save['roc_curve_data']

    save_path = os.path.join(output_dir, "test_results.json")
    with open(save_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    print(f"  > 最终测试指标已保存到 {save_path}")

# --- 修改此函数 ---
def run_visualizations(model, test_data, test_results, data_info, cfg, plots_dir, full_data=None):
    """
    调用 visualization.py 中的所有绘图函数。
    """
    print("--- 步骤 8: 生成可视化图表 ---")

    # 1. 准备数据 (从GPU转到CPU/Numpy)
    X_test_full, Y_test_dict_full = test_data
    soma_bias = data_info["scalers"]["soma_bias"]
    test_predictions = evaluation.predict_full_sequence(model, X_test_full, cfg)

    # 准备Numpy数组
    y_true_soma_cpu = Y_test_dict_full['soma'].cpu().numpy().ravel() + soma_bias
    y_pred_soma_cpu = test_predictions['soma'].cpu().numpy().ravel() + soma_bias
    y_true_spike_cpu = Y_test_dict_full['spike'].cpu().numpy().ravel()
    y_pred_prob_cpu = torch.sigmoid(test_predictions['spike_logits']).cpu().numpy().ravel()
    optimal_threshold = test_results['spike_optimal_threshold']
    y_pred_spike_binary_cpu = (y_pred_prob_cpu >= optimal_threshold).astype(int)
    time_step_ms = data_info["time_step_ms"]
    total_steps = len(y_true_soma_cpu)
    time_axis_s = np.arange(total_steps) * (time_step_ms / 1000.0)

    # 2. 绘制评估指标图 (ROC, Scatter, Cross-Corr)
    visualization.plot_evaluation_metrics(
        y_true_soma_cpu, y_pred_soma_cpu,
        y_true_spike_cpu, y_pred_prob_cpu,
        optimal_threshold,
        test_results['roc_curve_data'],
        time_step_ms,
        save_path=os.path.join(plots_dir, "evaluation_metrics.png")
    )
    print(f"  > 已保存: evaluation_metrics.png")

    # 3. 绘制轨迹对比图
    # 3a. 完整测试集
    visualization.plot_prediction_trace(
        y_true_soma_cpu, y_pred_soma_cpu,
        y_true_spike_cpu, y_pred_spike_binary_cpu,
        time_axis_s,
        title="Prediction vs. Ground Truth (Full Test Set)",
        save_path=os.path.join(plots_dir, "prediction_trace_test_set.png")
    )
    print(f"  > 已保存: prediction_trace_test_set.png")

    # --- 3b. 6秒切片 (新逻辑：必须包含一个脉冲) ---
    slice_seconds = cfg.EVAL_CONFIG["VISUALIZATION_SLICE_SECONDS"]
    slice_steps = int(slice_seconds * 1000 / time_step_ms)

    start_idx = 0 # 默认

    # 1. 查找所有脉冲
    spike_indices = np.where(y_true_spike_cpu > 0.5)[0]

    if total_steps < slice_steps:
        print(f"  > [警告] 测试集总步数 ({total_steps}) 不足以切片 {slice_seconds}s ({slice_steps} 步)。")
        print(f"  > 已跳过 6s 切片可视化。")
        slice_steps = 0 # 标记为跳过

    elif len(spike_indices) == 0:
        print(f"  > [警告] 6s 切片：在测试集中未找到脉冲。回退到从 25% 处开始切片。")
        potential_start_idx = total_steps // 4
        if potential_start_idx + slice_steps <= total_steps:
            start_idx = potential_start_idx
        # else: start_idx 保持为 0

    else:
        # 2. 找到了脉冲，选择一个并居中
        chosen_spike_idx = np.random.choice(spike_indices)
        start_idx = max(0, chosen_spike_idx - (slice_steps // 2))

        # 3. 确保切片不会越过结尾
        if start_idx + slice_steps > total_steps:
            start_idx = total_steps - slice_steps

        print(f"  > 6s 切片：在第 {chosen_spike_idx} 步找到脉冲。切片从 {start_idx} 步开始。")

    # 4. 仅在 slice_steps > 0 时绘图
    if slice_steps > 0:
        end_idx = start_idx + slice_steps

        # 缩放范围逻辑
        zoom_start_idx_rel = slice_steps // 3
        zoom_end_idx_rel = 2 * slice_steps // 3
        zoom_start_s = time_axis_s[start_idx + zoom_start_idx_rel]
        zoom_end_s = time_axis_s[start_idx + zoom_end_idx_rel]
        zoom_range_s = (zoom_start_s, zoom_end_s)

        visualization.plot_prediction_trace(
            y_true_soma_cpu[start_idx:end_idx],
            y_pred_soma_cpu[start_idx:end_idx],
            y_true_spike_cpu[start_idx:end_idx],
            y_pred_spike_binary_cpu[start_idx:end_idx],
            time_axis_s[start_idx:end_idx],
            title=f"Prediction vs. Ground Truth ({slice_seconds}s Slice)",
            save_path=os.path.join(plots_dir, "prediction_trace_6s_slice.png"),
            zoom_range_s=zoom_range_s,
            time_step_ms=time_step_ms
        )
        print(f"  > 已保存: prediction_trace_6s_slice.png")
    # --- 修改结束 ---

    # --- 4. 绘制TCN卷积核 (如果模型是TCN) ---
    if cfg.MODEL_CONFIG["MODEL_NAME"] == "tcn":
        visualization.plot_tcn_kernels(
            model,
            exc_indices=data_info["exc_indices"],
            inh_indices=data_info["inh_indices"],
            time_step_ms=time_step_ms,
            save_path=os.path.join(plots_dir, "tcn_kernels.png")
        )
        print(f"  > 已保存: tcn_kernels.png")

    # --- 5. 绘制完整数据集轨迹 (新) ---
    if full_data:
        print("  > 正在为完整数据集(Train+Val+Test)生成预测...")
        X_full_all, Y_full_all_dict = full_data

        # 运行预测
        full_predictions = evaluation.predict_full_sequence(model, X_full_all, cfg)

        # 准备Numpy数组 (注意：Y中的soma已减去偏置，这里加回来)
        full_true_soma_cpu = Y_full_all_dict['soma'].cpu().numpy().ravel() + soma_bias
        full_pred_soma_cpu = full_predictions['soma'].cpu().numpy().ravel() + soma_bias
        full_true_spike_cpu = Y_full_all_dict['spike'].cpu().numpy().ravel()

        full_pred_prob_cpu = torch.sigmoid(full_predictions['spike_logits']).cpu().numpy().ravel()
        # 注意: 我们使用 'test_results' 中的阈值，因为它是基于模型性能计算的
        full_pred_spike_binary_cpu = (full_pred_prob_cpu >= optimal_threshold).astype(int)

        full_total_steps = len(full_true_soma_cpu)
        full_time_axis_s = np.arange(full_total_steps) * (time_step_ms / 1000.0)

        visualization.plot_prediction_trace(
            full_true_soma_cpu, full_pred_soma_cpu,
            full_true_spike_cpu, full_pred_spike_binary_cpu,
            full_time_axis_s,
            title="Prediction vs. Ground Truth (Full Dataset: Train+Val+Test)",
            save_path=os.path.join(plots_dir, "prediction_trace_full_dataset.png")
        )
        print(f"  > 已保存: prediction_trace_full_dataset.png")

def main():
    start_time = time.time()

    # --- 0. 解析参数并更新配置 ---
    print("--- 步骤 0: 解析命令行参数 ---")
    args = parse_arguments()
    update_config_with_args(config, args)

    # --- 1. 设置实验 ---
    output_dir, plots_dir = setup_experiment_directory(config)

    # --- 2. 加载数据 ---
    train_loader, val_data, test_data, data_info = data_loader.get_data_loaders(config, output_dir)

    # --- 3. 初始化模型和优化器 ---
    print(f"--- 步骤 4: 初始化模型 ({config.MODEL_CONFIG['MODEL_NAME']}) ---")
    model = get_model(data_info, config).to(config.DEVICE)
    optimizer = get_optimizer(model, config)

    # 打印模型结构
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  > 模型总参数量: {total_params:,}")

    # --- 4. 训练模型 ---
    log_df = training_loop.run_training_loop(
        model, optimizer, train_loader, val_data, config, output_dir
    )

    # 保存训练日志
    log_df.to_csv(os.path.join(output_dir, "train_log.csv"), index=False)
    visualization.plot_learning_curves(log_df, os.path.join(plots_dir, "learning_curves.png"))
    print("  > 训练日志和学习曲线已保存。")

    # --- 5. 加载最佳模型并测试 ---
    print("--- 步骤 6: 加载最佳模型并在测试集上运行评估 ---")
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("  > [错误] 未找到 'best_model.pth'。评估已跳过。")
        return

    model.load_state_dict(torch.load(best_model_path))
    test_X_full, test_Y_dict_full = test_data

    # 运行预测
    test_predictions = evaluation.predict_full_sequence(model, test_X_full, config)

    # 计算指标
    final_loss_weights = config.TRAIN_CONFIG["LOSS_WEIGHT_VALUES"][-1]
    test_results = evaluation.evaluate_metrics(test_predictions, test_Y_dict_full, final_loss_weights, config)

    print("--- 步骤 7: 最终测试集结果 ---")
    print(f"  > 总损失: {test_results['loss_total']:.6f}")
    print(f"  > 脉冲 AUC: {test_results['spike_AUC']:.5f}")
    print(f"  > Soma VE: {test_results['soma_VE']:.5f}")
    print(f"  > 脉冲 F1 (最佳阈值): {test_results['spike_F1']:.5f}")
    print(f"    > 最佳阈值: {test_results['spike_optimal_threshold']:.4f}")
    print(f"    > Precision: {test_results['spike_Precision']:.5f}")
    print(f"    > Recall:    {test_results['spike_Recall']:.5f}")

    # 保存数字结果
    save_test_results(test_results, output_dir)

    # --- 6. 生成所有可视化图表 ---
    # --- START: 新增代码，用于拼接完整数据集 ---
    print("--- 步骤 7b: 准备完整数据集以进行最终可视化 ---")
    # 1. 从 train_loader 中提取 (T, C) 和 (T, 1) 的 CPU 张量
    X_train_tensor_cpu = train_loader.dataset.X
    Y_soma_train_tensor_cpu = train_loader.dataset.Y_soma
    Y_spike_train_tensor_cpu = train_loader.dataset.Y_spike

    # 2. 从 val_data 和 test_data 中获取 GPU 张量
    X_val_tensor, Y_val_dict = val_data
    # test_data 已经在前面被解包 (test_X_full, test_Y_dict_full)

    # 3. 将训练数据移动到 GPU
    X_train_tensor_gpu = X_train_tensor_cpu.to(config.DEVICE)
    Y_soma_train_tensor_gpu = Y_soma_train_tensor_cpu.to(config.DEVICE)
    Y_spike_train_tensor_gpu = Y_spike_train_tensor_cpu.to(config.DEVICE)

    # 4. 拼接所有数据 (T, C)
    X_full_dataset = torch.cat((X_train_tensor_gpu, X_val_tensor, test_X_full), dim=0)
    # 拼接 (T, 1)
    Y_soma_full_dataset = torch.cat((Y_soma_train_tensor_gpu, Y_val_dict['soma'], test_Y_dict_full['soma']), dim=0)
    Y_spike_full_dataset = torch.cat((Y_spike_train_tensor_gpu, Y_val_dict['spike'], test_Y_dict_full['spike']), dim=0)

    Y_full_dict = {
        'soma': Y_soma_full_dataset,
        'spike': Y_spike_full_dataset
    }
    # (如果 DVT 存在，也需要拼接 DVT)
    if data_info["dvt_channels"] > 0:
        Y_dvt_train_tensor_cpu = train_loader.dataset.Y_dvt
        Y_dvt_train_tensor_gpu = Y_dvt_train_tensor_cpu.to(config.DEVICE)
        Y_dvt_full_dataset = torch.cat((Y_dvt_train_tensor_gpu, Y_val_dict['dvt_pca'], test_Y_dict_full['dvt_pca']), dim=0)
        Y_full_dict['dvt_pca'] = Y_dvt_full_dataset

    full_data = (X_full_dataset, Y_full_dict)
    print(f"  > 完整数据集已拼接 (总步数: {X_full_dataset.shape[0]})")
    # --- END: 新增代码 ---

    # 修改第 470 行的函数调用
    run_visualizations(model, test_data, test_results, data_info, config, plots_dir, full_data)

    end_time = time.time()
    print(f"\n--- 完整运行已结束 ---")
    print(f"  > 总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"  > 所有结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed(42)

    main()