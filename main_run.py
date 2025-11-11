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

# 导入我们的模块
import config # 加载配置
import models # 从 models/__init__.py 导入
from utils import data_loader, training_loop, evaluation, visualization

def setup_experiment_directory(cfg):
    """
    创建本次运行的唯一输出目录，并复制配置文件。
    """
    # 1. 创建基于时间和配置的唯一运行名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从配置中提取关键参数用于命名
    model_name = cfg.MODEL_CONFIG["MODEL_NAME"]
    lr_initial = cfg.TRAIN_CONFIG["LEARNING_RATE_SCHEDULE"][0]
    window_size = cfg.MODEL_CONFIG["INPUT_WINDOW_SIZE"]

    run_name = f"{timestamp}_model-{model_name}_lr-{lr_initial}_win-{window_size}"

    output_dir = os.path.join("results", run_name)
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    # 2. 复制 config.py 快照到该目录
    shutil.copyfile("config.py", os.path.join(output_dir, "config_snapshot.py"))

    print(f"--- 实验已创建 ---")
    print(f"  > 运行名称: {run_name}")
    print(f"  > 输出目录: {output_dir}")

    return output_dir, plots_dir

def get_model(data_info, cfg):
    """
    根据 config.py 动态实例化模型。
    """
    model_name = cfg.MODEL_CONFIG["MODEL_NAME"]

    # --- 模型扩展点 ---
    # 当您在 models/ 文件夹中添加新模型时，
    # 只需要在这里添加一个 'elif' 分支。

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
    """
    lr = cfg.TRAIN_CONFIG["LEARNING_RATE_SCHEDULE"][0] # 获取初始学习率

    # 复现原项目的 L2 正则化
    # PyTorch 的 L2 正则化通过 'weight_decay' 参数传递给优化器
    weight_decay = cfg.MODEL_CONFIG["L2_REGULARIZATION"]

    # 复现原项目的 Nadam 优化器
    # PyTorch 中 NAdam 与 AdamW 类似，但 AdamW 在现代实践中更受推荐
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer

def save_test_results(results, output_dir):
    """
    将最终测试指标保存为 JSON 文件。
    """
    # 清理结果字典，移除大数据对象
    results_to_save = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            results_to_save[key] = value
        elif isinstance(value, np.ndarray):
            # 将 roc_curve_data 转换为列表以便序列化
            results_to_save[key] = value.tolist()

    # 单独移除 roc_curve_data，因为它太大了
    if 'roc_curve_data' in results_to_save:
        del results_to_save['roc_curve_data']

    save_path = os.path.join(output_dir, "test_results.json")
    with open(save_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    print(f"  > 最终测试指标已保存到 {save_path}")

def run_visualizations(model, test_data, test_results, data_info, cfg, plots_dir):
    """
    调用 visualization.py 中的所有绘图函数。
    """
    print("--- 步骤 8: 生成可视化图表 ---")

    # 1. 准备数据 (从GPU转到CPU/Numpy)
    X_test_full, Y_test_dict_full = test_data

    # 使用保存的偏置逆转Soma电压的中心化
    soma_bias = data_info["scalers"]["soma_bias"]

    # 运行一次完整的预测
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
        test_results['roc_curve_data'], # 从测试结果中传递roc数据
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

    # 3b. 6秒切片
    slice_steps = int(cfg.EVAL_CONFIG["VISUALIZATION_SLICE_SECONDS"] * 1000 / time_step_ms)
    start_idx = total_steps // 4 # 从 25% 处开始切片
    end_idx = start_idx + slice_steps

    visualization.plot_prediction_trace(
        y_true_soma_cpu[start_idx:end_idx],
        y_pred_soma_cpu[start_idx:end_idx],
        y_true_spike_cpu[start_idx:end_idx],
        y_pred_spike_binary_cpu[start_idx:end_idx],
        time_axis_s[start_idx:end_idx],
        title=f"Prediction vs. Ground Truth ({cfg.EVAL_CONFIG['VISUALIZATION_SLICE_SECONDS']}s Slice)",
        save_path=os.path.join(plots_dir, "prediction_trace_6s_slice.png"),
        zoom_range_s=(time_axis_s[start_idx + slice_steps // 3], time_axis_s[start_idx + 2 * slice_steps // 3]),
        time_step_ms=time_step_ms
    )
    print(f"  > 已保存: prediction_trace_6s_slice.png")

    # 4. 绘制TCN卷积核 (如果模型是TCN)
    if cfg.MODEL_CONFIG["MODEL_NAME"] == "tcn":
        visualization.plot_tcn_kernels(
            model,
            num_exc_channels=data_info["input_channels"] // 2, # 假设一半是兴奋性
            time_step_ms=time_step_ms,
            save_path=os.path.join(plots_dir, "tcn_kernels.png")
        )
        print(f"  > 已保存: tcn_kernels.png")

def main():
    start_time = time.time()

    # --- 1. 设置实验 ---
    output_dir, plots_dir = setup_experiment_directory(config)

    # --- 2. 加载数据 ---
    # data_info 包含: time_step_ms, input_channels, dvt_channels, ...
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
    # 使用最后一个epoch的损失权重进行最终评估
    final_loss_weights = config.TRAIN_CONFIG["LOSS_WEIGHTS_SCHEDULE"][-1]
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
    run_visualizations(model, test_data, test_results, data_info, config, plots_dir)

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