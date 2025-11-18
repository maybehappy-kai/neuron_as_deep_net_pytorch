import os
import json
import sys
import numpy as np
from scipy import signal # 必须导入，原项目函数需要

# 导入项目配置
try:
    import config
    print("成功导入 config.py。")
except ImportError:
    print("[错误] 无法导入 config.py。将使用硬编码的默认值 (-55.0, -67.7)。")
    class MockConfig:
        DATA_CONFIG = {
            "SOMA_VOLTAGE_THRESHOLD_MV": -55.0,
            "Y_SOMA_BIAS": -67.7
        }
    config = MockConfig()

# --------------------------------------------------------------------------
# 1. 从原项目 (integrate_and_fire_figure_replication.py) 复制的参数
# --------------------------------------------------------------------------

# --- 神经元和突触参数 (来自原项目) ---
NUM_EXC = 800
NUM_INH = 200
V_THRESH = -55.0      # (mV) 原项目的 v_threshold
V_RESET = -75.0       # (mV) 原项目的 v_reset
V_REST = V_RESET      # (mV)
TAU_M = 20.0          # (ms) 原项目的 membrane_time_const
W_EXC = 2.5           # (mV) 原项目的 1.0 (权重) * 2 (缩放因子)
W_INH = -2.0          # (mV) 原项目的 -1.0 (权重) * 2 (缩放因子)

# --- 输入生成参数 (来自原项目) ---
inst_rate_sampling_time_interval_options_ms   = [25,30,35,40,50,60,70,80,90,100]
temporal_inst_rate_smoothing_sigma_options_ms = [40,60,80,100]
inst_rate_sampling_time_interval_jitter_range   = 20
temporal_inst_rate_smoothing_sigma_jitter_range = 20

# 这些是原项目I&F示例中使用的特定范围
num_exc_spikes_per_100ms_range = [0, 50]
num_exc_inh_spike_diff_per_100ms_range = [-50, -15]

# --- 2. 新项目的数据集参数 ---
# (保留您的保存格式和文件结构)
DATA_SUBDIR = "data/LIF"
NUM_TIMESTEPS = 200000 # 保留您设定的长时程
TIME_STEP_MS = 1.0 # (原项目也使用 1ms)
NUM_DVT_COPIES = 30 # 保留 DVT 复制逻辑

# --------------------------------------------------------------------------
# 3. 从原项目 (integrate_and_fire_figure_replication.py) 复制的函数
# --------------------------------------------------------------------------
# 注意：此函数与L5PC模拟中的那个不同，这是I&F专用的版本
def generate_input_spike_trains_for_simulation(sim_duration_ms, num_exc_segments, num_inh_segments,
                                               num_exc_spikes_per_100ms_range, num_exc_inh_spike_diff_per_100ms_range):
    """
    这是从 integrate_and_fire_figure_replication.py 复制的输入生成函数。
    它生成非平稳的输入脉冲。
    """

    # 随机采样瞬时速率变化的时间间隔
    keep_inst_rate_const_for_ms = inst_rate_sampling_time_interval_options_ms[np.random.randint(len(inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * inst_rate_sampling_time_interval_jitter_range * np.random.rand() - inst_rate_sampling_time_interval_jitter_range)

    # 随机采样平滑sigma
    temporal_inst_rate_smoothing_sigma = temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - temporal_inst_rate_smoothing_sigma_jitter_range)

    num_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))

    # 创建粗糙的瞬时速率 (单位："total spikes per tree per 100 ms")
    num_ex_spikes_per_100ms   = np.random.uniform(low=num_exc_spikes_per_100ms_range[0], high=num_exc_spikes_per_100ms_range[1], size=(1, num_inst_rate_samples))
    num_inh_spikes_low_range  = np.maximum(0, num_ex_spikes_per_100ms + num_exc_inh_spike_diff_per_100ms_range[0])
    num_inh_spikes_high_range = num_ex_spikes_per_100ms + num_exc_inh_spike_diff_per_100ms_range[1]
    num_inh_spikes_per_100ms  = np.random.uniform(low=num_inh_spikes_low_range, high=num_inh_spikes_high_range, size=(1, num_inst_rate_samples))
    num_inh_spikes_per_100ms[num_inh_spikes_per_100ms < 0] = 0.0001

    # 转换为 "per_segment_per_1ms"
    ex_bas_spike_rate_per_1um_per_1ms   = num_ex_spikes_per_100ms   / (num_exc_segments  * 100.0)
    inh_bas_spike_rate_per_1um_per_1ms  = num_inh_spikes_per_100ms  / (num_inh_segments  * 100.0)

    # 空间 kron (在所有分支上均匀分布)
    ex_spike_rate_per_seg_per_1ms   = np.kron(ex_bas_spike_rate_per_1um_per_1ms  , np.ones((num_exc_segments,1)))
    inh_spike_rate_per_seg_per_1ms  = np.kron(inh_bas_spike_rate_per_1um_per_1ms , np.ones((num_inh_segments,1)))

    # 添加空间乘法随机性
    ex_spike_rate_per_seg_per_1ms  = np.random.uniform(low=0.5, high=1.5, size=ex_spike_rate_per_seg_per_1ms.shape ) * ex_spike_rate_per_seg_per_1ms
    inh_spike_rate_per_seg_per_1ms = np.random.uniform(low=0.5, high=1.5, size=inh_spike_rate_per_seg_per_1ms.shape) * inh_spike_rate_per_seg_per_1ms

    # 时间 kron (如果末尾有多余则裁剪)
    ex_spike_rate_per_seg_per_1ms  = np.kron(ex_spike_rate_per_seg_per_1ms , np.ones((1, keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_seg_per_1ms = np.kron(inh_spike_rate_per_seg_per_1ms, np.ones((1, keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]

    # 根据平滑 sigma 滤波瞬时速率

    # !!!!!!!!!!!!!
    # --- 修正点 ---
    # !!!!!!!!!!!!!
    # 从 signal.gaussian 修正为 signal.windows.gaussian
    smoothing_window = signal.windows.gaussian(int(1.0 + 7 * temporal_inst_rate_smoothing_sigma), std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]

    smoothing_window /= smoothing_window.sum()
    seg_inst_rate_ex_smoothed  = signal.convolve(ex_spike_rate_per_seg_per_1ms,  smoothing_window, mode='same')
    seg_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_seg_per_1ms, smoothing_window, mode='same')

    # 采样瞬时脉冲概率，然后采样实际脉冲
    ex_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_ex_smoothed)
    ex_spikes_bin      = np.random.rand(ex_inst_spike_prob.shape[0], ex_inst_spike_prob.shape[1]) < ex_inst_spike_prob

    inh_inst_spike_prob = np.random.exponential(scale=seg_inst_rate_inh_smoothed)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob

    all_spikes_bin = np.vstack((ex_spikes_bin, inh_spikes_bin))

    return all_spikes_bin


# --------------------------------------------------------------------------
# 4. 修改后的主逻辑
# --------------------------------------------------------------------------

# (我也将 config.py 中的 V_THRESH 和 V_RESET 替换回了原项目的参数，以确保发放)
LIF_V_THRESH = V_THRESH
LIF_V_RESET = V_RESET
LIF_V_REST = V_RESET

print(f"--- 正在创建LIF数据集 (使用原项目参数) ---")
print(f"  目标目录: {DATA_SUBDIR}")
print(f"  LIF 参数 (来自原项目):")
print(f"    V_Threshold: {LIF_V_THRESH} mV")
print(f"    V_Reset/Rest: {LIF_V_RESET} mV")
print(f"    TAU_M: {TAU_M} ms")
print(f"    W_Exc: {W_EXC} mV, W_Inh: {W_INH} mV")
print(f"  输入通道 (来自原项目): {NUM_EXC} Exc, {NUM_INH} Inh")

os.makedirs(DATA_SUBDIR, exist_ok=True)

# --- 3. 生成输入数据 (spike_matrix.npy) ---
# **替换为调用原项目的函数**
print("  > 正在生成非平稳输入脉冲 (来自原项目)...")
total_channels = NUM_EXC + NUM_INH
time_axis = np.arange(0, NUM_TIMESTEPS * TIME_STEP_MS, TIME_STEP_MS).reshape(-1, 1)

# 调用复制的函数
# 注意：原函数返回 (C, T)，我们需要 (T, C)
input_spikes_bin = generate_input_spike_trains_for_simulation(
    sim_duration_ms=NUM_TIMESTEPS,
    num_exc_segments=NUM_EXC,
    num_inh_segments=NUM_INH,
    num_exc_spikes_per_100ms_range=num_exc_spikes_per_100ms_range,
    num_exc_inh_spike_diff_per_100ms_range=num_exc_inh_spike_diff_per_100ms_range
)
input_spikes = input_spikes_bin.T # (T, C)

# (保留您的保存逻辑)
input_data = np.hstack((time_axis, input_spikes.astype(float)))
input_path = os.path.join(DATA_SUBDIR, "spike_matrix.npy")
np.save(input_path, input_data)
print(f"  [成功] 已保存: {input_path} (Shape: {input_data.shape})")

# --- 4. 运行LIF模拟以生成 (V_soma.npy) ---
# **保留您的LIF模拟器，但使用原项目的参数**
v_soma = np.full(NUM_TIMESTEPS, LIF_V_REST)
v = LIF_V_REST
dt_over_tau = TIME_STEP_MS / TAU_M

print("  > 正在运行LIF模拟...")
for t in range(1, NUM_TIMESTEPS):
    # 计算漏电项 (Leaky term)
    dv = (LIF_V_REST - v) * dt_over_tau
    v += dv

    # 计算突触输入 (使用 t-1 的脉冲来影响 t)
    synaptic_input = (input_spikes[t-1, :NUM_EXC].sum() * W_EXC) + \
                     (input_spikes[t-1, NUM_EXC:].sum() * W_INH)
    v += synaptic_input

    # 钳制电压，使其不低于静息电位
    v = v

    # 脉冲处理逻辑 (保留您的方式，这对新项目的数据加载器是兼容的)
    if v >= LIF_V_THRESH:
        # 使用一个新项目 data_loader 预期之外的值来标记脉冲
        # data_loader.py 会使用 -25mV 寻找局部最大值
        # 我们这里使用 0.0mV 标记，这符合原项目 L5PC 数据的做法
        v_soma[t] = 0.0
        v = LIF_V_RESET     # 在下一个时间步复位
    else:
        v_soma[t] = v

print("  > LIF模拟完成。")

# --- 5. 验证脉冲计数 (保留) ---
# **更新：我们检查 0.0mV**
v_soma_reshaped = v_soma.reshape(-1, 1)
spike_count = np.sum(v_soma_reshaped == 0.0)
print(f"  > [验证] 总共生成了 {spike_count} 个脉冲 (0mV点)。")
if spike_count == 0:
    print("  > [警告] 仍然没有脉冲！参数可能仍需调整。")
else:
    print(f"  > 平均发放率: {spike_count / (NUM_TIMESTEPS / 1000.0):.3f} Hz")


# (保留您的DVT复制和保存逻辑)
dvt_data = np.tile(v_soma_reshaped, (1, NUM_DVT_COPIES))
output_data = np.hstack((time_axis, v_soma_reshaped, dvt_data))
output_path = os.path.join(DATA_SUBDIR, "V_soma.npy")
np.save(output_path, output_data)
print(f"  [成功] 已保存: {output_path} (Shape: {output_data.shape})")

# --- 6. 生成通道元数据 (channel_info.json) ---
# (保留您的逻辑，但使用更新后的通道数)
channel_info = {
    "exc_indices": list(range(NUM_EXC)),
    "inh_indices": list(range(NUM_EXC, total_channels))
}

json_path = os.path.join(DATA_SUBDIR, "channel_info.json")
with open(json_path, 'w') as f:
    json.dump(channel_info, f, indent=4)
print(f"  [成功] 已保存: {json_path}")
print("--- 数据集生成完毕 ---")