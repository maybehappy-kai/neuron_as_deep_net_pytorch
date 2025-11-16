import subprocess
import os
import time
from queue import Queue, Empty
import itertools

# --- 1. 定义要并行使用的 GPU ---
# 根据您的要求，使用 GPU 0, 1, 2, 3
AVAILABLE_GPUS = [0, 1, 2, 3]

# --- 新增：定义每个 GPU 运行的任务数 ---
TASKS_PER_GPU = 4

# --- 2. 定义参数网格 ---
# 这是您想要运行的所有实验组合。
PARAM_GRID = {
    # --- 修改点 1：将 data_subdir 加入网格 ---
    'data_subdir': ['3_4_section_dend', '3_4_section'], # 你刚转换的两个
    # 'data_subdir': ['simple', '1ms', '3_4_section_dend', '3_4_section'], # 包含所有的4个

    'tcn_depth': [1, 2, 3, 4, 5, 6, 7, 8],
    'tcn_width': [32, 64, 128, 256],
    'tcn_kernel_first': [54], # 固定第一层核大小
    'tcn_kernel_rest': [24]   # 固定剩余层核大小
}

# --- 3. 固定的参数 ---
# --- 修改点 1：移除固定的 DATA_SUBDIR，简化 BASE_COMMAND ---
BASE_COMMAND = ['python', 'main_run.py']

# --- 新增：定义日志目录 ---
LOG_DIR = "logs_grid_search"


def main():
    # 创建工作队列和 GPU 队列
    job_queue = Queue()
    gpu_queue = Queue()

    # --- 修复：根据 TASKS_PER_GPU 填充队列 ---
    for gpu_id in AVAILABLE_GPUS:
        for _ in range(TASKS_PER_GPU):
            gpu_queue.put(gpu_id)

    # 生成所有参数组合
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 填充工作队列
    for i, params in enumerate(experiments):
        job_queue.put({'id': i, 'params': params})

    print(f"--- 实验总控脚本 ---")
    print(f"已将 {len(experiments)} 个实验任务加入队列。")
    print(f"将使用 {len(AVAILABLE_GPUS)} 个 GPU, 每个 GPU {TASKS_PER_GPU} 个任务 (共 {len(AVAILABLE_GPUS) * TASKS_PER_GPU} 个并行槽)。")
    print(f"日志将保存在: {LOG_DIR}/")


    # --- 修复：running_processes 现在以 GPU ID 为键，以进程列表为值 ---
    running_processes = {gpu_id: [] for gpu_id in AVAILABLE_GPUS}
    tasks_completed = 0
    total_tasks = len(experiments)

    try:
        while tasks_completed < total_tasks:
            # --- 步骤 A: 检查已完成的进程 (逻辑修改) ---
            finished_gpus = [] # 存储空闲出来的 GPU ID

            for gpu_id, process_list in running_processes.items():
                still_running_on_this_gpu = []
                for (proc, job) in process_list:
                    if proc.poll() is not None: # 进程已结束
                        if proc.returncode == 0:
                            print(f"[GPU {gpu_id}] 任务 {job['id']} (Data={job['params']['data_subdir']}, Depth={job['params']['tcn_depth']}) 已成功完成。")
                        else:
                            print(f"[GPU {gpu_id}] [警告] 任务 {job['id']} (Data={job['params']['data_subdir']}) 失败，返回码: {proc.returncode}。")

                        finished_gpus.append(gpu_id) # 将此 GPU ID 归还到队列
                        tasks_completed += 1
                    else:
                        # 进程仍在运行，将其放回
                        still_running_on_this_gpu.append((proc, job))

                # 更新此 GPU 的正在运行列表
                running_processes[gpu_id] = still_running_on_this_gpu

            # 将所有已完成的 GPU 槽位归还到队列
            for gpu_id in finished_gpus:
                gpu_queue.put(gpu_id)

            # --- 步骤 B: 启动新任务 ---
            while not gpu_queue.empty() and not job_queue.empty():
                try:
                    gpu_id = gpu_queue.get_nowait()
                    job = job_queue.get_nowait()

                    # 构建命令行
                    # 因为 data_subdir 已经在 job['params'] 里了，
                    # 这里的循环会自动把它加上，无需额外处理。
                    cmd = BASE_COMMAND.copy()
                    for key, value in job['params'].items():
                        cmd.append(f"--{key}")
                        cmd.append(str(value))

                    # 设置此子进程专用的 GPU
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                    print(f"[GPU {gpu_id}] 启动任务 {job['id']}/{total_tasks}: {' '.join(cmd[2:])}")

                    # 启动子进程
                    # --- 修改点 2：(推荐) 更新日志文件名以包含数据集 ---
                    log_filename = os.path.join(LOG_DIR, f"logs_job_{job['id']}_Data{job['params']['data_subdir']}_D{job['params']['tcn_depth']}_W{job['params']['tcn_width']}.log")

                    with open(log_filename, 'w') as log_file:
                        # --- 修复：不设置 cwd，让子进程继承父进程的 CWD (即项目根目录) ---
                        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

                    # --- 修复：将进程添加到对应 GPU ID 的列表中 ---
                    running_processes[gpu_id].append((proc, job))

                except Empty:
                    # 队列为空，跳出内部循环
                    break
                except Exception as e:
                    print(f"启动任务时出错: {e}")
                    # 如果启动失败，把 GPU 还回去
                    if 'gpu_id' in locals():
                        gpu_queue.put(gpu_id)

            # --- 步骤 C: 轮询间隔 ---
            time.sleep(10) # 每 10 秒检查一次进程状态

    except KeyboardInterrupt:
        print("\n--- 收到终止信号 ---")
        print("正在终止所有正在运行的子进程...")
        # --- 修复：遍历更新后的字典结构 ---
        for gpu_id, process_list in running_processes.items():
            for (proc, job) in process_list:
                print(f"正在终止 [GPU {gpu_id}] 上的任务 {job['id']}...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("所有子进程已终止。")

    print(f"\n--- 所有 {total_tasks} 个实验任务已完成 ---")

if __name__ == "__main__":
    # --- 修复：创建目录，但不再切换 (chdir) 进去 ---
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    main()