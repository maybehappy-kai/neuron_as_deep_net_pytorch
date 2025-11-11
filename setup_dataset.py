import os
import json

def create_channel_info():
    """
    引导用户输入数据集信息，并创建 channel_info.json 文件。
    """
    print("--- 数据集通道信息设置向导 ---")

    # 1. 获取数据集子文件夹
    data_subdir = input("请输入 data/ 文件夹中的数据集子目录名称 (例如 'dataset_A'): ")
    target_dir = os.path.join("data", data_subdir)

    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        print(f"\n[错误] 目录未找到: {target_dir}")
        print("请确保该子目录已存在于 'data/' 文件夹下。")
        return

    print(f"\n目标目录: {target_dir}")

    # 2. 获取通道数量
    try:
        num_exc = int(input("请输入兴奋性(Excitatory)通道的数量: "))
        num_inh = int(input("请输入抑制性(Inhibitory)通道的数量: "))

        if num_exc < 0 or num_inh < 0:
            raise ValueError("通道数不能为负。")

    except ValueError as e:
        print(f"\n[错误] 输入无效: {e}。请输入一个有效的非负整数。")
        return

    # 3. 生成通道索引列表
    # 索引基于 X_raw[:, 1:]，所以它们从 0 开始。
    # 假设兴奋性通道在前，抑制性通道在后。

    exc_indices = list(range(0, num_exc))
    inh_indices = list(range(num_exc, num_exc + num_inh))

    total_channels = num_exc + num_inh
    print(f"\n总共 {total_channels} 个数据通道。")
    print(f"  - 兴奋性通道索引: 0 到 {num_exc - 1}")
    print(f"  - 抑制性通道索引: {num_exc} 到 {total_channels - 1}")

    # 4. 准备 JSON 数据
    channel_info = {
        "exc_indices": exc_indices,
        "inh_indices": inh_indices
    }

    output_path = os.path.join(target_dir, "channel_info.json")

    # 5. 写入文件
    try:
        with open(output_path, 'w') as f:
            json.dump(channel_info, f, indent=4)
        print(f"\n[成功] 已创建 channel_info.json 文件于: {output_path}")
    except IOError as e:
        print(f"\n[错误] 无法写入文件: {e}")

if __name__ == "__main__":
    create_channel_info()