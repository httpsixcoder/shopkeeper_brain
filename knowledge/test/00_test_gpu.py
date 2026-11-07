import torch
import time


def test_gpu():
    print("=" * 50)
    print("🔍 开始检测 GPU 环境")
    print("=" * 50)

    # 1. 检查 CUDA 是否可用（核心中的核心）
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {cuda_available}")

    if not cuda_available:
        print("❌ 未能检测到 GPU，请检查 PyTorch 是否装成了 CPU 版本！")
        return

    # 2. 显卡基本信息
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")

    # 3. 显存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"总显存: {total_memory / 1024 ** 3:.2f} GB")

    # 4. 实际运算测试（矩阵乘法）
    print("\n" + "=" * 50)
    print("⚡ 正在对比 CPU 和 GPU 的矩阵运算速度")
    print("=" * 50)

    matrix_size = 5000  # 定义矩阵大小

    # --- CPU 测试 ---
    a_cpu = torch.randn(matrix_size, matrix_size)
    b_cpu = torch.randn(matrix_size, matrix_size)

    start_time = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU 计算耗时: {cpu_time:.4f} 秒")

    # --- GPU 测试 ---
    # 必须把数据放到显卡上
    a_gpu = a_cpu.to('cuda')
    b_gpu = b_cpu.to('cuda')

    # 预热一下（GPU第一次运算可能包含初始化时间）
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 等待 GPU 完成

    start_time = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 必须加这句，等 GPU 算完再计时
    gpu_time = time.time() - start_time
    print(f"GPU 计算耗时: {gpu_time:.4f} 秒")

    # 5. 结果对比
    print("\n" + "=" * 50)
    speed_up = cpu_time / gpu_time
    print(f"🚀 结论: GPU 比 CPU 快了 {speed_up:.1f} 倍！")
    print("=" * 50)


if __name__ == "__main__":
    test_gpu()