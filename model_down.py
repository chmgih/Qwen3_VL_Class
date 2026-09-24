
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-2B-Instruct')


# import torch

# print("="*50)
# print("PyTorch 版本:", torch.__version__)
# print("CUDA 是否可用:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("CUDA 版本:", torch.version.cuda)
#     print("GPU 数量:", torch.cuda.device_count())
#     print("当前 GPU 名称:", torch.cuda.get_device_name(0))

#     # ========== 新增：查看显存 ==========
#     total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
#     reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
#     allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
#     free_mem = total_mem - reserved_mem

#     print(f"GPU 总显存: {total_mem:.2f} GB")
#     print(f"已保留显存: {reserved_mem:.2f} GB")
#     print(f"已使用显存: {allocated_mem:.2f} GB")
#     print(f"可用显存: {free_mem:.2f} GB")
#     # ===================================

#     print("✅ CUDA 正常，可以使用GPU！")
# else:
#     print("❌ CUDA 不可用，只能用CPU！")

# print("="*50)