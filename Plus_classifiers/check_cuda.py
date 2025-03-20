
import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA设备数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前CUDA设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(0))
    
# 简单验证
if torch.cuda.is_available():
    # 创建一个测试张量并移到GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("测试张量在:", x.device)
else:
    print("CUDA不可用，仍使用CPU")