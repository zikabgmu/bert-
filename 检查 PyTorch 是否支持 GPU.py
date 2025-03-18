import torch
print(torch.cuda.is_available())  # 如果返回 True，说明 GPU 可用
print(torch.cuda.device_count())  # 返回可用 GPU 的数量
print(torch.cuda.get_device_name(0))  # 返回第一个 GPU 的名称
