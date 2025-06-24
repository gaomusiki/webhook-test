import torch
print(torch.__version__)  # 应该输出 2.7.1 或者你安装的版本号
print(torch.cuda.is_available())  # 应该输出 False，说明没有 CUDA 支持
