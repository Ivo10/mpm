import torch

# 假设这是你的原始tensor
original_tensor = torch.tensor([1, 0, -1, 1, 0, 1, -1, 0, 1])

# 生成train_mask
train_mask = (original_tensor == 1) | (original_tensor == 0)  # 直接使用布尔掩码

# 生成test_mask
test_mask = original_tensor == -1

print("Original tensor:", original_tensor)
print("Train mask:", train_mask)
print("Test mask:", test_mask)
print("Train masked tensor:", original_tensor[train_mask])  # 使用布尔掩码进行索引
print("Test masked tensor:", original_tensor[test_mask])  # 使用布尔掩码进行索引
