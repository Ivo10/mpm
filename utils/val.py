# 计算验证集准确率
def calculate_accuracy(val_result, y):
    predictions = val_result[:, 0] > 0.5
    correct = (predictions == y[:, 0]).sum().item()
    total = y.size(0)
    accuracy = correct / total
    return accuracy