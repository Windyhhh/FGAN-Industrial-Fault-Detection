"""
统一的参数配置文件 - 最终优化版
"""
# 特征提取参数
FEATURE_PARAMS = {
    'CPV': 0.99,
    'lag_number': 4
}

# 模型参数 - 针对泛化能力优化
MODEL_PARAMS = {
    'hidden_size': 128,
    'batch_size': 64,      # 增大批次大小以加快训练
    'num_epochs': 300,     # 减少到300个epoch以加快速度
    'lr': 0.0002,
    'beta1': 0.5,
    'dropout': 0.02,       # 进一步减小dropout
    'l2_penalty': 0.00005  # 进一步减小正则化
}

# 故障检测参数
DETECTION_PARAMS = {
    'threshold': 0.3,
    'window_size': 5,
    'confidence_level': 0.98  # 降低置信度以提高泛化
}

# 数据维度
DATA_DIMS = {
    'input_dim': 31,
    'feature_dim': None    # 自动确定
}

# 训练加速参数
TRAINING_PARAMS = {
    'num_workers': 8,
    'pin_memory': True,
    'use_amp': True,
    'batch_accumulation': 4
}