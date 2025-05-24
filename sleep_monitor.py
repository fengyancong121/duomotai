import pandas as pd
import numpy as np

# 1. 加载各模态特征
breath_df = pd.read_csv('breath_features.csv')
pulse_df = pd.read_csv('pulse_features.csv')
posture_df = pd.read_csv('posture_features.csv')

# 2. 确保样本数一致（按时间对齐）
min_samples = min(len(breath_df), len(pulse_df), len(posture_df))
breath_df = breath_df.iloc[:min_samples]
pulse_df = pulse_df.iloc[:min_samples]
posture_df = posture_df.iloc[:min_samples]

# 3. 横向拼接特征（确保所有列均为数值型）
fused_df = pd.concat([breath_df, pulse_df, posture_df], axis=1)

# 4. 检查是否有非数值列（例如字符串）
if fused_df.select_dtypes(include=['object']).shape[1] > 0:
    print("发现非数值列:", fused_df.select_dtypes(include=['object']).columns)
    fused_df = fused_df.select_dtypes(exclude=['object'])  # 移除字符串列

# 5. 特征标准化（手动实现）
def standardize_features(df):
    standardized_df = (df - df.mean()) / df.std()
    return standardized_df

scaled_features = standardize_features(fused_df)

# 6. 保存融合后的特征为CSV文件
scaled_features_df = pd.DataFrame(scaled_features, columns=fused_df.columns)
scaled_features_df.to_csv('fused_features.csv', index=False)
print("融合后特征已保存到 fused_features.csv 文件中。")