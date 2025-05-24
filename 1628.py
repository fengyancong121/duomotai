import pandas as pd

# 加载原始 CSV 文件
df = pd.read_csv('posture_features.csv')

# 将数据转换为一行一行的格式
df_transposed = df.set_index('feature').T.reset_index(drop=True)

# 保存转换后的数据到新的 CSV 文件
df_transposed.to_csv('posture_features_transposed.csv', index=False)

print("转换后的文件已保存为 posture_features_transposed.csv")