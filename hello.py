import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取音频文件
audio_path = 'D:\\Users\\huxi.wav'
y, sr = librosa.load(audio_path, sr=None)

# 归一化音频数据
y = librosa.util.normalize(y)

# 提取MFCC特征，，
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 提取频谱质心
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

# 提取频谱带宽
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# 计算频谱质心和频谱带宽的均值
centroid_mean = np.mean(spectral_centroids)
bandwidth_mean = np.mean(spectral_bandwidth)

print("Spectral Centroid Mean:", centroid_mean)
print("Spectral Bandwidth Mean:", bandwidth_mean)

# 可视化MFCC特征
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 可视化频谱质心
plt.figure(figsize=(12, 4))
plt.plot(spectral_centroids, color='b')
plt.title('Spectral Centroids')
plt.show()

# 可视化频谱带宽
plt.figure(figsize=(12, 4))
plt.plot(spectral_bandwidth, color='r')
plt.title('Spectral Bandwidth')
plt.show()

# 原代码保持不变，在最后添加：


# 提取呼吸特征（假设每个时间窗口提取一组特征）
breath_features = []
for i in range(mfccs.shape[1]):  # 按MFCC帧数遍历
    # 取当前帧的MFCC均值（或根据需求设计特征）
    mfcc_frame = mfccs[:, i]
    current_features = [
        centroid_mean,          # 频谱质心均值
        bandwidth_mean,         # 频谱带宽均值
        *mfcc_frame             # 13维MFCC
    ]
    breath_features.append(current_features)

# 保存为CSV或NumPy文件
breath_df = pd.DataFrame(breath_features, columns=['centroid_mean', 'bandwidth_mean'] + [f'mfcc_{i}' for i in range(13)])
breath_df.to_csv('breath_features.csv', index=False)