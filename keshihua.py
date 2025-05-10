
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
from scipy import signal
from scipy.stats import kurtosis, skew


def read_imu_data(file_path):
    """读取IMU数据文件"""
    try:
        data = pd.read_csv(file_path)
        print(f"数据读取成功，共读取到{len(data)}行数据")
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def extract_euler_features(data):
    """提取欧拉角特征"""
    if data is None or len(data) == 0:
        print("无有效数据可提取特征")
        return None

    features = {}
    angles = ['roll', 'pitch', 'yaw']

    for angle in angles:
        # 基本统计特征
        features[f'{angle}_mean'] = np.mean(data[angle])
        features[f'{angle}_std'] = np.std(data[angle])
        features[f'{angle}_max'] = np.max(data[angle])
        features[f'{angle}_min'] = np.min(data[angle])
        features[f'{angle}_range'] = features[f'{angle}_max'] - features[f'{angle}_min']
        features[f'{angle}_median'] = np.median(data[angle])
        features[f'{angle}_skewness'] = skew(data[angle])
        features[f'{angle}_kurtosis'] = kurtosis(data[angle])

        # 变化率特征
        diff = np.diff(data[angle])
        features[f'{angle}_diff_mean'] = np.mean(diff)
        features[f'{angle}_diff_std'] = np.std(diff)
        features[f'{angle}_diff_max'] = np.max(diff)
        features[f'{angle}_diff_min'] = np.min(diff)

        # 频率特征
        fs = 1 / np.mean(np.diff(data['timestamp']))  # 采样频率
        f, Pxx = signal.welch(data[angle], fs=fs, nperseg=min(256, len(data[angle])))
        features[f'{angle}_dominant_freq'] = f[np.argmax(Pxx)]
        features[f'{angle}_power_sum'] = np.sum(Pxx)

    return pd.DataFrame([features])


def plot_euler_angles(data):
    """绘制欧拉角随时间变化的曲线"""
    if data is None or len(data) == 0:
        print("无有效数据可绘制")
        return

    # 创建图形和子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25)

    # 绘制原始数据
    line1, = ax1.plot(data['timestamp'], data['roll'], 'r-', label='Roll')
    ax1.set_ylabel('Roll (deg)')
    ax1.grid(True)
    ax1.legend()

    line2, = ax2.plot(data['timestamp'], data['pitch'], 'g-', label='Pitch')
    ax2.set_ylabel('Pitch (deg)')
    ax2.grid(True)
    ax2.legend()

    line3, = ax3.plot(data['timestamp'], data['yaw'], 'b-', label='Yaw')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Yaw (deg)')
    ax3.grid(True)
    ax3.legend()

    # 添加范围滑动条
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = RangeSlider(
        ax_slider, 'Time Range',
        data['timestamp'].min(),
        data['timestamp'].max(),
        valinit=(data['timestamp'].min(), data['timestamp'].max())
    )

    def update(val):
        """更新图表显示范围"""
        start, end = val
        ax1.set_xlim(start, end)
        ax2.set_xlim(start, end)
        ax3.set_xlim(start, end)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.suptitle('Euler Angles over Time')
    plt.show()


def plot_3d_euler(data):
    """绘制3D欧拉角轨迹"""
    if data is None or len(data) == 0:
        print("无有效数据可绘制")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D轨迹
    ax.plot(data['roll'], data['pitch'], data['yaw'], 'b-', alpha=0.6)

    # 标记起点和终点
    ax.scatter(data['roll'].iloc[0], data['pitch'].iloc[0], data['yaw'].iloc[0],
               c='g', marker='o', s=100, label='Start')
    ax.scatter(data['roll'].iloc[-1], data['pitch'].iloc[-1], data['yaw'].iloc[-1],
               c='r', marker='o', s=100, label='End')

    ax.set_xlabel('Roll (deg)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_zlabel('Yaw (deg)')
    ax.set_title('3D Euler Angles Trajectory')
    ax.legend()
    plt.show()


def plot_angle_distributions(data):
    """绘制欧拉角分布直方图"""
    if data is None or len(data) == 0:
        print("无有效数据可绘制")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    angles = ['roll', 'pitch', 'yaw']
    colors = ['r', 'g', 'b']

    for ax, angle, color in zip(axes, angles, colors):
        ax.hist(data[angle], bins=30, color=color, alpha=0.7)
        ax.set_xlabel(f'{angle.capitalize()} (deg)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{angle.capitalize()} Distribution')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_fft_analysis(data):
    """绘制欧拉角FFT分析"""
    if data is None or len(data) == 0:
        print("无有效数据可绘制")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)

    angles = ['roll', 'pitch', 'yaw']
    colors = ['r', 'g', 'b']

    # 计算采样频率
    fs = 1 / np.mean(np.diff(data['timestamp']))

    for ax, angle, color in zip(axes, angles, colors):
        f, Pxx = signal.welch(data[angle], fs=fs, nperseg=min(256, len(data[angle])))
        ax.semilogy(f, Pxx, color=color)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [V**2/Hz]')
        ax.set_title(f'{angle.capitalize()} Power Spectral Density')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # 文件路径 - 替换为实际文件路径或使用文件选择对话框
    file_path = "imu_data_20250329_190440.txt"

    # 读取数据
    imu_data = read_imu_data(file_path)

    if imu_data is not None:
        # 提取特征
        features = extract_euler_features(imu_data)
        print("\n提取的欧拉角特征:")
        print(features.T)  # 转置以便更好地查看

        # 绘制时间序列图
        plot_euler_angles(imu_data)

        # 绘制3D轨迹图
        plot_3d_euler(imu_data)

        # 绘制角度分布图
        plot_angle_distributions(imu_data)

        # 绘制FFT分析图
        plot_fft_analysis(imu_data)

        if imu_data is not None:
            features = extract_euler_features(imu_data)
            posture_df = features.T.reset_index().rename(columns={'index': 'feature', 0: 'value'})
            posture_df.to_csv('posture_features.csv', index=False)


if __name__ == "__main__":
    main()