# "C:/Users/冯言聪/AppData/Local/Temp/untitled4136384970192062101sketches/sketch_250411a/ecg_data.txt"

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd

def load_data(file_path):
    """加载数据文件"""
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file if line.strip()]
    return np.array(data)

def preprocess_signal(signal, fs=1000, lowcut=0.5, highcut=45):
    """预处理信号：带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_peaks(signal, fs=1000, min_distance=0.5):
    """检测峰值(R波/脉搏波峰值)"""
    min_samples = int(min_distance * fs / 1000)
    if min_samples < 1:
        min_samples = 1  # 确保最小样本距离至少为1
    peaks, _ = find_peaks(signal, distance=min_samples, prominence=50)
    return peaks

def calculate_hr(peaks, fs=1000):
    """计算心率/脉搏率"""
    rr_intervals = np.diff(peaks) / fs * 1000  # 转换为毫秒
    heart_rate = 60 / (rr_intervals / 1000)  # 转换为bpm
    return heart_rate, rr_intervals

def extract_features(signal, peaks, fs=1000):
    """提取信号特征"""
    features = {}

    # 基本统计特征
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)

    # 心率变异性(HRV)特征
    rr_intervals = np.diff(peaks) / fs * 1000  # RR间期(毫秒)

    if len(rr_intervals) > 1:
        features['hr_mean'] = np.mean(60 / (rr_intervals / 1000))
        features['hr_std'] = np.std(60 / (rr_intervals / 1000))
        features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        features['pnn50'] = (features['nn50'] / len(rr_intervals)) * 100
    else:
        features['hr_mean'] = np.nan
        features['hr_std'] = np.nan
        features['rmssd'] = np.nan
        features['nn50'] = np.nan
        features['pnn50'] = np.nan

    return features, rr_intervals

def plot_signal(signal, peaks, fs=1000, title='Signal with Detected Peaks'):
    """绘制信号和检测到的峰值"""
    plt.figure(figsize=(15, 6))
    plt.plot(signal, label='Signal')
    plt.plot(peaks, signal[peaks], 'rx', label='Detected Peaks')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def plot_hrv(rr_intervals, title='Heart Rate Variability (RR Intervals)'):
    """绘制心率变异性(RR间期)"""
    if len(rr_intervals) < 2:
        print("Not enough peaks for HRV analysis")
        return

    plt.figure(figsize=(15, 6))
    plt.plot(rr_intervals, 'o-')
    plt.title(title)
    plt.xlabel('Beat Number')
    plt.ylabel('RR Interval (ms)')
    plt.grid()
    plt.show()

def plot_histogram(rr_intervals, title='RR Interval Histogram'):
    """绘制RR间期直方图"""
    if len(rr_intervals) < 2:
        print("Not enough peaks for histogram")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(rr_intervals, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel('RR Interval (ms)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def main(file_path):
    # 1. 加载数据
    raw_signal = load_data(file_path)

    # 2. 预处理信号
    filtered_signal = preprocess_signal(raw_signal)

    # 3. 检测峰值
    peaks = detect_peaks(filtered_signal)

    # 4. 计算心率和提取特征
    heart_rate, rr_intervals = calculate_hr(peaks)
    features, rr_intervals = extract_features(filtered_signal, peaks)

    # 5. 打印特征
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value:.2f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}")

    # 6. 可视化
    plot_signal(raw_signal, peaks, title='Raw Signal with Detected Peaks')
    plot_signal(filtered_signal, peaks, title='Filtered Signal with Detected Peaks')

    if len(peaks) > 1:
        plot_hrv(rr_intervals)
        plot_histogram(rr_intervals)

        # 绘制心率变化
        plt.figure(figsize=(15, 6))
        plt.plot(heart_rate, 'o-')
        plt.title('Heart Rate (BPM) Over Time')
        plt.xlabel('Beat Number')
        plt.ylabel('Heart Rate (BPM)')
        plt.grid()
        plt.show()

    pulse_df = pd.DataFrame([features])
    pulse_df.to_csv('pulse_features.csv', index=False)

    return features


if __name__ == "__main__":
    file_path = "D:\dachuang\脉搏\sketch_250411a\ecg_data.txt"
    # 替换为您的文件路径
    features = main(file_path)