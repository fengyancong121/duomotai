# -*- coding: utf-8 -*-
"""
脉搏信号分析系统 v2.1
作者：AI助手
最后更新：2025-05-24
功能：实现ECG/RPPG信号按十个数据一组分段特征提取
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

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

def extract_features(segment, peaks, fs=1000):
    """提取信号特征"""
    features = {}

    # 基本统计特征
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['min'] = np.min(segment)
    features['max'] = np.max(segment)

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

def main(file_path):
    # 1. 加载数据
    raw_signal = load_data(file_path)

    # 2. 预处理信号
    filtered_signal = preprocess_signal(raw_signal)

    # 3. 检测峰值
    peaks = detect_peaks(filtered_signal)

    # 4. 按十个数据一组分段
    group_size = 10  # 每组10个数据点
    segments = [filtered_signal[i:i + group_size] for i in range(0, len(filtered_signal), group_size)]

    # 5. 提取每组特征
    features_list = []
    for idx, seg in enumerate(segments):
        # 检测当前组内的峰值
        group_peaks = detect_peaks(seg)
        # 提取特征
        features, _ = extract_features(seg, group_peaks)
        features['segment_id'] = idx + 1
        features_list.append(features)

    # 6. 保存为CSV文件
    df = pd.DataFrame(features_list)
    df.to_csv('pulse_features.csv', index=False)
    print(f"特征提取完成，结果已保存到 pulse_features.csv 文件中。")

if __name__ == "__main__":
    file_path = r"D:\dachuang\脉搏\sketch_250411a\ecg_data.txt"
    main(file_path)