import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import numpy as np
from datetime import datetime
import os

# 设备配置
par_notification_characteristic = 0x0007
par_write_characteristic = 0x0005
par_device_addr = "e2:ea:d9:a5:ec:07"

# 数据存储配置
DATA_DIR = "imu_data"
os.makedirs(DATA_DIR, exist_ok=True)
data_file = None


def init_data_file():
    """初始化数据文件"""
    global data_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"imu_data_{timestamp}.txt")
    data_file = open(filename, 'w')
    data_file.write("timestamp,roll,pitch,yaw,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
    print(f"Data will be saved to: {filename}")
    return data_file


def save_imu_data(data):
    """保存IMU数据到文件"""
    if data_file:
        line = f"{data['timestamp']:.3f},{data['roll']:.6f},{data['pitch']:.6f},{data['yaw']:.6f},"
        line += f"{data['accel_x']:.6f},{data['accel_y']:.6f},{data['accel_z']:.6f},"
        line += f"{data['gyro_x']:.6f},{data['gyro_y']:.6f},{data['gyro_z']:.6f}\n"
        data_file.write(line)
        data_file.flush()


def parse_imu(buf):
    """解析IMU数据并保存到文件"""
    scale = {
        'accel': 0.00478515625,
        'angle': 0.0054931640625,
        'gyro': 0.06103515625
    }

    data = {
        'timestamp': 0,
        'roll': 0, 'pitch': 0, 'yaw': 0,
        'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
        'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0
    }

    if buf[0] == 0x11:
        ctl = (buf[2] << 8) | buf[1]
        data['timestamp'] = ((buf[6] << 24) | (buf[5] << 16) | (buf[4] << 8) | buf[3]) / 1000.0

        L = 7
        # 更安全的数值转换方法
        def to_signed_short(value):
            return value - 0x10000 if value > 0x7FFF else value

        # 解析加速度 (0x0001)
        if (ctl & 0x0001):
            raw = (buf[L + 1] << 8) | buf[L]
            data['accel_x'] = to_signed_short(raw) * scale['accel']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['accel_y'] = to_signed_short(raw) * scale['accel']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['accel_z'] = to_signed_short(raw) * scale['accel']; L += 2

        # 解析陀螺仪 (0x0004)
        if (ctl & 0x0004):
            raw = (buf[L + 1] << 8) | buf[L]
            data['gyro_x'] = to_signed_short(raw) * scale['gyro']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['gyro_y'] = to_signed_short(raw) * scale['gyro']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['gyro_z'] = to_signed_short(raw) * scale['gyro']; L += 2

        # 解析欧拉角 (0x0040)
        if (ctl & 0x0040):
            raw = (buf[L + 1] << 8) | buf[L]
            data['roll'] = to_signed_short(raw) * scale['angle']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['pitch'] = to_signed_short(raw) * scale['angle']; L += 2
            raw = (buf[L + 1] << 8) | buf[L]
            data['yaw'] = to_signed_short(raw) * scale['angle']

            # 保存到文件
            save_imu_data(data)

            # 打印当前欧拉角
            print(f"\rRoll: {data['roll']:7.2f}° Pitch: {data['pitch']:7.2f}° Yaw: {data['yaw']:7.2f}°", end="")

async def main():
    # 初始化数据文件
    init_data_file()

    print("Scanning for device...")
    device = await BleakScanner.find_device_by_address(par_device_addr)
    if not device:
        print(f"Device {par_device_addr} not found")
        return

    async with BleakClient(device) as client:
        print("Connected to device")

        # 启用通知
        await client.start_notify(par_notification_characteristic, lambda c, d: parse_imu(d))

        # 初始化设备
        init_commands = [
            bytes([0x29]),  # 保持连接
            bytes([0x46]),  # 高速模式
            bytes([0x12, 5, 255, 0, 0, 60, 1, 3, 5, 0x40, 0x00]),  # 订阅欧拉角
            bytes([0x19])  # 开始数据流
        ]

        for cmd in init_commands:
            await client.write_gatt_char(par_write_characteristic, cmd)
            await asyncio.sleep(0.2)

        print("Data collection started (Ctrl+C to stop)...")

        # 保持运行
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nData collection stopped")
        if data_file:
            data_file.close()
    except Exception as e:
        print(f"\nError: {e}")
        if data_file:
            data_file.close()