import numpy as np
import matplotlib.pyplot as plt

# 地球半径，单位：公里
R = 6371

# 定义纬度范围
latitudes = np.linspace(0, 89.9, 500)  # 从0°到89.9°，避免90°时的数值不稳定
phi = np.radians(latitudes)  # 转换为弧度

# 参考纬度 (标准纬度)
phi_0_polar = np.radians(60)  # 极射赤面投影的参考纬度
phi_0_lambert1 = np.radians(30)  # 兰勃特投影的第一个标准纬度
phi_0_lambert2 = np.radians(60)  # 兰勃特投影的第二个标准纬度
phi_0_mercator = np.radians(22.5)  # 墨卡托投影的标准纬度

# 计算极射赤面投影的地图放大系数 m 和极距 l
m_polar = (1 + np.sin(phi_0_polar)) / (1 + np.sin(phi))
l_polar = R * (1 + np.sin(phi_0_polar)) * np.cos(phi) / (1 + np.sin(phi))

# 计算兰勃特投影的圆锥常数 K
numerator_n = np.log(np.cos(phi_0_lambert1) / np.cos(phi_0_lambert2))
denominator_n = np.log(np.tan(np.pi/4 + phi_0_lambert2/2) / np.tan(np.pi/4 + phi_0_lambert1/2))
K_lambert = numerator_n / denominator_n

# 计算兰勃特投影的地图放大系数 m 和极距 l
m_lambert = np.cos(phi_0_lambert1) * ((1 + np.sin(phi_0_lambert1)) / np.cos(phi_0_lambert1))**K_lambert * \
            (1 / np.cos(phi)) * (np.cos(phi) / (1 + np.sin(phi)))**K_lambert

l_lambert = R * np.cos(phi_0_lambert1) / K_lambert * ((1 + np.sin(phi_0_lambert1)) / np.cos(phi_0_lambert1))**K_lambert * \
            (np.cos(phi) / (1 + np.sin(phi)))**K_lambert

# 计算墨卡托投影的地图放大系数 m 和极距 l
m_mercator = np.cos(phi_0_mercator) / np.cos(phi)
l_mercator = R * np.cos(phi_0_mercator) * np.log(np.tan(np.pi/4 + phi/2))

# 绘制地图放大系数 (m) 随纬度变化
plt.figure(figsize=(8, 6))
plt.plot(latitudes, m_polar, label='Polar', color='blue')
plt.plot(latitudes, m_lambert, label='Lambert', color='red')
plt.plot(latitudes, m_mercator, label='Mercator', color='green')
plt.xlabel('Latitude (°)')
plt.ylabel('Map Scaling Factor (m)')
plt.title('m-Latitude Relations of Different Map Projection Types')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 91, 10))
plt.show()

# 绘制极距 (l) 随纬度变化
plt.figure(figsize=(8, 6))
plt.plot(latitudes, l_polar, label='Polar', color='blue', linestyle='--')
plt.plot(latitudes, l_lambert, label='Lambert', color='red', linestyle='--')
plt.plot(latitudes, l_mercator, label='Mercator', color='green', linestyle='--')
plt.xlabel('Latitude (°)')
plt.ylabel('Distance to Pole (l) [km]')
plt.title('l-Latitude Relations of Different Map Projection Types')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 91, 10))
plt.show()