# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:49:35 2025
@author: Chen Yong
Modified for:
  1. Hovmöller diagram (time-space color map)
  2. Line plots at specific grid points (x=60, 100, 120, 140)
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
c = 20.0  # 平流速度 (m/s)
dx = 400  # 空间步长 (m)
dt = 2    # 时间步长 (s)
nx = 360  # 空间网格数
nt = 300  # 时间步数

# 空间网格 (闭合圈，等分为360份)
x = np.linspace(0, 360, nx, endpoint=False)  # 0到360度，不包括360

# 初始条件
k = 3 / dx  # 波数
u0 = 20 * np.cos(np.radians(3 * x))  # u_m^0 = 20 * cos(3m°)

# 初始化数值解和存储数组
u = u0.copy()
u_new = np.zeros(nx)
u_history = np.zeros((nt, nx))  # 存储所有时间步的风速

# 迭代求解并保存历史数据
for n in range(nt):
    for m in range(nx):
        m_prev = (m - 1) % nx
        m_next = (m + 1) % nx
        u_new[m] = u[m] - c * dt / (2 * dx) * (u[m_next] - u[m_prev])  # 中央差分
    u = u_new.copy()
    u_history[n, :] = u  # 保存当前时间步的结果

# 时间网格
time_grid = np.arange(nt)

# 图1：风速填色图（Hovmöller图）
plt.figure(figsize=(10, 6))
plt.pcolormesh(x, time_grid, u_history, shading='auto', cmap='rainbow')
plt.colorbar(label='Zonal Velocity (m/s)')
plt.xlabel('Grid Point (degree)')
plt.ylabel('Time grid')
plt.title('Hovmöller Diagram: u(x, t)')
plt.grid(linestyle='--', alpha=0.5)
plt.show()

# 图2：特定格点的风速随时间变化
selected_points = [60, 100, 120, 140]  # 需要分析的格点
plt.figure(figsize=(10, 6))
for point in selected_points:
    plt.plot(time_grid, u_history[:, point], label=f'x = {point}')
plt.xlabel('Time grid')
plt.ylabel('Zonal Velocity (m/s)')
plt.title('Time Evolution of u at Selected Grid Points')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()