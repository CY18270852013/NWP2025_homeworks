# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 21:36:12 2025

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
dx = 0.1
dt = 0.004
lambda_val = dt / dx  # 0.04
M = 11                # 网格点数：0到10
x = np.linspace(0, 1, M, endpoint=True)
num_steps = 900       # 时间步数

# 初始条件 u(x,0) = sin(2πx)
u0 = np.sin(2 * np.pi * x)
u0[0] = 0.0          # 强制边界点初始值为0
u0[10] = 0.0

# 第一种差分格式（显式）
u_prev = u0.copy()
# 前向欧拉启动第一步
u_current = np.zeros_like(u0)
for m in range(M):
    if m == 0 or m == 10:  # 边界点跳过计算
        continue
    m_plus = m + 1
    m_minus = m - 1
    # 处理越界点
    u_plus = u0[m_plus] if m_plus < M else 0.0
    u_minus = u0[m_minus] if m_minus >= 0 else 0.0
    flux = (u_plus**2 - u_minus**2) / (4 * dx)
    u_current[m] = u0[m] - dt * flux

# 强制边界条件
u_current[0] = 0.0
u_current[10] = 0.0

K1 = [0.5 * np.sum(u0**2), 0.5 * np.sum(u_current**2)]

for _ in range(num_steps - 1):
    u_next = np.zeros_like(u_current)
    for m in range(M):
        if m == 0 or m == 10:  # 边界点保持为0
            u_next[m] = 0.0
            continue
        m_plus = m + 1
        m_minus = m - 1
        # 处理越界点
        u_plus = u_current[m_plus] if m_plus < M else 0.0
        u_minus = u_current[m_minus] if m_minus >= 0 else 0.0
        term = (u_plus + u_current[m])**2 - (u_current[m] + u_minus)**2
        u_next[m] = u_prev[m] - 0.01 * term  # 0.01 = 1/4 * λ
    
    # 更新时间层并保持边界
    u_prev, u_current = u_current.copy(), u_next.copy()
    K1.append(0.5 * np.sum(u_current**2))

# 第二种差分格式（隐式迭代）
u = u0.copy()
K2 = [0.5 * np.sum(u**2)]

for _ in range(num_steps):
    u_guess = u.copy()
    for __ in range(100):  # 最大迭代次数
        u_bar = 0.5 * (u_guess + u)
        u_next = np.zeros_like(u)
        for m in range(M):
            if m == 0 or m == 10:  # 边界点保持为0
                u_next[m] = 0.0
                continue
            m_plus = m + 1
            m_minus = m - 1
            # 处理越界点
            u_bar_plus = u_bar[m_plus] if m_plus < M else 0.0
            u_bar_minus = u_bar[m_minus] if m_minus >= 0 else 0.0
            sum_terms = u_bar_plus + u_bar[m] + u_bar_minus
            diff_terms = u_bar_plus - u_bar_minus
            u_next[m] = u[m] - (1/6) * lambda_val * sum_terms * diff_terms
        
        # 强制边界条件
        u_next[0] = 0.0
        u_next[10] = 0.0
        
        # 判断收敛
        if np.max(np.abs(u_next - u_guess)) < 1e-5:
            break
        u_guess = u_next.copy()
    
    u = u_next.copy()
    K2.append(0.5 * np.sum(u**2))

# 绘图
plt.figure(figsize=(12, 7))
plt.plot(np.arange(len(K1)), K1, 
         label=r'$u_j^{n+1} = u_j^{n-1} - \frac{1}{4}\lambda\left[(u_{j+1}^n + u_j^n)^2 - (u_j^n + u_{j-1}^n)^2\right]$',
         color='blue', linewidth=1.5)
plt.plot(np.arange(len(K2)), K2, 
         label=r'$u_j^{n+1} = u_j^n - \frac{1}{6}\lambda\left[(\overline{u}_{j+1} + \overline{u}_j + \overline{u}_{j-1})(\overline{u}_{j+1} - \overline{u}_{j-1})\right]$',
         color='red', linewidth=1.5)

# 坐标轴设置
plt.xlim(0, 900)
plt.ylim(0, 3)  # 调整纵轴范围
plt.xlabel('Time Step (n)', fontsize=12)
plt.ylabel(r'$\frac{1}{2} \sum u_m^{n^2}$', fontsize=14, rotation=90, labelpad=10)
plt.title('Answer to Question No.1 (Rigid Boundary)', fontsize=14)

# 图例设置
plt.legend(loc='upper right', fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()