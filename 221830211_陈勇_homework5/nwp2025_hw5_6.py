# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:48:56 2025

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
num_steps = 250       # 时间步数为250

# 定义差分格式计算函数（刚性边界条件）
def compute_k_rigid(initial_u):
    u_prev = initial_u.copy()  # u^{n-1}
    
    # 强制边界点初始值为0
    u_prev[0] = 0.0
    u_prev[10] = 0.0

    # 前向欧拉启动第一步
    u_current = np.zeros_like(u_prev)
    for m in range(M):
        if m == 0 or m == 10:  # 跳过边界点
            continue
        m_plus = m + 1
        m_minus = m - 1
        # 处理越界点
        u_plus = u_prev[m_plus] if m_plus < M else 0.0
        u_minus = u_prev[m_minus] if m_minus >= 0 else 0.0
        u_current[m] = u_prev[m] - lambda_val * u_prev[m] * (u_plus - u_minus)
    
    # 强制边界点保持为0
    u_current[0] = 0.0
    u_current[10] = 0.0

    K = [0.5 * np.sum(initial_u**2), 0.5 * np.sum(u_current**2)]
    
    # 跳蛙法迭代
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
            u_next[m] = u_prev[m] - lambda_val * u_current[m] * (u_plus - u_minus)
        
        # 更新时间层并保持边界
        u_prev, u_current = u_current.copy(), u_next.copy()
        K.append(0.5 * np.sum(u_next**2))
    
    return K

# 两种初始条件（初始化后强制边界为0）
u_sin = np.sin(2 * np.pi * x)
u_sin[0] = 0.0
u_sin[10] = 0.0

u_const = 1.5 + np.sin(2 * np.pi * x)
u_const[0] = 0.0
u_const[10] = 0.0

# 计算动能
K_sin = compute_k_rigid(u_sin)
K_const = compute_k_rigid(u_const)

# 绘图
plt.figure(figsize=(10, 6))

# 绘制两种初值的动能曲线
plt.plot(np.arange(len(K_sin)), K_sin, 
         label=r'$u_0 = \sin(2\pi x)$', 
         color='blue', 
         linewidth=1.5)

plt.plot(np.arange(len(K_const)), K_const, 
         label=r'$u_0 = 1.5 + \sin(2\pi x)$', 
         color='red', 
         linewidth=1.5)

# 坐标轴设置
plt.xlim(0, 250)
plt.ylim(0, 160)
plt.xlabel('Time Step (n)', fontsize=12)
plt.ylabel(r'$\frac{1}{2} \sum u_m^{n^2}$', fontsize=14, rotation=0, labelpad=25)

# 标题和图例
plt.title('Answer to Question No.3 (Rigid Boundary)\n'
          r'$u_j^{n+1} = u_j^{n-1} - \lambda u_j^n (u_{j+1}^n - u_{j-1}^n)$', 
          fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()