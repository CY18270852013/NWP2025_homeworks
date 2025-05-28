# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:34:33 2025

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
c = 20.0    # 平流速度 (m/s)
dx = 400    # 空间步长 (m)
dt = 2      # 时间步长 (s)
nx = 360    # 空间网格数
tbar = 360  # 总时间步数
CFL = c*dt/dx  # CFL数=0.1

# 空间网格 (闭合圈，等分为360份)
x = np.linspace(-180, 180, nx, endpoint=False)*400  # 转换为实际距离(m)

# 初始条件
u0 = np.where(x < 0, 0.0, 1.0)  # 严格的阶跃函数初始条件

# 初始化存储数组
u_history = np.zeros((tbar+1, nx))
u_history[0] = u0.copy()

# 蛙跳格式初始化 (需要两个时间层)
u_prev = u0.copy()
u_current = u0.copy()

# 用前向欧拉完成第一步
for m in range(nx):
    m_prev = (m - 1) % nx
    m_next = (m + 1) % nx
    u_current[m] = u0[m] - c*dt/(2*dx)*(u0[m_next] - u0[m_prev])
u_history[1] = u_current.copy()

# 蛙跳法迭代
for n in range(1, tbar):
    u_next = np.zeros(nx)
    for m in range(nx):
        m_prev = (m - 1) % nx
        m_next = (m + 1) % nx
        # 蛙跳格式：u^{n+1} = u^{n-1} - CFL*(u^{n}_{m+1} - u^{n}_{m-1})
        u_next[m] = u_prev[m] - CFL*(u_current[m_next] - u_current[m_prev])
    
    u_prev = u_current.copy()
    u_current = u_next.copy()
    u_history[n+1] = u_current.copy()

# 创建14个子图布局
plt.figure(figsize=(15, 20))
selected_t = list(range(0, 361, 30))  # 0到360每隔30取一个时间步

for idx, t in enumerate(selected_t):
    plt.subplot(7, 2, idx+1)
    plt.plot(x / 400, u_history[t], 'k-', lw=1)  # 转换单位为km
    # plt.xlim(-72, 72)  # 对应180*400m=72km
    plt.xlim(-180, 180)
    plt.ylim(-0.5, 1.5)  # 固定y轴范围显示振荡
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'tbar = {t}', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('u (m/s)')

plt.tight_layout()
plt.savefig('numerical_solutions.png', dpi=300)
plt.show()