import numpy as np
import matplotlib.pyplot as plt

# 参数设置
dx = 0.1
dt = 0.004
lambda_val = dt / dx  # 0.04
M = 11                # 网格点数：0到10
x = np.linspace(0, 1, M, endpoint=True)
num_steps = 250       # 时间步数为250

# 定义差分格式计算函数
def compute_k(initial_u):
    # 初始化存储
    u_prev = initial_u.copy()  # u^{n-1}
    
    # 前向欧拉启动第一步
    u_current = np.zeros_like(u_prev)
    for m in range(M):
        m_plus = (m + 1) % M
        m_minus = (m - 1) % M
        u_current[m] = u_prev[m] - lambda_val * u_prev[m] * (u_prev[m_plus] - u_prev[m_minus])
    
    K = [0.5 * np.sum(initial_u**2), 0.5 * np.sum(u_current**2)]
    
    # 跳蛙法迭代
    for _ in range(num_steps - 1):
        u_next = np.zeros_like(u_current)
        for m in range(M):
            m_plus = (m + 1) % M
            m_minus = (m - 1) % M
            u_next[m] = u_prev[m] - lambda_val * u_current[m] * (u_current[m_plus] - u_current[m_minus])
        
        # 更新时间层
        u_prev, u_current = u_current.copy(), u_next.copy()
        K.append(0.5 * np.sum(u_next**2))
    
    return K

# 两种初始条件
K_sin = compute_k(np.sin(2 * np.pi * x))
K_const = compute_k(1.5 + np.sin(2 * np.pi * x))

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

# 修改坐标轴范围
plt.xlim(0, 250)     # 横轴范围设为0-250
plt.ylim(0, 25)      # 纵轴范围设为0-25

# 坐标轴标签设置
plt.xlabel('Time Step (n)', fontsize=12)
plt.ylabel(r'$\frac{1}{2} \sum u_m^{n^2}$', 
          fontsize=14, 
          rotation=90, 
          labelpad=10)

# 标题和图例
plt.title('Answer to Question No.3(cyclical boundary)\n'
          r'$u_j^{n+1} = u_j^{n-1} - \lambda u_j^n (u_{j+1}^n - u_{j-1}^n)$', 
          fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()