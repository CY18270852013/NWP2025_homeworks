import numpy as np
import matplotlib.pyplot as plt

# 参数设置
dx = 0.1
dt = 0.004
lambda_val = dt / dx  # 0.04
M = 11  # 网格点数：0到10
x = np.linspace(0, 1, M, endpoint=True)
num_steps = 3000  # 时间步数

# 初始条件 u(x,0) = 1.5 + sin(2πx)
u0 = 1.5 + np.sin(2 * np.pi * x)

# 第一种差分格式
u_prev = u0.copy()
# 使用前向欧拉启动第一步
u_current = np.zeros_like(u0)
for m in range(M):
    m_plus = (m + 1) % M
    m_minus = (m - 1) % M
    flux = (u0[m_plus]**2 - u0[m_minus]**2) / (4 * dx)
    u_current[m] = u0[m] - dt * flux

K1 = [0.5 * np.sum(u0**2), 0.5 * np.sum(u_current**2)]

for _ in range(num_steps - 1):
    u_next = np.zeros_like(u_current)
    for m in range(M):
        m_plus = (m + 1) % M
        m_minus = (m - 1) % M
        term = (u_current[m_plus] + u_current[m])**2 - (u_current[m] + u_current[m_minus])**2
        u_next[m] = u_prev[m] - 0.01 * term  # 0.01 = 1/4 * λ
    u_prev, u_current = u_current.copy(), u_next.copy()
    K1.append(0.5 * np.sum(u_current**2))

# 第二种差分格式
u = u0.copy()
K2 = [0.5 * np.sum(u**2)]

for _ in range(num_steps):
    u_guess = u.copy()
    converged = False
    for __ in range(100):  # 最大迭代次数
        u_bar = 0.5 * (u_guess + u)
        u_next = np.zeros_like(u)
        for m in range(M):
            m_plus = (m + 1) % M
            m_minus = (m - 1) % M
            sum_terms = u_bar[m_plus] + u_bar[m] + u_bar[m_minus]
            diff_terms = u_bar[m_plus] - u_bar[m_minus]
            u_next[m] = u[m] - (1/6) * lambda_val * sum_terms * diff_terms
        if np.max(np.abs(u_next - u_guess)) < 1e-5:
            converged = True
            break
        u_guess = u_next.copy()
    u = u_next.copy()
    K2.append(0.5 * np.sum(u**2))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(len(K1)), K1, 
    label=r'$u_j^{n+1} = u_j^{n-1} - \frac{1}{4}\lambda\left[(u_{j+1}^n + u_j^n)^2 - (u_j^n + u_{j-1}^n)^2\right]$', 
    color='blue'
)

plt.plot(
    np.arange(len(K2)), K2, 
    label=r'$u_j^{n+1} = u_j^n - \frac{1}{6}\lambda\left[(\overline{u}_{j+1} + \overline{u}_j + \overline{u}_{j-1})(\overline{u}_{j+1} - \overline{u}_{j-1})\right]$', 
    color='red'
)
plt.xlabel('Time Step (n)', fontsize=12)
plt.ylabel('Average Kinetic Energy (K(n))', fontsize=12)
plt.title('Answer to Question No.2(Initial $u_0 = 1.5 + \sin(2\pi x)$, cyclical boundary)', fontsize=14)
plt.legend()
plt.grid(True)
plt.xlim(0, 3000)
plt.ylim(14.85, 15.3)
plt.show()