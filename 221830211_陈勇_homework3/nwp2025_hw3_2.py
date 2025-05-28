import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 参数设置 Parameters
c = 20.0    # 平流速度 (m/s)
dx = 400    # 空间步长 (m)
dt = 2      # 时间步长 (s)
nx = 360    # 空间网格数
nt = 300    # 时间步数
fps = 15    # GIF帧率

# 空间网格 (闭合圈，等分为360份)
x = np.linspace(0, 360, nx, endpoint=False)

# 初始条件 Initial condition
u0 = 20 * np.cos(np.radians(3 * x))  # 初始波形: 3个完整周期

# 初始化存储数组
u_history = np.zeros((nt, nx))
u = u0.copy()

# 数值求解
for n in range(nt):
    u_new = np.zeros_like(u)
    for m in range(nx):
        m_prev = (m - 1) % nx  # 周期性边界
        m_next = (m + 1) % nx
        u_new[m] = u[m] - c * dt/(2*dx) * (u[m_next] - u[m_prev])
    u = u_new
    u_history[n, :] = u

# 创建动画
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 360)
ax.set_ylim(-25, 25)
ax.set_xlabel("Grid Point (degree)", fontsize=12)
ax.set_ylabel("Zonal Wind Speed (m/s)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
title = ax.set_title(f"Numerical Solution of Advection Equation (Time = 0s)", fontsize=14)
line, = ax.plot([], [], "b-", linewidth=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, u_history[frame])
    title.set_text(f"Numerical Solution of Advection Equation (Time = {frame*dt}s)")
    return line, title

# 生成动画对象
ani = FuncAnimation(
    fig, 
    update, 
    frames=nt,
    init_func=init,
    blit=True,
    interval=1000/fps  # 控制动画速度
)

# 保存为GIF
print("Generating GIF animation, please wait...")
ani.save("wind_speed_animation.gif", 
         writer=PillowWriter(fps=fps), 
         progress_callback=lambda i, n: print(f"Progress: {i+1}/{n} frames"))
print("GIF animation saved as wind_speed_animation.gif")

# 显示静态图供预览 Show static plot for preview
plt.figure(figsize=(12, 6))
plt.plot(x, u0, label="Initial time")
plt.plot(x, u_history[-1], label=f"Final time (t={nt*dt}s)")
plt.xlabel("Grid Point (degree)")
plt.ylabel("Zonal Wind Speed (m/s)")
plt.title("Comparison of Initial and Final Wind Speed Distribution")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()