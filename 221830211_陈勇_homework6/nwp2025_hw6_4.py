import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.font_manager import FontProperties
import os

# 全局参数
GRID_SPACE = 300000    # 格点间距(m)
TIME_STEP = 600        # 时间步长(s)
SMOOTH_S = 0.5         # 平滑系数
FORECAST_DAYS = 1      # 预报天数

def safe_load_data():
    data_path = r"C:\Users\Chen Yong\Desktop\数值天气预报上机实习\221830211_陈勇_homework6\nwp_homework6\500_0800Z29APR1973.txt"
    with open(data_path, 'r', encoding='ANSI') as f:
        data = np.loadtxt(f, skiprows=1)
        if data.shape != (16, 20):
            raise ValueError("数据维度应为16行20列，实际获取：{}".format(data.shape))
        return data * 10

def parse_parameters():
    param_path = r"C:\Users\Chen Yong\Desktop\数值天气预报上机实习\221830211_陈勇_homework6\nwp_homework6\PARAM.txt"
    with open(param_path, 'r', encoding='ANSI') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    try:
        # 读取m参数保持16×20
        m_start = next(i for i, line in enumerate(lines) if "地图放大系数m_ij" in line)
        m = np.array([list(map(float, lines[i].split())) for i in range(m_start+1, m_start+17)])
        
        # 读取f参数保持16×20
        f_start = next(i for i, line in enumerate(lines) if "Coriolis系数f_ij" in line)
        f = np.array([list(map(float, lines[i].split())) for i in range(f_start+1, f_start+17)]) * 1e-4
    except StopIteration:
        raise ValueError("参数文件格式错误")
    
    return m, np.where(np.abs(f) < 1e-10, 1e-10, f)

# 初始化参数矩阵 (16×20)
m_param, f_coriolis = parse_parameters()
grav = 9.8

# 获取网格尺寸 (16行20列)
z_init = safe_load_data()
n_rows, n_cols = z_init.shape  # (16, 20)

# 初始化风场
u_init = np.zeros_like(z_init)
v_init = np.zeros_like(z_init)

# 计算地转风初始场
for i in range(n_rows):  # 纬度方向16行
    for j in range(n_cols):  # 经度方向20列
        # 经向风计算
        if i == 0:  # 北边界
            v_term = (z_init[i+1, j] - z_init[i, j]) / GRID_SPACE
        elif i == n_rows-1:  # 南边界
            v_term = (z_init[i, j] - z_init[i-1, j]) / GRID_SPACE
        else:  # 内部点
            v_term = (z_init[i+1, j] - z_init[i-1, j]) / (2*GRID_SPACE)
        
        v_init[i, j] = (m_param[i, j] * grav / f_coriolis[i, j]) * v_term

        # 纬向风计算
        if j == 0:  # 西边界
            u_term = (z_init[i, j+1] - z_init[i, j]) / GRID_SPACE
        elif j == n_cols-1:  # 东边界
            u_term = (z_init[i, j] - z_init[i, j-1]) / GRID_SPACE
        else:  # 内部点
            u_term = (z_init[i, j+1] - z_init[i, j-1]) / (2*GRID_SPACE)
        
        u_init[i, j] = -(m_param[i, j] * grav / f_coriolis[i, j]) * u_term

# 微分计算函数
def y_derivative(arr):  # 经度方向导数（沿列）
    deriv = np.zeros_like(arr)
    deriv[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2*GRID_SPACE)
    return deriv

def x_derivative(arr):  # 纬度方向导数（沿行）
    deriv = np.zeros_like(arr)
    deriv[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2*GRID_SPACE)
    return deriv

def yy_term(a, b):  # 经度二阶项
    term = np.zeros_like(a)
    term[:, 1:-1] = ((a[:, 1:-1] + a[:, 2:]) * (b[:, 2:] - b[:, 1:-1])/(2*GRID_SPACE) +
                    (a[:, :-2] + a[:, 1:-1]) * (b[:, 1:-1] - b[:, :-2])/(2*GRID_SPACE)) / 2
    return term

def xx_term(a, b):  # 纬度二阶项
    term = np.zeros_like(a)
    term[1:-1, :] = ((a[1:-1, :] + a[2:, :]) * (b[2:, :] - b[1:-1, :])/(2*GRID_SPACE) +
                    (a[:-2, :] + a[1:-1, :]) * (b[1:-1, :] - b[:-2, :])/(2*GRID_SPACE)) / 2
    return term

def calc_tendency(u, v, z):
    f_total = f_coriolis + u*y_derivative(m_param) - v*x_derivative(m_param)
    
    # 动量方程
    du_dt = -m_param * (xx_term(u, u) + yy_term(v, u) + grav*x_derivative(z)) + f_total*v
    dv_dt = -m_param * (xx_term(u, v) + yy_term(v, v) + grav*y_derivative(z)) - f_total*u
    
    # 连续方程
    dz_dt = -(m_param**2) * (xx_term(u, z/m_param) + yy_term(v, z/m_param) + 
                             (z/m_param)*(x_derivative(u) + y_derivative(v)))
    
    # 边界置零
    du_dt[[0, -1], :] = 0; du_dt[:, [0, -1]] = 0
    dv_dt[[0, -1], :] = 0; dv_dt[:, [0, -1]] = 0
    dz_dt[[0, -1], :] = 0; dz_dt[:, [0, -1]] = 0
    
    return du_dt, dv_dt, dz_dt

def euler_backward(u, v, z):
    k1_u, k1_v, k1_z = calc_tendency(u, v, z)
    u_pred = u + TIME_STEP * k1_u
    v_pred = v + TIME_STEP * k1_v
    z_pred = z + TIME_STEP * k1_z
    k2_u, k2_v, k2_z = calc_tendency(u_pred, v_pred, z_pred)
    return u + TIME_STEP*k2_u, v + TIME_STEP*k2_v, z + TIME_STEP*k2_z

def central_diff(u_prev2, v_prev2, z_prev2, tendency):
    return (u_prev2 + 2*TIME_STEP*tendency[0], 
            v_prev2 + 2*TIME_STEP*tendency[1], 
            z_prev2 + 2*TIME_STEP*tendency[2])

# 平滑函数
def temporal_smooth(data_3d):
    return ((1-SMOOTH_S)*data_3d[...,1] + 
            SMOOTH_S/2*(data_3d[...,2] + data_3d[...,0]))

def spatial_smooth9(u, v, z):
    smoothed = [arr.copy() for arr in [u, v, z]]
    for i in [0, -1]:
        for j in [0, -1]:
            for k, arr in enumerate([u, v, z]):
                term1 = SMOOTH_S*(1-SMOOTH_S)*(arr[i+1,j] + arr[i-1,j] + arr[i,j+1] + arr[i,j-1] - 4*arr[i,j])/2
                term2 = (SMOOTH_S**2)*(arr[i+1,j+1] + arr[i+1,j-1] + arr[i-1,j+1] + arr[i-1,j-1] -4*arr[i,j])/4
                smoothed[k][i,j] += term1 + term2
    return smoothed

def spatial_smooth5(u, v, z):
    smoothed = [arr.copy() for arr in [u, v, z]]
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            for k, arr in enumerate([u, v, z]):
                smoothed[k][i,j] += SMOOTH_S/4*(arr[i+1,j] + arr[i-1,j] + arr[i,j+1] + arr[i,j-1] -4*arr[i,j])
    return smoothed

# 主积分循环
total_steps = FORECAST_DAYS * 144 + 1  # 增加一个时间步缓冲
u = np.zeros((n_rows, n_cols, total_steps))
v = np.zeros_like(u)
z = np.zeros_like(u)

# 设置初始场
u[...,0], v[...,0], z[...,0] = u_init, v_init, z_init

current_step = 0
for cycle in range(FORECAST_DAYS*2):
    start_step = cycle * 72
    end_step = (cycle+1)*72
    while current_step < end_step and current_step < total_steps-1:
        # 欧拉后差阶段
        if current_step < start_step + 6:
            if current_step == 0:
                u[...,1], v[...,1], z[...,1] = euler_backward(u_init, v_init, z_init)
                current_step += 1
            else:
                next_step = current_step + 1
                u[...,next_step], v[...,next_step], z[...,next_step] = euler_backward(
                    u[...,current_step], v[...,current_step], z[...,current_step])
                current_step += 1
        
        # 三步法起步
        elif current_step == start_step + 6:
            du, dv, dz = calc_tendency(u[...,current_step], v[...,current_step], z[...,current_step])
            u_half = u[...,current_step] + 0.5*TIME_STEP*du
            v_half = v[...,current_step] + 0.5*TIME_STEP*dv
            z_half = z[...,current_step] + 0.5*TIME_STEP*dz
            du2, dv2, dz2 = calc_tendency(u_half, v_half, z_half)
            next_step = current_step + 1
            u[...,next_step] = u[...,current_step] + TIME_STEP*du2
            v[...,next_step] = v[...,current_step] + TIME_STEP*dv2
            z[...,next_step] = z[...,current_step] + TIME_STEP*dz2
            current_step += 1
        
        # 中央差分
        else:
            tendency = calc_tendency(u[...,current_step], v[...,current_step], z[...,current_step])
            next_step = current_step + 1
            u[...,next_step], v[...,next_step], z[...,next_step] = central_diff(
                u[...,current_step-1], v[...,current_step-1], z[...,current_step-1], tendency)
            current_step += 1
        
        # 时间平滑在36和37时间层
        if current_step in [37, 38]:
            n = current_step - 1
            if n >= 1 and (n+1) < total_steps:
                u[...,n] = temporal_smooth(u[..., [n-1, n, n+1]])
                v[...,n] = temporal_smooth(v[..., [n-1, n, n+1]])
                z[...,n] = temporal_smooth(z[..., [n-1, n, n+1]])
        
        # 空间平滑（每6步）
        if current_step % 6 == 0 and current_step < end_step:
            u[...,current_step], v[...,current_step], z[...,current_step] = spatial_smooth9(
                u[...,current_step], v[...,current_step], z[...,current_step])
    
    # 最终平滑（每个周期结束）
    if current_step < total_steps:
        u[...,current_step], v[...,current_step], z[...,current_step] = spatial_smooth5(
            u[...,current_step], v[...,current_step], z[...,current_step])

# 坐标转换参数
EARTH_RADIUS = 6371000
REF_I, REF_J = 5, 8
REF_LON, REF_LAT = 90.7780, 52.9192
PROJ_K = 0.71557
phi0 = np.deg2rad(30)  # 标准纬度
leq = (EARTH_RADIUS * np.cos(phi0)/PROJ_K) * ((1+np.sin(phi0))/np.cos(phi0))**PROJ_K
l0 = leq * (np.cos(np.deg2rad(REF_LAT))/(1+np.sin(np.deg2rad(REF_LAT))))**PROJ_K

# 生成经纬度网格
lon_grid = np.zeros((16, 20))
lat_grid = np.zeros((16, 20))
for i in range(16):
    for j in range(20):
        dx = (i - REF_I) * GRID_SPACE
        dy = (REF_J - j) * GRID_SPACE
        dist = np.sqrt(dx**2 + (l0 - dy)**2)
        ratio = (dist/leq)**(2/PROJ_K)
        lat_grid[i,j] = np.rad2deg(np.arcsin((1 - ratio)/(1 + ratio)))
        # 处理dx=0的情况
        if dx == 0:
            lon_offset = 0
        else:
            lon_offset = np.rad2deg(np.arcsin(dx/(dist + 1e-9)))/PROJ_K  # 避免除以零
        lon_grid[i,j] = REF_LON + lon_offset

def plot_contour(data, title, filename, levels=np.arange(5180, 5881, 40)):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    proj = ccrs.NorthPolarStereo(central_longitude=115, true_scale_latitude=45)
    
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.set_extent([85,150,35,70], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--')
    ax.gridlines(draw_labels=True, linestyle=':')
    
    lon = np.linspace(85,150,20)
    lat = np.linspace(70,35,16)
    cs = ax.contour(lon, lat, data, levels=levels,
                   colors='darkred', linewidths=1.5, transform=ccrs.PlateCarree())
    
    plt.clabel(cs, fmt='%d', fontsize=10)
    plt.title(title, fontproperties=font)
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# plot_contour(z_init, '500hPa高度场 初始时刻', 'z0.png')
forecast_step = 144  # 24小时预报对应144个时间步
plot_contour(z[...,forecast_step], '500hPa高度场 24小时预报', 'z24.png')

np.savetxt(r"C:\Users\Chen Yong\Desktop\数值天气预报上机实习\221830211_陈勇_homework6\nwp_homework6\\"+"z_forecast.txt", z[...,forecast_step] / 10, fmt='%6.1f',
          header="模式预报结果（单位：位势十米）", comments='', encoding='ANSI')