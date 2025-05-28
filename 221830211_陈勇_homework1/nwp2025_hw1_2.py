import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = 'C:/Users/Chen Yong/Desktop/数值天气预报上机实习/data_hw1/data/NetCDF/hLat.198101-201012.clt.nc'

# 打开NetCDF文件
ds = xr.open_dataset(file_path)

# 提取变量
u = ds['uwnd']  # 纬向风
v = ds['vwnd']  # 经向风
h = ds['hgt']   # 位势高度

# 计算30年平均
u_mean = u.mean(dim='time')
v_mean = v.mean(dim='time')
h_mean = h.mean(dim='time')

# 计算空间差分
dx = 2.5 * 111000  # 经度间隔2.5度，转换为米（假设地球半径为111000米）
du_dx = u_mean.differentiate('lon') / dx  #偏导数
dh_dx = h_mean.differentiate('lon') / dx  #偏导数

# 常数
f = 2 * 7.2921e-5 * np.sin(np.deg2rad(75))  # 科里奥利参数，75°N
g = 9.81  # 重力加速度

# 计算各项
u_du_dx = u_mean * du_dx
minus_fv = -f * v_mean
minus_g_dh_dx = -g * dh_dx

# 对高度层取平均，并去除多余的纬度维度
u_du_dx_mean = u_du_dx.mean(dim='lev').squeeze()
minus_fv_mean = minus_fv.mean(dim='lev').squeeze()
minus_g_dh_dx_mean = minus_g_dh_dx.mean(dim='lev').squeeze()

# 提取经度
lon = ds['lon']

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制三项
plt.plot(lon, u_du_dx_mean, label=r'$u \frac{\partial u}{\partial x}$')
plt.plot(lon, minus_fv_mean, label=r'$-fv$')
plt.plot(lon, minus_g_dh_dx_mean, label=r'$-g \frac{\partial h}{\partial x}$')

# 添加图例和标签
plt.legend()
plt.xlabel('Longitude (degrees)')
plt.ylabel('Value')
plt.title('30-year Mean Terms at 75°N')
plt.grid(True)

# 显示图形
plt.show()