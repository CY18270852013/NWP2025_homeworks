import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = 'C:/Users/Chen Yong/Desktop/数值天气预报上机实习/data_hw1/data/NetCDF/hLat.198101-201012.clt.nc'

# 打开NetCDF文件
ds = xr.open_dataset(file_path)

# 提取1月的数据
ds_jan = ds.sel(time=ds['time.month'] == 1)

# 计算多年1月平均
ds_jan_mean = ds_jan.mean(dim='time')

# 提取北纬75度的经向风数据，并去掉纬度维度
v = ds_jan_mean['vwnd'].sel(lat=75, method='nearest').squeeze()

# 计算科氏力项 -fv
f = 2 * 7.2921e-5 * np.sin(np.deg2rad(75))  # 科里奥利参数，75°N
minus_fv = -f * v

# 提取经度和高度层数据
lon = ds_jan_mean['lon'].values
lev = ds_jan_mean['lev'].values

# 确保数据维度正确
# minus_fv 的形状应为 (lev, lon)
if minus_fv.dims == ('lev', 'lon'):
    # 创建热力图
    plt.figure(figsize=(14, 8))

    # 绘制热力图
    heatmap = plt.pcolormesh(
        lon,  # 横轴：经度，形状为 (144,)
        lev,  # 纵轴：高度层，形状为 (17,)
        minus_fv,  # 数据，形状为 (17, 144)
        shading='auto',
        cmap='coolwarm'  # 颜色映射
    )

    # 添加颜色条
    cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(r'$-fv$ (m s$^{-2}$)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)  # 设置颜色条刻度字体大小

    # 设置标题和坐标轴标签
    plt.title('1981-2010 January Mean Coriolis Term at 75°N', fontsize=16)
    plt.xlabel('Longitude (degrees)', fontsize=14)
    plt.ylabel('Pressure Level (hPa)', fontsize=14)

    # 设置纵轴刻度为高度层
    plt.yticks(lev, fontsize=12)

    # 设置横轴刻度密度
    plt.xticks(np.arange(0, 360, 30), fontsize=12)  # 每30度一个刻度

    # 反转纵轴，使高度从低到高显示
    plt.gca().invert_yaxis()

    # 显示图形
    plt.show()
else:
    print("Error: Data dimensions are not compatible. Expected (lev, lon), got", minus_fv.dims)