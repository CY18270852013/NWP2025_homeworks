import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 文件路径
file_path = 'C:/Users/Chen Yong/Desktop/数值天气预报上机实习/data_hw1/data/NetCDF/uvhT.198101-201012.clmt.nc'

# 打开NetCDF文件
ds = xr.open_dataset(file_path)

# 检查数据集中的维度
print("Dataset dimensions:", ds.dims)

# 提取1月的数据（month=0 表示1月）
ds_jan = ds.isel(month=0)  # 使用 isel 按索引提取数据

# 计算多年1月平均（如果数据已经是多年平均，则跳过此步骤）
# 这里假设数据已经是多年平均，直接使用 ds_jan

# 提取500hPa高度的纬向风数据
u_500hPa = ds_jan['uwnd'].sel(level=500, method='nearest')

# 计算纬向风的平流项 -u * ∂u/∂x
dx = 2.5 * 111000  # 经度间隔2.5度，转换为米（假设地球半径为111000米）
du_dx = u_500hPa.differentiate('lon') / dx  # 计算纬向风的经向梯度
u_du_dx = -u_500hPa * du_dx  # 计算纬向风的平流项

# 提取纬度和经度数据
lat = ds_jan['lat'].values
lon = ds_jan['lon'].values

# 创建地图投影
projection = ccrs.PlateCarree()

# 创建高分辨率图像
plt.figure(figsize=(14, 8), dpi=300)

# 创建地图
ax = plt.axes(projection=projection)

# 添加海岸线和国界
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 绘制热力图
heatmap = ax.pcolormesh(
    lon,  # 横轴：经度
    lat,  # 纵轴：纬度
    u_du_dx,  # 数据，形状为 (lat, lon)
    shading='auto',
    cmap='coolwarm',  # 颜色映射
    transform=projection
)

# 添加颜色条
cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.02, aspect=30)
cbar.set_label(r'$-u \frac{\partial u}{\partial x}$ (m s$^{-2}$)', fontsize=12)
cbar.ax.tick_params(labelsize=10)  # 设置颜色条刻度字体大小

# 设置标题和坐标轴标签
plt.title('1981-2010 January Mean Zonal Wind Advection at 500 hPa', fontsize=16)
ax.set_xlabel('Longitude (degrees)', fontsize=14)
ax.set_ylabel('Latitude (degrees)', fontsize=14)

# 设置横轴和纵轴刻度
ax.set_xticks(np.arange(-180, 181, 60), crs=projection)
ax.set_yticks(np.arange(-90, 91, 30), crs=projection)
ax.gridlines(draw_labels=True, linestyle='--')

# 显示图形
plt.show()