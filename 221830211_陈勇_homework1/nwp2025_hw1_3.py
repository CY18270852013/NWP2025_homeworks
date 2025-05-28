import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = 'C:/Users/Chen Yong/Desktop/数值天气预报上机实习/data_hw1/data/NetCDF/hLat.198101-201012.clt.nc'

# 打开NetCDF文件
ds = xr.open_dataset(file_path)

# 提取500hPa高度的位势高度数据
h_500hPa = ds['hgt'].sel(lev=500, method='nearest')

# 计算多年月平均（1981年~2010年）
h_500hPa_monthly_mean = h_500hPa.groupby('time.month').mean(dim='time')

# 计算气压梯度项 -g * ∂h/∂x
dx = 2.5 * 111000  # 经度间隔2.5度，转换为米（假设地球半径为111000米）
dh_dx = h_500hPa_monthly_mean.differentiate('lon') / dx  # 计算位势高度的经向梯度
pressure_gradient_term = -9.81 * dh_dx  # 计算气压梯度项

# 对纬度维度取平均（因为数据是单纬度75°N）
pressure_gradient_term_mean = pressure_gradient_term.mean(dim='lat')

# 确保数据维度正确
# 检查 pressure_gradient_term_mean 的形状
print("Pressure gradient term shape:", pressure_gradient_term_mean.shape)

# 如果 pressure_gradient_term_mean 的形状是 (month, lon)，则直接使用
if pressure_gradient_term_mean.dims == ('month', 'lon'):
    # 提取数据值
    data = pressure_gradient_term_mean.values  # 数据形状应为 (12, 144)
    lon = pressure_gradient_term_mean['lon'].values  # 经度值，形状为 (144,)
    months = np.arange(1, 13)  # 月份，形状为 (12,)

    # 创建热力图
    plt.figure(figsize=(12, 6))

    # 绘制热力图
    heatmap = plt.pcolormesh(
        lon,  # 横轴：经度，形状为 (144,)
        months,  # 纵轴：月份（1到12月），形状为 (12,)
        data,  # 数据，形状为 (12, 144)
        shading='auto',
        cmap='coolwarm'  # 颜色映射
    )

    # 添加颜色条
    cbar = plt.colorbar(heatmap, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(r'$-g \frac{\partial h}{\partial x}$ (m s$^{-2}$)')

    # 设置标题和坐标轴标签
    plt.title('1981-2010 Monthly Mean Pressure Gradient Term at 75°N (500 hPa)', fontsize=14)
    plt.xlabel('Longitude (degrees)', fontsize=12)
    plt.ylabel('Month', fontsize=12)

    # 设置纵轴刻度为月份
    plt.yticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # 显示图形
    plt.show()
else:
    print("Error: Data dimensions are not compatible. Expected (month, lon), got", pressure_gradient_term_mean.dims)