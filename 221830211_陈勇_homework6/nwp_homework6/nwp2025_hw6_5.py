import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.font_manager import FontProperties

base_path = r"C:\Users\Chen Yong\Desktop\数值天气预报上机实习\221830211_陈勇_homework6\nwp_homework6\\"
height_file = "z_forecast.txt"
param_file = "PARAM.txt"

def load_height_data():
    """读取500hPa高度场数据"""
    with open(base_path + height_file, 'r', encoding='ANSI') as f:
        lines = [line for line in f if line.strip()][1:]
    return np.loadtxt(lines, dtype=np.float64) * 10  # 转换为位势米

def parse_parameters():
    """解析地图参数"""
    with open(base_path + param_file, 'r', encoding='ANSI') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    m_start = next(i for i, line in enumerate(lines) if "地图放大系数m_ij" in line)
    f_start = next(i for i, line in enumerate(lines) if "Coriolis系数f_ij" in line)
    
    map_factor = np.array([list(map(float, lines[i].split())) for i in range(m_start+1, m_start+17)])
    coriolis = np.array([list(map(float, lines[i].split())) for i in range(f_start+1, f_start+17)]) * 1e-4
    
    coriolis = np.where(np.abs(coriolis) < 1e-10, 1e-10, coriolis)
    return map_factor, coriolis

# 地转风计算
def calculate_geostrophic_wind(height, map_factor, coriolis, dx=300e3):
    """计算高精度地转风场"""
    gravity = 9.8
    dy = dx
    
    # 使用中心差分计算梯度
    dZdy, dZdx = np.gradient(height * gravity, dy, dx)
    
    # 稳定计算
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (map_factor / coriolis) * dZdy
        v = (map_factor / coriolis) * dZdx
    
    # 增强边界处理
    for var in [u, v]:
        var[:, [0, -1]] = 0.5*(var[:, [1, -2]] + var[:, [2, -3]])
        var[[0, -1], :] = 0.5*(var[[1, -2], :] + var[[2, -3], :])
    
    return np.nan_to_num(u), np.nan_to_num(v)

def plot_wind_field(u, v):
    """绘制密集风场矢量图"""
    proj = ccrs.NorthPolarStereo(
        central_longitude=115,
        true_scale_latitude=45
    )
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection=proj)
    
    # 设置地理范围（东经85-150，北纬35-70）
    ax.set_extent([85, 150, 35, 70], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.4)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6)
    
    lon = np.linspace(85, 150, 20)
    lat = np.linspace(70, 35, 16)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    skip = 1
    q = ax.quiver(
        lon_grid[::skip, ::skip],
        lat_grid[::skip, ::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        color='black',
        scale=800,
        width=0.002, 
        headwidth=2.8, 
        headlength=3.5,
        pivot='middle',
        transform=ccrs.PlateCarree()
    )
    
    plt.quiverkey(q, 0.88, 1.05, 50, 
                 '50 m/s', 
                 labelpos='E',
                 fontproperties={'size': 10, 'weight': 'bold'},
                 color='black',
                 labelsep=0.05)
    
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    plt.title("1973年4月30日08时500hPa预报风场", 
             fontproperties=font, 
             pad=18,
             loc='center')
    
    plt.savefig('dense_wind.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    height = load_height_data()
    map_factor, coriolis = parse_parameters()
    
    u, v = calculate_geostrophic_wind(height, map_factor, coriolis)
    
    plot_wind_field(u, v)