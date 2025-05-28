import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.font_manager import FontProperties

base_path = r"C:\Users\Chen Yong\Desktop\数值天气预报上机实习\221830211_陈勇_homework6\nwp_homework6\\"
height_file = "500_0800Z29APR1973.txt"

def load_height_data():
    """读取500hPa高度场数据"""
    with open(base_path + height_file, 'r', encoding='ANSI') as f:
        data = np.loadtxt(f, skiprows=1, dtype=np.float64)
    return data * 10  # 转换为位势米

def apply_boundary_nine_point_smoothing(data, S=0.5):
    """边界九点平滑"""
    smoothed_data = data.copy()
    original_data = data.copy()
    n_rows, n_cols = data.shape

    # 上边界内侧（i=1）
    for j in range(1, n_cols-1):
        i = 1
        F = original_data[i, j]
        term1 = (S / 2) * (1 - S) * (
            original_data[i+1, j] + original_data[i, j+1] + original_data[i-1, j] + original_data[i, j-1] - 4*F
        )
        term2 = (S**2 / 4) * (
            original_data[i+1, j+1] + original_data[i+1, j-1] + original_data[i-1, j+1] + original_data[i-1, j-1] - 4*F
        )
        smoothed_data[i, j] = F + term1 + term2

    # 下边界内侧（i=n_rows-2）
    for j in range(1, n_cols-1):
        i = n_rows - 2
        F = original_data[i, j]
        term1 = (S / 2) * (1 - S) * (
            original_data[i+1, j] + original_data[i, j+1] + original_data[i-1, j] + original_data[i, j-1] - 4*F
        )
        term2 = (S**2 / 4) * (
            original_data[i+1, j+1] + original_data[i+1, j-1] + original_data[i-1, j+1] + original_data[i-1, j-1] - 4*F
        )
        smoothed_data[i, j] = F + term1 + term2

    # 左边界内侧（j=1）
    for i in range(1, n_rows-1):
        j = 1
        F = original_data[i, j]
        term1 = (S / 2) * (1 - S) * (
            original_data[i+1, j] + original_data[i, j+1] + original_data[i-1, j] + original_data[i, j-1] - 4*F
        )
        term2 = (S**2 / 4) * (
            original_data[i+1, j+1] + original_data[i+1, j-1] + original_data[i-1, j+1] + original_data[i-1, j-1] - 4*F
        )
        smoothed_data[i, j] = F + term1 + term2

    # 右边界内侧（j=n_cols-2）
    for i in range(1, n_rows-1):
        j = n_cols - 2
        F = original_data[i, j]
        term1 = (S / 2) * (1 - S) * (
            original_data[i+1, j] + original_data[i, j+1] + original_data[i-1, j] + original_data[i, j-1] - 4*F
        )
        term2 = (S**2 / 4) * (
            original_data[i+1, j+1] + original_data[i+1, j-1] + original_data[i-1, j+1] + original_data[i-1, j-1] - 4*F
        )
        smoothed_data[i, j] = F + term1 + term2

    return smoothed_data

def apply_inner_five_point_smoothing(data, S=0.5):
    """内点五点平滑"""
    smoothed_data = data.copy()
    original_data = data.copy()
    n_rows, n_cols = data.shape

    # 遍历所有内点（i=2到n_rows-3，j=2到n_cols-3）
    for i in range(2, n_rows-2):
        for j in range(2, n_cols-2):
            F = original_data[i, j]
            term = (S / 4) * (
                original_data[i+1, j] + original_data[i, j+1] + original_data[i-1, j] + original_data[i, j-1] - 4*F
            )
            smoothed_data[i, j] = F + term

    return smoothed_data

def plot_initial_height(height):
    """绘制高度场"""
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    proj = ccrs.NorthPolarStereo(central_longitude=115, true_scale_latitude=45)
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([85, 150, 35, 70], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='navy')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--', linewidth=0.8, edgecolor='gray')
    ax.gridlines(draw_labels=True, linewidth=0.6, color='dimgray', alpha=0.8, linestyle=':')
    lon = np.linspace(85, 150, 20)
    lat = np.linspace(70, 35, 16)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    levels = np.arange(5000, 6100, 40)
    cs = ax.contour(lon_grid, lat_grid, height, levels=levels, colors='darkred', linewidths=1.5, transform=ccrs.PlateCarree())
    plt.clabel(cs, fmt='%d', fontsize=12, colors='darkred', inline_spacing=15)
    
    # 添加标签
    plt.title("1973年4月29日08时500hPa等压面位势高度场", fontproperties=font, pad=10, loc='center')
    ax.text(0.12, 0.08, '85E', transform=ax.transAxes, fontsize=10)
    ax.text(0.88, 0.08, '150E', transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.92, '70N', transform=ax.transAxes, fontsize=10)
    ax.text(0.5, 0.08, '35N', transform=ax.transAxes, fontsize=10)
    
    plt.savefig('smoothed_height.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    height_data = load_height_data()
    assert height_data.shape == (16, 20), "数据维度错误"
    
    # 平滑
    height_data = apply_boundary_nine_point_smoothing(height_data, S=0.5)
    height_data = apply_inner_five_point_smoothing(height_data, S=0.5)

    plot_initial_height(height_data)