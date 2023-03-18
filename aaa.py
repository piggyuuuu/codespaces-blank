import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easyocr
import re

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取图像
img = cv2.imread('v2-948def5ef922df0751307bb64b758901_r.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 生成高程数据
elevation = gray.astype(float)

# 归一化高程数据
elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

# 定义地形大小
width = elevation.shape[1]
height = elevation.shape[0]

# 添加随机噪声
noise = np.random.normal(scale=0.1, size=(height, width))
elevation += noise

# 对高程数据进行平滑处理
elevation_smooth = cv2.GaussianBlur(elevation, (0, 0), sigmaX=5, sigmaY=5)

# 设置高度缩放因子
scale_factor = 10

# 生成x, y网格坐标
x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

# 生成3D模型
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 画出地形
ax.plot_surface(x, y, elevation_smooth * scale_factor, cmap='terrain')

# 使用easyocr识别地名、图例和比例尺
reader = easyocr.Reader(['ch_sim'])
result = reader.readtext(gray)

# 定义函数来过滤基于行政级别的地名
def filter_by_admin_level(result, admin_level):
    filtered_result = []
    for detection in result:
        text = detection[1]
        if re.search(r'\b{}\b'.format(admin_level), text):
            filtered_result.append(detection)
    return filtered_result

# 过滤出符合行政级别的地名、图例和比例尺
province_result = filter_by_admin_level(result, '省|自治区|特别行政区')
legend_result = filter_by_admin_level(result, '米$')
scale_result = filter_by_admin_level(result, '(KM|公里|米)$')

# 将识别出的文本绘制到3D图中，并根据行政级别进行字体样式和大小的调整
for detection in province_result:
    text = detection[1]
    points = detection[0]
    x_center = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y_center = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    admin_level = re.search(r'\b{}\b'.format('省|自治区|特别行政区'), text).group()
    if admin_level == '省':
        font_weight = 'bold'
        font_size = 20
    else:
        font_weight = 'normal'
        font_size = 10 + 5 * len(admin_level)
    ax.text(x_center, y_center, elevation_smooth[int(y_center), int(x_center)], text, z=elevation_smooth[int(y_center), int(x_center)], color='black', fontsize=font_size, fontweight=font_weight)

for detection in legend_result:
    text = detection[1]
    points = detection[0]
    x_center = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y_center = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    ax.text(x_center, y_center, elevation_smooth[int(y_center), int(x_center)], text, z=elevation_smooth[int(y_center), int(x_center)], color='black', fontsize=15, fontweight='normal')

for detection in scale_result:
    text = detection[1]
    points = detection[0]
    x_center = (points[0][0] + points[2][0] + points[3][0]) / 4
    y_center = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    ax.text(x_center, y_center, elevation_smooth[int(y_center), int(x_center)], text, z=elevation_smooth[int(y_center), int(x_center)], color='black', fontsize=15, fontweight='normal')


# 显示3D模型
plt.show()
