import os
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d


def downsample_ply(input_path, output_path, downsample_factor=0.5):
    """
    读取PLY文件 -> 下采样 -> 保存结果
    :param input_path: 输入PLY文件路径
    :param output_path: 输出PLY文件路径
    :param downsample_factor: 下采样比例(保留点的百分比)
    """
    # 1. 用plyfile读取PLY
    ply_data = PlyData.read(input_path)
    vertex_data = ply_data['vertex']

    # 提取xyz和rgb
    x = np.asarray(vertex_data['x'])
    y = np.asarray(vertex_data['y'])
    z = np.asarray(vertex_data['z'])
    points = np.vstack((x, y, z)).T

    # 检查颜色是否存在（处理可能没有rgb的情况）
    # has_color = all([prop.name in vertex_data.dtype.names for prop in ['red', 'green', 'blue']])
    # if has_color:
    r = np.asarray(vertex_data['red'])
    g = np.asarray(vertex_data['green'])
    b = np.asarray(vertex_data['blue'])
    colors = np.vstack((r, g, b)).T
    # else:
    #     colors = np.zeros_like(points)  # 创建占位颜色

    # 2. 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化到[0,1]

    # 3. 随机下采样
    down_pcd = pcd.random_down_sample(downsample_factor)

    # 4. 转换回numpy数组
    down_points = np.asarray(down_pcd.points)
    down_colors = np.asarray(down_pcd.colors) * 255.0  # 恢复[0,255]范围

    # 5. 用plyfile保存下采样结果
    # 构造结构化数组
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # 注意：如果原始数据没有颜色，这里会创建全0的颜色值
    vertex_arr = np.zeros(down_points.shape[0], dtype=dtype)
    vertex_arr['x'] = down_points[:, 0]
    vertex_arr['y'] = down_points[:, 1]
    vertex_arr['z'] = down_points[:, 2]

    # if has_color:
    vertex_arr['red'] = down_colors[:, 0].astype(np.uint8)
    vertex_arr['green'] = down_colors[:, 1].astype(np.uint8)
    vertex_arr['blue'] = down_colors[:, 2].astype(np.uint8)
    # else:
        # 移除颜色信息（如果不需要）
        # dtype = dtype[:3]  # 仅保留坐标
        # vertex_arr = vertex_arr.astype(dtype)

    # 创建PlyElement并保存
    vertex_element = PlyElement.describe(vertex_arr, 'vertex')
    PlyData([vertex_element]).write(output_path)


if __name__ == '__main__':
    int_dir = "/hy-tmp/outputs/plys"
    out_dir = "/hy-tmp/multi_view_all"
    scenes = ["delivery_area", "delivery_area", "bridge", "living_room", "exhibition_hall", "relief", "relief_2", ]
        # "botanical_garden", "boulders", "bridge", "courtyard", "",
        #       "door", "electro", "", "", "kicker", "lecture_room",
        #       "", "lounge", "meadow", "observatory", "office", "old_computer",
        #       "pipes", "playground", "statue", "terrace", "terrace_2", "terrains"]

    for scene in scenes:
        print(scene)
        downsample_ply(
            input_path=os.path.join(int_dir, scene+".ply"),
            output_path=os.path.join(int_dir, out_dir, scene+".ply"),
            downsample_factor=0.7  # 保留10%的点
        )
