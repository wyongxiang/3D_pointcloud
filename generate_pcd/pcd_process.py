import os.path

import numpy as np
import open3d as o3d


class PcdProcess:
    def __init__(self, num_sampled=2048):
        self.num_sampled = num_sampled

    def sampling(self, pcd_file, save_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        # 将点云转换为NumPy数组以便进行处理
        points = np.asarray(pcd.points)
        # 使用最远点采样选取2048个点
        sample_indices = self.farthest_point_sampling(points, self.num_sampled)

        # 根据索引抽取点云
        sampled_points = points[sample_indices]

        # 将抽样后的点云转换回open3d的PointCloud对象以进行进一步处理或可视化
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        o3d.io.write_point_cloud(save_file, sampled_pcd)
        # sampled_pcd.paint_uniform_color([0, 1, 0])
        # 可视化抽样后的点云
        pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, sampled_pcd])

    def pc_normalize(self, pcd_file, save_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_points = np.asarray(pcd.points).astype('float32')
        pcd_normal_points = self._pc_normalize(pcd_points)

        pcd_normal = o3d.geometry.PointCloud()
        pcd_normal.points = o3d.utility.Vector3dVector(pcd_normal_points)

        # pcd.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(save_file, pcd_normal)
        # o3d.visualization.draw_geometries([pcd_normal])

    def rotate_with_center(self, pcd_file, save_file, axis='X', angle=45):
        # 读取PCD文件
        pcd_original = o3d.io.read_point_cloud(pcd_file)
        # 计算点云的质心（中心点）
        center = pcd_original.get_center()
        # 将点云数据转换为numpy数组以便操作，并将点云平移到原点
        points = np.asarray(pcd_original.points) - center
        if axis == 'X':
            # 定义旋转角度，这里以绕X轴旋转30度为例
            theta_x = np.radians(angle)  # 转换角度为弧度
            # 构建绕X轴的旋转矩阵
            rotation_matrix_x = np.array([[1, 0, 0],
                                          [0, np.cos(theta_x), -np.sin(theta_x)],
                                          [0, np.sin(theta_x), np.cos(theta_x)]])
            rotation_matrix = rotation_matrix_x
        elif axis == 'Y':
            theta_y = np.radians(angle)  # 转换角度为弧度
            # 构建绕Y轴的旋转矩阵
            rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                          [0, 1, 0],
                                          [-np.sin(theta_y), 0, np.cos(theta_y)]])
            rotation_matrix = rotation_matrix_y
        else:
            rotation_matrix = None
        # 应用旋转
        rotated_points = np.dot(points, rotation_matrix)

        # 将旋转后的点云移回到原来的中心位置
        rotated_points += center

        # 使用旋转后的点创建新的点云对象
        pcd_rotated = o3d.geometry.PointCloud()
        pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)
        # pcd_rotated.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(save_file, pcd_rotated)
        # 可视化原始点云和旋转后的点云
        o3d.visualization.draw_geometries([pcd_original, pcd_rotated],
                                          window_name="Point Cloud Visualization",
                                          width=1280, height=720)

    def farthest_point_sampling(self, points, num_samples):
        """最远点采样"""
        indices = np.zeros(num_samples, dtype=int)
        points = np.array(points)
        farthest = np.random.randint(0, len(points))
        distances = np.linalg.norm(points - points[farthest], axis=1)
        for i in range(1, num_samples):
            farthest = np.argmax(distances)
            indices[i] = farthest
            distances = np.minimum(distances, np.linalg.norm(points - points[farthest], axis=1))
        return indices

    def _pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


def test():
    src_pcd_file = '../data/xxx.pcd'
    save_file = '../data/generate_pcd.pcd'
    pcd_process = PcdProcess(num_sampled=2048)
    pcd_process.sampling(src_pcd_file, save_file)
    pcd_process.pc_normalize(src_pcd_file, save_file)


if __name__ == '__main__':
    test()
