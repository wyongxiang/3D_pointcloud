import numpy as np
import SimpleITK as sitk
import open3d as o3d
import h5py
from random import sample


class DataConv:
    def __init__(self):
        pass

    def nifit2pcd(self, nifit_file, pcd_file):
        nii_info = sitk.ReadImage(nifit_file)
        nii_data = sitk.GetArrayFromImage(nii_info)  # z, y, x
        nii_data = np.transpose(nii_data, (2, 1, 0))  # x, y, z
        # get point of mast
        points = np.array(np.where(nii_data > 0)).T  # x, y, z
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 添加颜色或其他属性
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # 保存为PCD文件
        o3d.io.write_point_cloud(pcd_file, pcd)
        return

    def nifit2h5(self, nifit_file, h5_file, num_points=2048):
        nii_info = sitk.ReadImage(nifit_file)
        nii_data = sitk.GetArrayFromImage(nii_info)

        spacing = nii_info.GetSpacing()
        origin = nii_info.GetOrigin()
        direction = nii_info.GetDirection()

        # 获取非零像素的索引
        valid_pixels = np.argwhere(nii_data != 0)

        # 如果有效像素数量超过2048，进行随机采样
        if valid_pixels.shape[0] > num_points:
            sampled_indices = np.array(sample(list(valid_pixels), num_points))
        else:
            # 如果有效像素不足2048，可以重复某些点或者填充零，但这里简单地使用所有找到的点
            sampled_indices = valid_pixels

        # 将索引转换为模型期望的格式，例如，对于ModelNet40，通常有一个'data'和'label'字段
        with h5py.File(h5_file, 'w') as hf:
            hf.create_dataset('data', data=sampled_indices, compression='gzip')
            hf.create_dataset('spacing', data=spacing, compression='gzip')
            hf.create_dataset("origin", data=origin, compression='gzip')
            hf.create_dataset("direction", data=direction, compression='gzip')
        print(f"Mask has been downsampled and saved as an HDF5 file with {num_points} points.")

    def pcd2nifit(self, pcd_file, src_file, dst_file):
        # 读取点云数据
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_points = np.asarray(pcd.points).astype(int)
        # 初始化NIfTI图像的数据
        nii_info = sitk.ReadImage(src_file)
        nii_data = sitk.GetArrayFromImage(nii_info)  # z, y, x
        dst_nii_data = np.zeros(nii_data.shape)
        dst_nii_data = np.transpose(dst_nii_data, (2, 1, 0))

        # 遍历点云中的点，分配到对应的体素
        for point in pcd_points:
            idx = point.astype(int)
            dst_nii_data[tuple(idx)] = 1  # 假设存在点则体素值为1，可按需调整

        # 创建SimpleITK图像对象并保存为NIfTI
        dst_nii_data = np.transpose(dst_nii_data, (2, 1, 0))
        itk_image = sitk.GetImageFromArray(dst_nii_data)
        itk_image.CopyInformation(nii_info)

        sitk.WriteImage(itk_image, dst_file)

    def get_surface_from_nii(self, nifit_file, pcd_file):
        import nibabel as nib
        from vedo import Volume, Points, write
        '''
        # nibabel 库
        # 读取 .nii.gz 文件
        nii_img = nib.load(nifit_file)
        # 获取数据并确保是二值图像
        data = nii_img.get_fdata() > 0  # 假设非零值代表感兴趣区域
        '''
        # simpleITK
        nii_img = sitk.ReadImage(nifit_file)
        nii_data = sitk.GetArrayFromImage(nii_img)
        nii_data = np.transpose(nii_data, (2, 1, 0))
        data = nii_data > 0

        # 使用vedo生成表面
        vol = Volume(data)
        surf = vol.isosurface()

        # 将表面转换为点云
        point_cloud = Points(surf.vertices)
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.vertices)
        # pcd.paint_uniform_color([0, 0, 1])
        o3d.io.write_point_cloud(pcd_file, pcd)

    def visualize_pcd(self, pcd_file):
        # 读取PCD文件
        pcd = o3d.io.read_point_cloud(pcd_file)
        # 设置色彩
        pcd.paint_uniform_color([1, 0, 0])
        # 可视化点云
        o3d.visualization.draw_geometries([pcd])


def test():
    src_file = '../data/mask.nii.gz'
    pcd_file = '../data/mask.pcd'
    save_file = '../data/generate_mask.nii.gz'
    data_conv = DataConv()
    data_conv.nifit2pcd(src_file, pcd_file)
    data_conv.visualize_pcd(pcd_file)
    data_conv.pcd2nifit(pcd_file, src_file,save_file)


if __name__ == '__main__':
    test()