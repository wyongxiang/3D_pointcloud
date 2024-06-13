import os.path
import time
import open3d as o3d
import numpy as np
import SimpleITK as sitk


# icp点云配准
def execute_icp(source_cloud, target_cloud, max_iterations=200, threshold=0.02):
    """
    使用Open3D库执行ICP点云配准。

    参数:
    - source_cloud: 源点云（Open3D.geometry.PointCloud对象）
    - target_cloud: 目标点云（Open3D.geometry.PointCloud对象）
    - max_iterations: 最大迭代次数，默认为200
    - threshold: 配准的容差，默认为0.02

    返回:
    - reg: 配准结果，包含变换矩阵
    """
    # 计算源点云的界框并缩放以更好地配准
    source_cloud.paint_uniform_color([1, 0, 0])  # 将源点云涂成红色以便可视化
    source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target_cloud.paint_uniform_color([0, 1, 0])  # 将目标点云涂成绿色以便可视化
    target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 执行ICP配准
    reg = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    return reg


# gicp点云配准
def execute_gicp(source_cloud, target_cloud, max_iterations=200, threshold=100):
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0],
                             [0.0, 0.0, 0.0, 1.0]])  # 初始变换矩阵，一般由粗配准提供

    # 计算源点云的界框并缩放以更好地配准
    source_cloud.paint_uniform_color([1, 0, 0])  # 将源点云涂成红色以便可视化
    source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target_cloud.paint_uniform_color([0, 1, 0])  # 将目标点云涂成绿色以便可视化
    target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # -------------------------------------------------
    generalized_icp = o3d.pipelines.registration.registration_generalized_icp(
        source_cloud, target_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations))  # 设置最大迭代次数

    return generalized_icp


def registration(source_bone_nii_file, target_bone_nii_file, source_ureter_nii_file, target_ureter_nii_file, output_nii_file):
    start = time.time()
    # bone mase
    source_bone_nii_info = sitk.ReadImage(source_bone_nii_file)
    source_bone_nii_data = sitk.GetArrayFromImage(source_bone_nii_info)  # z, y, x
    # get point of mast
    source_bone_points = np.array(np.where(source_bone_nii_data > 0)).T  # z, y, x
    print(f"source_bone_points shape:{source_bone_points.shape}")
    # 创建Open3D点云对象
    source_bone_pcd = o3d.geometry.PointCloud()
    source_bone_pcd.points = o3d.utility.Vector3dVector(source_bone_points)

    target_bone_nii_info = sitk.ReadImage(target_bone_nii_file)
    target_bone_nii_data = sitk.GetArrayFromImage(target_bone_nii_info)  # z, y, x
    # get point of mast
    target_bone_points = np.array(np.where(target_bone_nii_data > 0)).T  # z, y, x
    print(f"target_bone_points shape: {target_bone_points.shape}")
    # 创建Open3D点云对象
    target_bone_pcd = o3d.geometry.PointCloud()
    target_bone_pcd.points = o3d.utility.Vector3dVector(target_bone_points)

    # ureter data
    source_ureter_nii_info = sitk.ReadImage(source_ureter_nii_file)
    source_ureter_nii_data = sitk.GetArrayFromImage(source_ureter_nii_info)  # z, y, x
    # get point of mast
    source_ureter_points = np.array(np.where(source_ureter_nii_data > 0)).T  # z, y, x
    print(f"source_ureter_points shape:{source_ureter_points.shape}")
    # 创建Open3D点云对象
    source_ureter_pcd = o3d.geometry.PointCloud()
    source_ureter_pcd.points = o3d.utility.Vector3dVector(source_ureter_points)

    target_ureter_nii_info = sitk.ReadImage(target_ureter_nii_file)
    target_ureter_nii_data = sitk.GetArrayFromImage(target_ureter_nii_info)  # z, y, x
    # get point of mast
    target_ureter_points = np.array(np.where(target_ureter_nii_data > 0)).T  # z, y, x
    print(f"target_ureter_points shape: {target_ureter_points.shape}")
    # 创建Open3D点云对象
    target_ureter_pcd = o3d.geometry.PointCloud()
    target_ureter_pcd.points = o3d.utility.Vector3dVector(target_ureter_points)

    gicp_start = time.time()
    # 配准
    # reg_result = execute_icp(source_bone_pcd, target_bone_pcd)  # icp
    reg_result = execute_gicp(source_bone_pcd, target_bone_pcd)  # gicp
    source_pcd_transformed = source_ureter_pcd.transform(reg_result.transformation)
    gicp_end = time.time()
    # 保存结果
    result_pcd_points = np.asarray(source_pcd_transformed.points)
    # 初始化NIfTI图像的数据
    result_nii_data = np.zeros(target_ureter_nii_data.shape)

    # 遍历点云中的点，分配到对应的体素
    for point in result_pcd_points:
        idx = point.astype(int)
        if idx[0] >= result_nii_data.shape[0] or idx[1] >= result_nii_data.shape[1]  or idx[2] >= result_nii_data.shape[2]:
            continue
        result_nii_data[tuple(idx)] = 1  # 假设存在点则体素值为1，可按需调整

    # 创建SimpleITK图像对象并保存为NIfTI
    result_nii_data_points = np.array(np.where(result_nii_data > 0)).T  # z, y, x
    print(f"result_nii_data_points shape:{result_nii_data_points.shape}")
    itk_image = sitk.GetImageFromArray(result_nii_data)
    itk_image.CopyInformation(target_bone_nii_info)

    sitk.WriteImage(itk_image, output_nii_file)
    print(f"result save as {output_nii_file}")
    end = time.time()
    print(f"gicp time:{round(gicp_end - gicp_start, 8)} s")
    print(f"all time:{round(end - start, 8)} s")


def run():
    import shutil
    root_dir = "./temp/registration_ureter_wyx"
    source_bone_nii_file = f"{root_dir}/masks_bone/V_mask.nii.gz"
    target_bone_nii_file = f"{root_dir}/masks_bone/A_mask.nii.gz"

    source_ureter_nii_file = f"{root_dir}/masks_ureter/V_mask.nii.gz"
    target_ureter_nii_file = f"{root_dir}/masks_ureter/A_mask.nii.gz"
    if not os.path.exists(source_bone_nii_file) or not os.path.exists(target_bone_nii_file):
        print(f"source_bone_nii file not exist:{source_bone_nii_file}")
    save_nii_dir = f"{root_dir}/masks_ureter_debug510"
    if not os.path.exists(save_nii_dir):
        os.makedirs(save_nii_dir, exist_ok=True)
    basename = os.path.basename(target_ureter_nii_file)
    output_nii_file = f"{save_nii_dir}/gicp_pred_{basename}"
    registration(source_bone_nii_file, target_bone_nii_file,source_ureter_nii_file, target_ureter_nii_file, output_nii_file)
    shutil.copy(source_ureter_nii_file, save_nii_dir)
    shutil.copy(target_ureter_nii_file, save_nii_dir)


if __name__ == '__main__':
    run()
