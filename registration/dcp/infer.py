import numpy as np
import torch
import time
import os
from model import DCP
from util import transform_point_cloud, npmat2euler
import argparse
from scipy.spatial.transform import Rotation
from data import ModelNet40
import glob
import h5py
import open3d as o3d


def transform_input(pointcloud):
    """
    random rotation and transformation the input
    pointcloud: N*3
    """

    anglex = np.random.uniform() * np.pi / 4
    angley = np.random.uniform() * np.pi / 4
    anglez = np.random.uniform() * np.pi / 4

    # anglex = 0.04
    # angley = 0.04
    # anglez = 0.04

    print('angle: ', anglex, angley, anglez)

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    R_ba = R_ab.T
    translation_ab = np.array([np.random.uniform(-0.5, 0.5),
                               np.random.uniform(-0.5, 0.5),
                               np.random.uniform(-0.5, 0.5)])

    # translation_ab = np.array([0.01,0.05,0.05])
    print('trans: ', translation_ab)

    translation_ba = -R_ba.dot(translation_ab)

    pointcloud1 = pointcloud[:, :3].T

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

    euler_ab = np.asarray([anglez, angley, anglex])
    euler_ba = -euler_ab[::-1]
    rotation_ba = Rotation.from_euler('zyx', euler_ba)

    pointcloud1 = np.random.permutation(pointcloud1.T)
    pointcloud2 = np.random.permutation(pointcloud2.T)

    return pointcloud1.astype('float32'), pointcloud2.astype('float32'), \
        rotation_ab, translation_ab, rotation_ba, translation_ba


def run_one_pointcloud(src, target, net):
    if len(src.shape) == 2 and len(target.shape) == 2:  ##  (N,3)

        print("src/target shape:", src.shape, target.shape)

        src = np.expand_dims(src[:, :3], axis=0)
        src = np.transpose(src, [0, 2, 1])  ##  (1, 3, 1024)
        target = np.expand_dims(target[:, :3], axis=0)
        target = np.transpose(target, [0, 2, 1])  ##  (1, 3, 1024)

    net.eval()

    src = torch.from_numpy(src).cuda()
    target = torch.from_numpy(target).cuda()

    rotation_ab_pred, translation_ab_pred, \
        rotation_ba_pred, translation_ba_pred = net(src, target)

    target_pred = transform_point_cloud(src, rotation_ab_pred,
                                        translation_ab_pred)

    src_pred = transform_point_cloud(target, rotation_ba_pred,
                                     translation_ba_pred)

    # put on cpu and turn into numpy
    src_pred = src_pred.detach().cpu().numpy()
    src_pred = np.transpose(src_pred[0], [1, 0])

    target_pred = target_pred.detach().cpu().numpy()
    target_pred = np.transpose(target_pred[0], [1, 0])

    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()

    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()

    return src_pred, target_pred, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/dcp_v2.t7',
                        metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--customizer', type=bool, default=False, metavar='N',
                        help='Whether to use customizer dataset')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # net prepared
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(args.model_path), strict=False)
    if args.customizer:
        """
        f = h5py.File('./data/registration_bone_data/masks_AD_h5_2048/case_011_D.h5', 'r')
        data = f['data'][:].astype('float32')  # (2048, 2048, 3) <class 'numpy.ndarray'>
        f.close()
        # index = np.random.randint(data.shape[0])
        # index = 0
        # point1 = data[index, :, :]
        point1 = pc_normalize(data)
        tgt_f = h5py.File('./data/registration_bone_data/masks_AD_h5_2048/case_011_D.h5', 'r')
        tgt_data = tgt_f['data'][:].astype('float32')
        tgt_f.close()
        # index = 0
        # point2 = data[index, :, :]
        point2 = pc_normalize(tgt_data)
        """
        print("using customizer dataset")
        src_pcd = o3d.io.read_point_cloud("./data/masks_ureter_surface_pcd/V_mask_2048.pcd")
        tgt_pcd = o3d.io.read_point_cloud("./data/masks_ureter_surface_pcd/A_mask_resampled_2048.pcd")
        data1 = np.asarray(src_pcd.points).astype('float32')
        data2 = np.asarray(tgt_pcd.points).astype('float32')
        point1 = pc_normalize(data1)
        point2 = pc_normalize(data2)
    else:
        print(f"using modelnet40 data")
        f = h5py.File('./data/modelnet40_ply_hdf5_2048/ply_data_train2.h5', 'r')
        data = f['data'][:].astype('float32')  # (2048, 2048, 3) <class 'numpy.ndarray'>
        f.close()
        # index = np.random.randint(data.shape[0])
        index = 0

        point1 = data[index, :, :]
        _, point2, _, _, _, _ = transform_input(point1)

        # src1=o3d.io.read_point_cloud("/home/pride/3d_registration/dcp-master/0_modelnet_src.ply")
        # point1=np.asarray(src1.points)
        # print(point1)
        # _, point2, _, _, _, _ = transform_input(point1)

    src, target = point1, point2

    ## run
    # src_pred是目标数据往源数据上配， target_pred是源数据往目标数据上配
    src_pred, target_pred, r_ab, t_ab, r_ba, t_ba, = run_one_pointcloud(src, target, net)

    print("#############  src -> target :\n", r_ab, t_ab)
    print("#############  src <- target :\n", r_ba, t_ba)

    # np->open3d
    src_cloud = o3d.geometry.PointCloud()
    src_cloud.points = o3d.utility.Vector3dVector(point1)
    tgt_cloud = o3d.geometry.PointCloud()
    tgt_cloud.points = o3d.utility.Vector3dVector(point2)
    trans_cloud = o3d.geometry.PointCloud()
    trans_cloud.points = o3d.utility.Vector3dVector(src_pred)

    trans_cloud2 = o3d.geometry.PointCloud()
    trans_cloud2.points = o3d.utility.Vector3dVector(target_pred)

    # view
    src_cloud.paint_uniform_color([1, 0, 0])
    tgt_cloud.paint_uniform_color([0, 1, 0])
    trans_cloud.paint_uniform_color([0, 0, 1])
    trans_cloud2.paint_uniform_color([0, 1, 1])
    # 保存点云
    save_result_dir = "./result_masks_ureter_surface_pcd/"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)
    src_pcd_file = f"{save_result_dir}/V_src_pcd_2048.pcd"
    tgt_pcd_file = f"{save_result_dir}/A_resampled_tgt_pcd_2048.pcd"
    trans_pcd_file = f"{save_result_dir}/V2A_resampled_trans_pcd_2048.pcd"
    trans_pcd_file2 = f"{save_result_dir}/A2V_resampled_trans_pcd_2048.pcd"
    o3d.io.write_point_cloud(src_pcd_file, src_cloud)
    o3d.io.write_point_cloud(tgt_pcd_file, tgt_cloud)
    o3d.io.write_point_cloud(trans_pcd_file, trans_cloud)
    o3d.io.write_point_cloud(trans_pcd_file2, trans_cloud2)
    # 可视化
    # o3d.visualization.draw_geometries([src_cloud, tgt_cloud, trans_cloud], width=800)
