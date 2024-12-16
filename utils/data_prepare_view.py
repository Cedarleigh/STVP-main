"""View data preprocessing"""
from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

dataset_path = ' '  # datapath
anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(dataset_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'meta/class_names.txt'))]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

sub_grid_size = 0.04
original_pc_folder = os.path.join(dirname(dataset_path), 'original_viewply')
sub_pc_folder =os.path.join(dirname(dataset_path), 'inputview_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'


def convert_pc2ply(anno_path, save_path):

    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
       class_name = os.path.basename(f).split('.')[0].split('_')[1]
       if class_name == 'visiable':

           pc = pd.read_csv(f, header=None, delim_whitespace=True).values
           pc[:,3:6]=190
       else:
           pc = pd.read_csv(f, header=None, delim_whitespace=True).values
           pc[:,3:6]=0
#        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]

       data_list.append(pc)  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)

    write_ply(save_path, (xyz, colors), ['x', 'y', 'z', 'red', 'green', 'blue'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors= DP.grid_sub_sampling(xyz, colors,labels=None,grid_size=sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split('\\')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('\\')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('\\')[-1][:-4]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx], f)


if __name__ == '__main__':

    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('\\')
        out_file_name = elements[-1] + out_format
        convert_pc2ply(annotation_path,join(original_pc_folder, out_file_name))

