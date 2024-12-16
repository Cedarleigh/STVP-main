"""First iterator"""
from os.path import join
from helper_ply import read_ply
from helper_tool import Config as cfg
from helper_tool import DataProcessing as DP
import re
import numpy as np
import tensorflow as tf
import time, pickle,glob, os,sys
from sklearn.neighbors import KDTree


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
    
class Lastloader:
    def __init__(self, val_area_idx,test_area_idx):
        self.name = '8i'
        self.path = ' ' # datapath
        self.ignored_labels = np.array([])
        
        self.val_area_idx=np.array([])
        self.val_area_idx=val_area_idx
        self.test_area_idx=test_area_idx
#        self.val_split = 'Frame' + str(val_area_idx)
        self.all_files = glob.glob(join(BASE_DIR,self.path,'original_ply','*.ply'))
        self.all_files = sorted(self.all_files, key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0'%s)])
        self.all_files=self.all_files[0:176]+self.all_files[235:299]+self.all_files[300:333]+self.all_files[335:354]+self.all_files[357:376]+self.all_files[383:396]+self.all_files[427:444]+self.all_files[470:491]+self.all_files[512:552]+self.all_files[556:594]+self.all_files[600:650]+self.all_files[663:667]+self.all_files[674:676]+self.all_files[688:692]+self.all_files[739:751]+self.all_files[755:884]+self.all_files[900:929]+self.all_files[934:1108]
        #
       
        self.blockfiles = glob.glob(join(BASE_DIR, self.path, 'blockindex', '*'))
        self.blockfiles = sorted(self.blockfiles,key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)])
        
        self.tilefiles=glob.glob(join(BASE_DIR,self.path,'smatileindex','*'))
        self.tilefiles= sorted(self.tilefiles, key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0'%s)])

        # Initiate containers
        self.val_proj = []
        self.test_proj=[]
        self.test_labels = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [],'testing':[]}
        self.input_colors = {'training': [], 'validation': [],'testing':[]}
        self.input_labels = {'training': [], 'validation': [],'testing':[]}
        self.input_names = {'training': [], 'validation': [],'testing':[]}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        
    
    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1].split('\\')[-1].split('.ply')[-2]
            name_idx=int(re.findall('(\d+)', cloud_name)[0])
            if name_idx in self.val_area_idx:
                cloud_split = 'validation'
            elif name_idx in self.test_area_idx:
                cloud_split='testing'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_names[cloud_split] += [cloud_name]
            self.input_labels[cloud_split] += [sub_labels]
            size = sub_colors.shape[0] * 4 * 7

        print(self.input_names)
        print('\nPreparing reprojected indices for validation and testing')
        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1].split('\\')[-1].split('.ply')[-2]
            name_idx=int(re.findall('(\d+)', cloud_name)[0])
            # Validation projection and labels
            if name_idx in self.val_area_idx:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx,labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
            elif name_idx in self.test_area_idx:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx,labels = pickle.load(f)
                self.test_proj+= [proj_idx]
                self.test_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
                 
    # Generate the input data flow

    def get_batch_gen(self, split):
        def sample(cloud_idx,pointslen,center_point,knn_num):
            noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            # Check if the number of points in the selected cloud is less than the predefined num_points
            if pointslen < knn_num:
                # Query all points within the cloud
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=pointslen)[1][0]
            else:
                # Query the predefined number of points
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=int(knn_num+1))[1][0]
            return queried_idx

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i,tree in enumerate(self.input_colors[split]):
             self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
             self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            for cloud_idx, name in enumerate(self.input_names[split]):

                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)  # #framet-1
                #input tiles
                idx = int(re.findall('(\d+)', name)[0])
                tilefiles = self.tilefiles[idx - 1]
                tilefile = glob.glob(join(tilefiles, '*'))  ##block
                tilefile = sorted(tilefile,key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)])

                blockfiles=self.blockfiles[idx-1]
                blockfile = glob.glob(join(blockfiles, '*.npy'))  ##tile
                blockfile = sorted(blockfile,key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)])
                possi = np.where(self.possibility[split][cloud_idx] <= 0.01)[0].shape[0]

                for fileidx in range(len(tilefile)):

                    smatilefs = glob.glob(join(tilefile[fileidx], '*.npy'))
                    smatilefs = sorted(smatilefs, key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)]) ##block

                    if (possi <= 2e+4 and split=='testing'):
                        point_ind = np.argmin(self.possibility[split][cloud_idx])
                        center_point = points[point_ind, :].reshape(1, -1)
                        queried_idx = sample(cloud_idx,len(points), center_point, cfg.num_points)

                    else:
                        queried_idx = []
                        if (len(smatilefs) != 0):
                                for sidx in range(len(smatilefs)):
                                    blockidx = np.load(smatilefs[sidx], allow_pickle=True)
                                    '''Output the points and possibility of this block'''
                                    # block_point = points[blockidx].reshape(-1, 3)
                                    block_possibility = self.possibility[split][cloud_idx][blockidx].reshape(blockidx.shape[0])

                                    '''choose center points'''
                                    center_ind = np.argmin(block_possibility)
                                    centerpoint = points[blockidx[center_ind]]
                                    KNN_num = cfg.num_points / len(smatilefs)
                                    block_choose_ind=sample(cloud_idx,len(blockidx),centerpoint,KNN_num)
                                    queried_idx.extend( block_choose_ind)
                        else:
                                blockidx = np.load(blockfile[fileidx])
                                '''Output the points and possibility of this block'''
                                # block_point = points[blockidx]
                                block_possibility = self.possibility[split][cloud_idx][blockidx].reshape(blockidx.shape[0])

                                '''choose center points'''
                                center_ind = np.argmin(block_possibility)
                                centerpoint = points[blockidx[center_ind]]
                                KNN_num = cfg.num_points
                                block_choose_ind=sample(cloud_idx,len(blockidx),centerpoint,KNN_num)
                                queried_idx.extend(block_choose_ind)

                    queried_idx = np.array(queried_idx)#.reshape(-1)
                    queried_idx = queried_idx[:cfg.num_points]

                    # delta = np.square(1 - dists / np.max(dists))#.reshape(-1, 1)
                    self.possibility[split][cloud_idx][queried_idx] += 0.5  # .reshape(-1,1)
                    self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
                    queried_pc_xyz = points[queried_idx]  # .reshape(-1,3)
                    queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]  # .reshape(-1,3)
                    queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]  # .reshape(-1)
                    if len(queried_pc_xyz) < cfg.num_points:
                        queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                            DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx,
                                        cfg.num_points)
                    if True:
                        yield (queried_pc_xyz.astype(np.float32),
                               queried_pc_colors.astype(np.float32),
                               queried_idx.astype(np.int32),
                               np.array([cloud_idx], dtype=np.int32))

        gen_func =spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None])
        return gen_func,gen_types, gen_shapes



    @staticmethod
    def get_tf_mapping():
        """Integrate and connect data, including location information, nearby point information, etc."""
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features,batch_pc_idx, batch_cloud_idx):
            batch_rgb=batch_features
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []
            input_colors=[]

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                sub_colors=batch_rgb[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]

                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                input_colors.append(batch_rgb)
                batch_xyz = sub_points
                batch_rgb=sub_colors
            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_pc_idx, batch_cloud_idx]
            input_list+=input_colors

            return input_list

        return tf_map


    def init_input_pipeline(self):
        print('Initiating input pipelines\n')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('testing')

        # Generator builds dataset
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        # Set up batch_size
        self.batch_train_data = self.train_data.batch(cfg.batch_size,drop_remainder=True)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size,drop_remainder=True)
        self.batch_test_data=self.test_data.batch(cfg.val_batch_size,drop_remainder=True)
        
        map_func = self.get_tf_mapping()
        # Set up preprocessing functions
        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data=self.batch_test_data.map(map_func=map_func)
        # Set up preloading of next batch of files
        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data=self.batch_test_data.prefetch(cfg.val_batch_size)

        # Constructs a reinitializable iterator, providing iterator outputs and operations
        # to initialize it with either the training or validation set.
        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs_last = iter.get_next()
        self.train_init_op_last = iter.make_initializer(self.batch_train_data)
        self.val_init_op_last = iter.make_initializer(self.batch_val_data)
        self.test_init_op_last=iter.make_initializer(self.batch_test_data)
