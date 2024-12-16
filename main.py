"""Data loading, model training, testing, and data visualization"""
from os.path import join
from STVPNet import Network
from test import ModelTester
from helper_tool import Config as cfg
from helper_tool import Plot
from loader import Nowloader
from last_loader import Lastloader
import numpy as np
import tensorflow as tf
import argparse, os, sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Import data path
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

    
def load_data():


            data1 = Lastloader(np.concatenate([np.arange(0, 30, 1),np.arange(300,333,1),np.arange(739,751,1),np.arange(900,929,1)],axis=0),np.concatenate([np.arange(30,60,1),np.arange(472,491,1),np.arange(756,785,1),np.arange(935,971,1)],axis=0)) #无标签的#np.arange(1,11,1),np.arange(11,21,1)
            data1.init_input_pipeline()
            data2 = Nowloader(np.concatenate([np.arange(1, 31, 1),np.arange(301,334,1),np.arange(740,752,1), np.arange(901, 930, 1)], axis=0), np.concatenate([np.arange(31,61,1),np.arange(473,492,1),np.arange(757,786,1),np.arange(936,972,1)],axis=0)) #由标签的#np.arange(2,12,1),np.arange(12,22,1)
            data2.init_input_pipeline()

            return data1, data2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='the number of GPUs to use [default: 0,1,2,3]')  
    parser.add_argument('--mode', type=str, default='test', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES=0


    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode
    dataset = load_data()

    if Mode == 'train':
       model = Network(dataset,cfg)
       model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path != 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]  # -1
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
#        tester.Printgraph(model, dataset)
        tester.test(model, dataset)

    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
                
