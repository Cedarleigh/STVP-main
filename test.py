"""Evaluate the performance of the trained model on the test dataset"""
from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
from helper_tool import Config as cfg
import tensorflow as tf
import numpy as np
import time,re,os,sys,glob
from pykdtree.kdtree import KDTree
from plyfile import PlyData, PlyElement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.path= self.path = ' '  # datapath
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + '.txt', 'a')
        self.block= glob.glob(join(BASE_DIR, self.path, 'blockindex', '*'))
        self.block=sorted(self.block,key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)])      
        self.block=np.array(self.block)[dataset[1].test_area_idx-1]
       
        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.ones(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset[1].input_labels['testing']]
        self.point_idx=[np.ones(shape=[l.shape[0], 1], dtype=np.float32)
                           for l in dataset[1].input_labels['testing']]
    
    def test(self, model, dataset, num_votes=100):
      
        # Smoothing parameter for votes
        test_smooth = 0.95
        self.dataset=dataset
        # Initialise iterator with validation/test data
        
        self.sess.run(dataset[1].test_init_op)
        self.sess.run(dataset[0].test_init_op_last)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset[1].label_values:
            if label_val not in dataset[1].ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset[1].test_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None
        makedirs(join(test_path, 'val_preds1')) if not exists(join(test_path, 'val_preds1')) else None
        step_id = 0
        epoch_id = 0
        last_min=-0.5
        while  epoch_id < 500:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                      
                       )

                stacked_probs,stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                # print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])  # #(2,num,2)
                
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]   
                                            
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                    self.point_idx[c_i][p_idx] = p_idx.reshape(-1, 1)

                step_id += 1

            except tf.errors.OutOfRangeError:
                
               new_min = np.min(dataset[1].min_possibility['testing'])
               log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)
               print("-----------------" +str(epoch_id)+ "------------------------------")
               if  last_min+1<new_min:
                    last_min+=1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []

                    num_val = len(dataset[1].input_labels['testing'])

                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        preds = dataset[1].label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset[1].input_labels['testing'][i_test]

                        # Confs
                        confusion_list += [confusion_matrix(labels, preds,labels=dataset[1].label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                    positive_classes = np.sum(C, axis=0)
                    true_positive_classes = np.diagonal(C)
                    precision = true_positive_classes[1] / float(positive_classes[1])
                    gt_classes = np.sum(C, axis=1)
                    recall = true_positive_classes[1] / float(gt_classes[1])

                    log_out('precision:{}'.format(precision), self.Log_file)
                    log_out('recall:{}'.format(recall), self.Log_file)
                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_out(s + '\n', self.Log_file)
                        # Project predictions
                    log_out('\nReproject Vote', self.Log_file)
                    proj_probs_list = []
                    

                    for i_val in range(num_val):

                        proj_idx = dataset[1].test_proj[i_val]
                        probs = self.test_probs[i_val][proj_idx, :]
                        proj_probs_list += [probs]     
        
                         
                    # Show vote results
                    log_out('Confusion on full clouds', self.Log_file)
                    confusion_list = []
                       
                    preds_all=[]
                    labels_all=[]
                    for i_test in range(num_val):
                        # Get the predicted labels
                        preds = dataset[1].label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

                        preds_all.extend(preds)
                        
                        # Confusion
                        labels = dataset[1].test_labels[i_test]
                        acc = np.sum(preds == labels) / len(labels)
                        log_out(dataset[1].input_names['testing'][i_test] + ' Acc:' + str(acc), self.Log_file)
                        labels_all.extend(labels)

                        confusion_list += [confusion_matrix(labels, preds,labels=dataset[1].label_values)]
                        name = dataset[1].input_names['testing'][i_test] + '.ply'
                        write_ply(join(test_path, 'val_preds', name), [preds, labels], ['pred', 'label'])
                    preds_all=np.array(preds_all)
                    labels_all=np.array(labels_all)
                    acc_all=np.sum(preds_all==labels_all)/len(labels_all)
                    log_out('Overall Acc:'+str(acc_all),self.Log_file)
                    '''Block precision calculation'''

                    # blockidx_list = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                    #                  '10': [], '11': [],'12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [], '19': [],
                    #                  '20': [],'21': [], '22': [], '23': [], '24': [], '25': [], '26': [], '27': [], '28': [],
                    #                  '29': [], '30': [], '31': [], '32': [], '33': [], '34': [], '35': [], '36': [], '37': [], '38': [],
                    #                  '39': [], '40': []}
                    blockidx_list=[]

                    for  j_val in range(num_val):
                        blockfiles=self.block[j_val]
                        blockfile = glob.glob(join(blockfiles, '*.npy'))
                        blockfile = sorted(blockfile,key=lambda s: [(s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)])
                        block_list=[]
                        for ij_val in range(len(blockfile)):                            
                            index = np.load(blockfile[ij_val])
                            block_list.extend(index)
                        blockidx_list+=[block_list]
                    labels_all1 = []
                    preds_all1=[]
                    for ij_test in range(num_val):
                         # Get the predicted labels
                         preds = dataset[1].label_values[np.argmax(proj_probs_list[ij_test], axis=1)].astype(np.uint8)
                         # Confusion
                         #for k in range(len(blockidx_list[ij_test])):
                         predidx=blockidx_list[ij_test]
                         oneshape=np.where(preds[predidx]==1)[0].shape
                         labels = dataset[1].test_labels[ij_test]
                         labeloneshape=np.where(labels[predidx]==1)[0].shape
                         if (oneshape[0]>0):
                             preds[predidx]=1
                         else:
                             preds[predidx]=0
                         if(labeloneshape[0]>0):
                             labels[predidx]=1
                         else:
                             labels[predidx]=0
                         allpoints = np.array(dataset[1].input_trees['testing'][ij_test].data, copy=False)
                         predidx2=np.where(preds==1)[0]
                         xyz=allpoints[predidx2].reshape(-1,3)
                         np.savetxt(join(test_path, 'val_preds1',str(ij_test)+".txt"),xyz) 
                         predidx2=np.where(preds==0)[0]     
                         xyz=allpoints[predidx2].reshape(-1,3)
                         np.savetxt(join(test_path, 'val_preds1',str(ij_test)+"_1.txt"),xyz)
                         
                         predidx3=np.where(labels==1)[0]
                         xyz=allpoints[predidx3].reshape(-1,3)
                         np.savetxt(join(test_path, 'val_preds1',str(ij_test)+"_label.txt"),xyz)
                         predidx3=np.where(labels==0)[0]
                         xyz=allpoints[predidx3].reshape(-1,3)
                         np.savetxt(join(test_path, 'val_preds1',str(ij_test)+"_label_1.txt"),xyz)
                         
                         acc = np.sum(preds == labels) / len(labels)
                         log_out(dataset[1].input_names['testing'][ij_test] + ' Acc:' + str(acc), self.Log_file)

                         confusion_list += [confusion_matrix(labels, preds,labels=dataset[1].label_values)]
                         name = dataset[1].input_names['testing'][ij_test] + '.ply'
                         write_ply(join(test_path, 'val_preds1',name), [preds, labels], ['pred', 'label'])
                         preds_all1.extend(preds)
                         labels_all1.extend(labels)
                    preds_all1 = np.array(preds_all1)
                    labels_all1 = np.array(labels_all1)
                    acc_all1 = np.sum(preds_all1 == labels_all1) / len(labels_all1)
                    log_out('Overall Acc:' + str(acc_all1), self.Log_file)
                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0)
                    positive_classes = np.sum(C, axis=0)
                    true_positive_classes = np.diagonal(C)
                    precision = true_positive_classes[1] / float(positive_classes[1])
                    gt_classes = np.sum(C, axis=1)
                    recall = true_positive_classes[1] / float(gt_classes[1])

                    log_out('precision:{}'.format(precision), self.Log_file)
                    log_out('recall:{}'.format(recall), self.Log_file)
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                    log_out('-' * len(s), self.Log_file)
                    log_out(s, self.Log_file)
                    log_out('-' * len(s) + '\n', self.Log_file)
                    print('finished \n')
                    self.sess.close()
                    return                      
             
               self.sess.run(dataset[1].test_init_op)
               self.sess.run(dataset[0].test_init_op_last)

               epoch_id += 1
               step_id = 0
               continue

        return

