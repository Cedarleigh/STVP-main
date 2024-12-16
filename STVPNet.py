import gc
from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
tf.reset_default_graph()
def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset[1].flat_inputs   # #Second iterator input
        flat_inputs_last=dataset[0].flat_inputs_last  # #First iterator input
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        # with tf.variable_scope('inputs_view', reuse=tf.AUTO_REUSE):
        #         self.inputs_view = dict()
        #         num_layers = self.config.num_layers
        #         self.inputs_view['xyz'] = flat_inputs_view[:num_layers]
        #         self.inputs_view['neigh_idx'] = flat_inputs_view[num_layers: 2 * num_layers]
        #         self.inputs_view['sub_idx'] = flat_inputs_view[2 * num_layers:3 * num_layers]
        #         self.inputs_view['interp_idx'] = flat_inputs_view[3 * num_layers:4 * num_layers]
        #         #            self.inputs_view['interp_idx'] = flat_inputs_view[3 * num_layers:4 * num_layers]
        #         self.inputs_view['features'] = flat_inputs_view[4 * num_layers]
        #         self.inputs_view['input_inds'] = flat_inputs_view[4 * num_layers + 1]
        #         self.inputs_view['cloud_inds'] = flat_inputs_view[4 * num_layers + 2]

        with tf.variable_scope('inputs_view',reuse=tf.AUTO_REUSE):
            self.inputs_view = dict()
            num_layers = self.config.num_layers
            self.inputs_view['xyz'] = flat_inputs[5*num_layers+4:6*num_layers+4]
            self.inputs_view['neigh_idx'] = flat_inputs[6*num_layers+4: 7* num_layers+4]
            self.inputs_view['sub_idx'] = flat_inputs[7 * num_layers+4:8 * num_layers+4]
            self.inputs_view['interp_idx'] = flat_inputs[8 * num_layers+4:9 * num_layers+4]
#            self.inputs_view['interp_idx'] = flat_inputs_view[3 * num_layers:4 * num_layers]
            self.inputs_view['features'] = flat_inputs[9 * num_layers+4]
            self.inputs_view['input_inds'] = flat_inputs[10 * num_layers + 1]
            self.inputs_view['cloud_inds'] = flat_inputs[10 * num_layers + 2]
                
        with tf.variable_scope('inputs_last',reuse=tf.AUTO_REUSE):
            self.inputs_last = dict()
            num_layers = self.config.num_layers
            self.inputs_last['xyz'] = flat_inputs_last[:num_layers]
            self.inputs_last['neigh_idx'] = flat_inputs_last[num_layers: 2 * num_layers]
            self.inputs_last['sub_idx'] = flat_inputs_last[2 * num_layers:3 * num_layers]
            self.inputs_last['interp_idx'] = flat_inputs_last[3 * num_layers:4 * num_layers]
            self.inputs_last['features'] = flat_inputs_last[4 * num_layers]
            self.inputs_last['input_inds'] = flat_inputs_last[4 * num_layers + 1]
            self.inputs_last['cloud_inds'] = flat_inputs_last[4 * num_layers + 2]
            self.inputs_last['rgb']=flat_inputs_last[4 * num_layers + 3:5*num_layers+3]

        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['rgb']= flat_inputs[4 * num_layers + 4:5*num_layers+4]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
        
            self.class_weights = DP.get_class_weights(dataset[1].name)
            self.Log_file = open('log_train_' + '.txt', 'a')
            

        """Feature extraction and processing of different layers of the model 
           (Slayer, Tlayer, F2onelayers and Fallayers)"""

        with tf.variable_scope('Slayer', reuse=tf.AUTO_REUSE):
            t0=time.time()
            self.feature_spa,self.f_encoder_list=self.inference(self.inputs,self.is_training)
            print('Slayer done in {:.1f}s'.format(time.time() - t0))
        with tf.variable_scope('Tlayer', reuse=tf.AUTO_REUSE):
            t0=time.time()
            self.feature_tem=self.inference_temporal(self.inputs,self.inputs_last,self.is_training)
            print('Tlayer done in {:.1f}s'.format(time.time() - t0))
        with tf.variable_scope('F2onelayers', reuse=tf.AUTO_REUSE):
            t0=time.time()
            self.f_sa = self.F2one(self.is_training)
            print('F2onelayers done in {:.1f}s'.format(time.time() - t0))
        with tf.variable_scope('Falllayers', reuse=tf.AUTO_REUSE):
            t0=time.time()
            self.logits=self.Fall(self.is_training)
            print('Falllayers done in {:.1f}s'.format(time.time() - t0))
        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)
        
        # optimizer
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # results
        with tf.variable_scope('results', reuse=tf.AUTO_REUSE):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)
            # Scalar logging during training
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        
#        b=self.sess.run(self.labels,{self.is_training:False})
    def inference_temporal(self, inputs,inputs_last,is_training):
        """
        Process the temporal features of the current frame and the previous frame and calculate the temporal saliency features.

        parameter:
        - inputs: input data of the current frame, including features and spatial information.
        - inputs_last: input data of the previous frame, including features and spatial information.
        - is_training: Boolean value indicating whether the model is in training mode.

        return:
        - feature_tem: Processed temporal feature representation that captures the saliency changes between the current frame and the previous frame.

        """
        d_out = self.config.d_out
#        feature = inputs['features']
        feature_last=inputs_last['features']
#        feature = tf.layers.dense(feature, 8, activation=None,name='fc0')
        feature_last=tf.layers.dense(feature_last, 8, activation=None,name='fc0_last')
#        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature_last = tf.nn.leaky_relu(tf.layers.batch_normalization(feature_last, -1, 0.99, 1e-6, training=is_training))
#        feature = tf.expand_dims(feature, axis=2)
        feature_last=tf.expand_dims(feature_last,axis=2)
        # ###########################Encoder############################
        featchange_list = []
        # Traverse all layers to extract features
        for i in range(self.config.num_layers):
            if i==0:  # The first layer directly obtains features from f_encoder_list
                feature = self.f_encoder_list[i+1]
                featchange_list.append(self.f_encoder_list[i])
            else:
            # Other layers process features using dilated_res_block and reduce the number of features by random sampling
                feature_encoder=self.dilated_res_block(featchange_list[i], inputs['xyz'][i],inputs['neigh_idx'][i], d_out[i],
                                                     'Encoder_layer_change_' + str(i), is_training)
                feature = self.random_sample(feature_encoder, inputs['sub_idx'][i])

            # Process the features of the previous frame
            f_encoder_i_last = self.dilated_res_block(feature_last, inputs_last['xyz'][i],inputs_last['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_last_' + str(i), is_training) #,inputs_last['rgb'][i]
            #f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            f_sampled_i_last = self.random_sample(f_encoder_i_last, inputs_last['sub_idx'][i])
            feature_last=f_sampled_i_last

            # Compare the features of the current frame and the previous frame to obtain the temporal saliency mask
            featchange_i=self.compare(feature,feature_last,d_out[i]*2,i,is_training)
            featchange_list.append(featchange_i)

        feature_tem = helper_tf_util.conv2d(featchange_list[-1],featchange_list[-1].get_shape()[3].value, [1, 1],
                                       'decoder_0',
                                        [1, 1], 'VALID', True, is_training)
        feature_tem=self.decoder(feature_tem,featchange_list,self.inputs['interp_idx'],2,'temporal',is_training)  # 特征处理
        feature_tem=tf.squeeze(feature_tem,[2])

        return feature_tem

    def compare(self,feature_now, feature_last,d_out,layer,is_training):
        """
        Compare the changes in features of the current frame (feature_now) and the previous frame (feature_last) to calculate temporal saliency.

         parameter:
         - feature_now: feature representation of the current frame.
        - feature_last: feature representation of the previous frame.
        - d_out: The output dimension of each layer.
        - layer: The currently processed layer number, used to adjust the parameters of pooling.
        - is_training: Boolean value indicating whether the model is in training mode.

        return:
        - featchange: the saliency adjustment result of the current frame feature, that is, the temporal saliency feature.
        """
        if layer==4:
            g_now=helper_tf_util.max_pool2d(feature_now,[self.config.num_points/(4**(layer+1)/2),1], scope='now-maxpooling',padding='VALID')
            g_last=helper_tf_util.max_pool2d(feature_last,[self.config.num_points/(4**(layer+1)/2),1], scope='last-maxpooling',padding='VALID')
        else:    
            g_now=helper_tf_util.max_pool2d(feature_now,[self.config.num_points/(4**(layer+1)),1], scope='now-maxpooling',padding='VALID')
            g_last=helper_tf_util.max_pool2d(feature_last,[self.config.num_points/(4**(layer+1)),1], scope='last-maxpooling',padding='VALID')
        g_concat=tf.concat([g_now,g_last], axis=3)
        score=helper_tf_util.conv2d(g_concat,self.config.d_out[layer]*2,[1,1],'mlp_'+str(d_out),[1,1],'VALID',True,is_training)
        score=1/(1+tf.exp(score))
        featchange=feature_now*score
        return featchange
       
    def inference(self, inputs,is_training):
        """Extracting spatial features"""
        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None,name='fc0_last')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i],inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_last_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)


        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                       'decoder_0_last',
                                        [1, 1], 'VALID', True, is_training)
        feature=self.decoder(feature,f_encoder_list,self.inputs['interp_idx'],2,'spacial',is_training)
        feature=tf.squeeze(feature,[2])
        return feature,f_encoder_list

    # ###########################Decoder############################
    def decoder(self,feat,f_list,interp,num_classes,name,is_training):
        f_decoder_list = []
        for j in range(self.config.num_layers):
            '''inputs['interp_idx']'''
            f_interp_i = self.nearest_interpolation(feat, interp[-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_list[-j - 2], f_interp_i], axis=3),
                                                          f_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          name+'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feat = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], name+'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1],name+'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp2')
        f_out= helper_tf_util.conv2d(f_layer_drop,num_classes, [1, 1], name+'fc3', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)

        return f_out

    def F2one(self,is_training):
                           
        """Feature fusion and saliency detection module"""
        f_concat=tf.concat([self.feature_tem,self.feature_spa], axis=2)
        f_concat=tf.expand_dims(f_concat,axis=2)
        f_sa = self.att_pooling(f_concat,2, 'F2one' + 'att_pooling_3', is_training)
        f_sa=helper_tf_util.conv2d(f_sa,2,[1,1],'fsa',[1,1],'VALID',True,is_training)
#        f_sa1=helper_tf_util.avg_pool2d(f_sa,[self.config.num_points,1], scope='avgpooling3',padding='VALID')
#        f_tile= tf.tile(f_sa1,[1,self.config.num_points,1,1])
        f_sa=tf.squeeze(f_sa,[2])
        return f_sa

    def Fall2(self,is_training):
        f_vout=self.inputs_view['features']
        f_vout=f_vout[:,:,4:6]
        f_sa=self.f_sa #self.f_sa
        f_vout=tf.expand_dims(f_vout, axis=2)
        f_vout1=helper_tf_util.conv2d(f_vout,2,[1,1],'fvout',[1,1],'VALID',True,is_training)

        f_vout=tf.squeeze(f_vout1,[2])
        f_out=f_sa*f_vout
        return f_out

    def Fall(self,is_training):
        """Fusion view features"""
        f_vout=self.inputs_view['features']
        f_vout=f_vout[:,:,4:6]
        # fvout_mean=tf.reduce_mean(f_vout,axis=1)
        # fvout_mean=tf.expand_dims(fvout_mean, axis=1)
        # f_vout=tf.tile(fvout_mean,[1,self.config.num_points,1])
        f_sa=self.f_sa
        f_concat=tf.concat([f_vout,f_sa], axis=2)
        f_concat=tf.expand_dims(f_concat,axis=2)
        f_out = self.att_pooling(f_concat,2, 'Fall2' + 'att_pooling_3', is_training)
        f_out=tf.reshape(f_out,[-1,self.config.num_points,2])
        return f_out
    
    def train(self, dataset):
        """Model training function"""
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        
        self.sess.run(dataset[0].train_init_op_last)
        self.sess.run(dataset[1].train_init_op)       

       
        inputs=dataset
#        dataset=dataset[1]
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step) ##(40960,7) logits
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:
                m_iou = self.evaluate(inputs)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)
                tf.reset_default_graph()
                tf.keras.backend.clear_session()
                gc.collect()

                self.training_epoch += 1
                
                self.sess.run(inputs[0].train_init_op_last)
                self.sess.run(inputs[1].train_init_op)

                # Update learning rate

                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
       
        self.sess.run(dataset[0].val_init_op_last)
        self.sess.run(dataset[1].val_init_op)

        
        
        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid,labels=(np.arange(0, self.config.num_classes, 1)))#np.arange(0, self.config.num_classes, 1)
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n]+0.000000000000000001)
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)
        
        precision=true_positive_classes[1]/float(positive_classes[1])
        recall=true_positive_classes[1]/float(gt_classes[1])
        
        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('precision:{}'.format(precision), self.Log_file)
        log_out('recall:{}'.format(recall), self.Log_file)
        # log_out('loss:{}'.format(l_out),self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou 

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + '_mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
    


