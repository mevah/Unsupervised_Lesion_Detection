#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:25:16 2020

@author: himeva
"""

import numpy as np
import tensorflow as tf
from utils.batches import get_batches, get_batches_brats, tile, plot_batch
import os, time

import tensorlayer as tl

def main(data_path=None, model_path=None, ckpt_path=None):

    batch_size = 8
    image_w=240
    image_h=240
    c=1


    batches = get_batches([batch_size, image_w, image_h, c], data_path, train=True, box_factor=0)

	###======================== HYPER-PARAMETERS ============================###
    batch_size = 8
    n_epoch = 200

    nw=240
    nh=240
    with tf.device('/cpu:0'):
        saver = tf.train.import_meta_graph(model_path)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            ###======================== DEFINE MODEL =======================###
            #t_image = tf.placeholder('float32', [batch_size, nw, nh, 1], name='input_image')
	            ## labels are 0, 1, 2, 3
            #t_seg_init = tf.placeholder('int32', [batch_size, nw, nh,1], name='target_segment')
            #t_seg =tf.one_hot(t_seg_init, depth=6, axis=-1)
            #t_seg = tf.reshape(t_seg, [batch_size, nw, nh, 6])
	            ##category (Brats or HCP)
            #t_cat_init = tf.placeholder('int64', [batch_size],  name='image_category')
            #t_cat = tf.cast(t_cat_init, tf.int32)
	            ##mask for reconstruction
            #t_mask = tf.placeholder('float32', [batch_size, nw, nh, 1], name='mask')



            #saver = tf.train.import_meta_graph(model_path)

            saver.restore(sess,ckpt_path)
            graph= tf.get_default_graph()
            op = sess.graph.get_operations()
            #print('here are the operations')
            #print([m for m in op][1])
            #print('here are the tensors')
            #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
            output_tensor = graph.get_tensor_by_name("u_net_1/conv4_2/Relu:0")

    tf.global_variables_initializer()

    total_out=[]
    for epoch in range(0, n_epoch+1):
        print('epoch no:' , epoch+1)
        batch_out=[]
        images, labels, category = next(batches)[:3]
        b_labels=labels.astype('int')
        b_images=images[:,:,:,:]
        b_images.shape = (batch_size, nw, nh, 1)
        print(b_images.shape)
        catlist=[]

        for p in range(batch_size):
            if category[p][0]==1:
                cat = 1
            else:
                cat = 0
            catlist.append(cat)
        b_category=np.asarray(catlist)

        mask = np.zeros(np.shape(b_images))
        mask[b_category==1] = (labels[b_category==1] ==1)

        encoder_output = sess.run(output_tensor, {'input_image:0': b_images, 'target_segment:0':
                                                                 b_labels, 'image_category:0': b_category,
                                                              'mask:0':mask})
        batch_out.append(np.array(encoder_output))
        total_out.append(batch_out)
        np.save("encoderconv4healthyrec_output_list.npy", total_out)
        #print(encoder_output)

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_path', type=str, default='all')
    parser.add_argument('--model_path', type=str, default='all')
    parser.add_argument('--ckpt_path', type=str, default='all')
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.ckpt_path)



'''python calculate_RSA.py --data_path /scratch_net/biwidl201/himeva/MasterThesis-Meva/data/unilabel-train_t1c_noedema.hdf5 --model_path /scratch_net/biwidl201/himeva/MasterThesis-Meva/rec_checkpoint/u_net_rec_{}.ckpt-10000.meta --ckpt_path /scratch_net/biwidl201/himeva/MasterThesis-Meva/rec_checkpoint/'''
