#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:20:16 2020

@author: himeva
"""
    

import numpy as np
import matplotlib.pyplot as plt
o_mean= np.load('VanillaVAE/BraTS2018/VanillaVAE0.00010.0010.010.11/reconstructions/reconstr_mean20.00.00.00.00.npy')
o_std= np.load('VanillaVAE/BraTS2018/VanillaVAE0.00010.0010.010.11/reconstructions/reconstr_std20.00.00.00.00.npy')
for i in range(o_mean.shape[0]):
    image_mean= o_mean[i,:,:,:].squeeze()
    image_std=  o_std[i,:,:,:].squeeze()
    image= np.random.normal(image_mean, image_std)
    plt.imshow((image_mean))
    plt.show()
    
# o_mean= np.load('VanillaVAE/BraTS2018/VanillaVAE0.00010.0010.010.11/reconstructions/reconstr_mean31.55.00.00.00.npy')
# o_std= np.load('VanillaVAE/BraTS2018/VanillaVAE0.00010.0010.010.11/reconstructions/reconstr_std31.55.00.00.00.npy')
# for i in range(o_mean.shape[0]):
#     image_mean= o_mean[i,:,:,:].squeeze()
#     image_std=  o_std[i,:,:,:].squeeze()
#     image= np.random.normal(image_mean, image_std)
#     plt.imshow(image)
#     plt.show() 