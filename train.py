# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from data import ComTransTrain


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
latent_dim = 64   # hidden layer dimension

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = ComTransTrain(batch_size=batch_size)

# source, target sentence
x, y = data.source, data.target

# make embedding matrix for source and target
emb_x = tf.sg_emb(name='emb_x', voca_size=data.voca_size, dim=latent_dim)
emb_y = tf.sg_emb(name='emb_y', voca_size=data.voca_size, dim=latent_dim)

# encode graph
with tf.sg_context(name='encode', act='relu', bn=True):
    enc = (x
           .sg_lookup(emb=emb_x)
           .sg_aconv1d(rate=1)
           .sg_aconv1d(rate=2)
           .sg_aconv1d(rate=4)
           .sg_aconv1d(rate=8)
           .sg_aconv1d(rate=16)
           .sg_aconv1d(rate=1)
           .sg_aconv1d(rate=2)
           .sg_aconv1d(rate=4)
           .sg_aconv1d(rate=8)
           .sg_aconv1d(rate=16))




tf.sg_print(enc)

##
