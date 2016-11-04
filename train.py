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

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = ComTransTrain(batch_size=batch_size)

# source, target sentence
x, y = data.source, data.target


