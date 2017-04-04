import sugartensor as tf
from model import *
from data import ComTrans


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = ComTrans(batch_size=batch_size)

# source, target sentence
x, y = data.source, data.target
# shift target for training source
y_in = tf.concat([tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]], axis=1)
# vocabulary size
voca_size = data.voca_size

# make embedding matrix for source and target
emb_x = tf.sg_emb(name='emb_x', voca_size=voca_size, dim=latent_dim)
emb_y = tf.sg_emb(name='emb_y', voca_size=voca_size, dim=latent_dim)

# latent from embed table
z_x = x.sg_lookup(emb=emb_x)
z_y = y_in.sg_lookup(emb=emb_y)

# encode graph ( atrous convolution )
enc = encode(z_x)

# concat merge target source
enc = enc.sg_concat(target=z_y)

# decode graph ( causal convolution )
dec = decode(enc, voca_size)

# cross entropy loss with logit and mask
loss = dec.sg_ce(target=y, mask=True)

# train
tf.sg_train(loss=loss, log_interval=30, lr=0.0001, ep_size=data.num_batch, max_ep=20)

