from model import *
from data import WmtFrEn, ComTrans


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

# command line argument
tf.sg_arg_def(bs=(16, "Batch size. The default is 16."))
tf.sg_arg_def(lr=(0.0001, "Learning rate. The default is 0.0001."))
tf.sg_arg_def(ep=(20, "Maximum epoch. The default is 20."))
tf.sg_arg_def(li=(60, "Logging interval. The default is 60."))
tf.sg_arg_def(corpus=('wmt', 'Corpus to use. The default is "wmt".'))

#
# inputs
#

if tf.sg_arg().corpus == 'wmt':

    # WMT en_de parallel corpus for train
    train = WmtFrEn(batch_size=tf.sg_arg().bs)
    # vocabulary size
    voca_size = train.voca_size

    # source, target sentence
    x, y = train.fr, train.en
elif tf.sg_arg().corpus == 'comtrans':

    # ComTrans parallel corpus input tensor
    train = ComTrans(batch_size=tf.sg_arg().bs)

    # source, target sentence
    x, y = train.source, train.target

# shift target for training source
y_in = tf.concat([tf.zeros((tf.sg_arg().bs, 1), tf.sg_intx), y[:, :-1]], axis=1)

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
loss = dec.sg_ce(target=y, mask=True, name='ce')

#
# train
#
tf.sg_train(loss=loss, lr=tf.sg_arg().lr,
            ep_size=train.num_batch, max_ep=tf.sg_arg().ep,
            log_interval=tf.sg_arg().li, save_dir='asset/train/%s' % tf.sg_arg().corpus)

