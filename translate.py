# -*- coding: utf-8 -*-
import numpy as np
from model import *
from data import WmtFrEn


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 10

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = WmtFrEn()

# place holders
x = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, data.max_len))
y_in = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, data.max_len))
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

# greedy search policy
label = dec.sg_argmax()


#
# translate
#

# sample french sentences for source language
sources = [
    u"Et pareil phénomène ne devrait pas occuper nos débats ?",
    u"Mais nous devons les aider sur la question de la formation .",
    u"Les videurs de sociétés sont punis .",
    u"Après cette période , ces échantillons ont été analysés et les résultats illustrent bien la quantité de dioxine émise au cours des mois écoulés .",
    u"Merci beaucoup , Madame la Commissaire .",
    u"Le Zimbabwe a beaucoup à gagner de l ' accord de partenariat et a un urgent besoin d ' aide et d ' allégement de la dette .",
    u"Le gouvernement travailliste de Grande-Bretagne a également des raisons d ' être fier de ses performances .",
    u"La plupart d' entre nous n' a pas l' intention de se vanter des 3 millions d' euros .",
    u"Si le Conseil avait travaillé aussi vite que ne l' a fait M. Brok , nous serions effectivement bien plus avancés .",
    u"Le deuxième thème important concerne la question de la gestion des contingents tarifaires ."
]

# to batch form
sources = data.to_batch(sources)

# run graph for translating
with tf.Session() as sess:
    # init session vars
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/wmt'))

    # initialize character sequence
    pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
    pred = np.zeros((batch_size, data.max_len)).astype(np.int32)

    # generate output sequence
    for i in range(data.max_len):
        # predict character
        out = sess.run(label, {x: sources, y_in: pred_prev})
        # update character sequence
        if i < data.max_len - 1:
            pred_prev[:, i + 1] = out[:, i]
        pred[:, i] = out[:, i]

# print result
print('\nsources : --------------')
data.print_index(sources)
print('\ntargets : --------------')
data.print_index(pred)
