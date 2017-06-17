# -*- coding: utf-8 -*-
import numpy as np
import msgpack
from model import *
from data import WmtFrEn
from nltk.translate import bleu_score


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# inputs
#

# load develop set
with open('asset/data/wmt_fr_en_dev.msgpack', 'rb') as f:
    ground_truth, sources = msgpack.unpackb(f.read())

# load dictionary
data = WmtFrEn()

# place holders
x = tf.placeholder(dtype=tf.sg_intx, shape=(1, data.max_len))
y_in = tf.placeholder(dtype=tf.sg_intx, shape=(1, data.max_len))
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

# run graph for translating
with tf.Session() as sess:
    # init session vars
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/wmt'))

    # loop for all dev set
    hypos, refers = [], []
    for n, (source, target) in enumerate(zip(sources, ground_truth)):

        # initialize character sequence
        pred_prev = np.zeros((1, data.max_len)).astype(np.int32)
        pred = np.zeros((1, data.max_len)).astype(np.int32)

        # generate output sequence
        for i in range(data.max_len):

            # zero-padding
            source += [0] * (data.max_len - len(source))

            # predict character
            out = sess.run(label, {x: [source], y_in: pred_prev})

            # update character sequence
            if i < data.max_len - 1:
                pred_prev[:, i + 1] = out[:, i]
            pred[:, i] = out[:, i]
        pred = pred[0]

        # save result
        hypos.append(data.to_str(pred))
        refers.append(data.to_str(target))

        # print result
        print('(%d/%d) th processed.' % (n, len(sources)))
        print('s: %s' % data.to_str(source, split=False))
        print('t: %s' % data.to_str(target, split=False))
        print('p: %s' % data.to_str(pred, split=False))

    with open('asset/data/wmt_fr_en_res.msgpack', 'wb') as f:
        f.write(msgpack.packb([refers, hypos]))
        print('asset/data/wmt_fr_en_res.msgpack was saved')

    # calc bleu score
    bleu = bleu_score.corpus_bleu(refers, hypos,
                                  smoothing_function=bleu_score.SmoothingFunction().method1) * 100
    print('BLEU:', bleu)
