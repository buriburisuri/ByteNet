# -*- coding: utf-8 -*-
import sugartensor as tf
from sklearn.cross_validation import train_test_split
import numpy as np


__author__ = 'buriburisuri@gmail.com'


class ComTransTrain(object):

    def __init__(self, batch_size=32, name='train'):

        # load train corpus
        sources, targets = self._load_corpus(mode='train')

        # to constant tensor
        source = tf.convert_to_tensor(sources)
        target = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([source, target])

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                             num_threads=32, capacity=batch_size*64,
                                             min_after_dequeue=batch_size*32, name=name)

        # split data
        self.source, self.target = batch_queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))

    def _load_corpus(self, mode='train'):

        # load en-fr parallel corpus
        from nltk.corpus import comtrans
        als = comtrans.aligned_sents('alignment-en-fr.txt')

        # make character-level parallel corpus
        all_byte, sources, targets = [], [], []
        for al in als:
            src = [ord(ch) for ch in ' '.join(al.words)]  # source language byte stream
            tgt = [ord(ch) for ch in ' '.join(al.mots)]  # target language byte stream
            sources.append(src)
            targets.append(tgt)
            all_byte.extend(src + tgt)

        # make vocabulary
        self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i
        self.voca_size = len(self.index2byte)

        # remove small length sentence
        for s, t in zip(sources, targets):
            if len(s) < 30 or len(t) < 30:
                sources.remove(s)
                targets.remove(t)

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sources)):
            sources[i] = [self.byte2index[ch] for ch in sources[i]] + [1]
            targets[i] = [self.byte2index[ch] for ch in targets[i]] + [1]

        # zero-padding
        self.max_len = 300
        for i in range(len(sources)):
            sources[i] += [0] * (self.max_len - len(sources[i]))
            targets[i] += [0] * (self.max_len - len(targets[i]))

        # split data
        if mode == 'train':
            sources, _, targets, _ \
                = train_test_split(sources, targets, test_size=0.2, random_state=27521)
        elif mode == 'test':
            _, sources, _, targets \
                = train_test_split(sources, targets, test_size=0.2, random_state=27521)

        return sources, targets

    def print_index(self, indices):
        for index in indices:
            print ''.join([unichr(self.index2byte[i]) for i in index if i > 1])


class ComTransTest(ComTransTrain):

    def __init__(self, batch_size=32, name='test'):

        # load train corpus
        sources, targets = self._load_corpus(mode='test')

        # to constant tensor
        source = tf.convert_to_tensor(sources)
        target = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([source, target])

        # create batch queue
        batch_queue = tf.train.batch([source, target], batch_size,
                                     num_threads=32, capacity=batch_size*64,
                                     allow_smaller_final_batch=True, name=name)

        # split data
        self.source, self.target = batch_queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Test data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))
