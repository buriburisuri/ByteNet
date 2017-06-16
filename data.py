import unicodedata
from data_flow import *


__author__ = 'namju.kim@kakaobrain.com'


# max
_max_len = 250


# pre-calculated for speed-up.
_set_size = 1456501


@sg_ventilator()
def _vent_corpus(opt):

    # load en-de parallel corpus
    with open('asset/data/wmt_fr_en_train.msgpack', 'rb') as f:
        en_codes, de_codes = msgpack.unpackb(f.read())

    # loop for each epoch
    for en, de in zip(en_codes, de_codes):
        # zero-padding
        en += [0] * (_max_len - len(en))
        de += [0] * (_max_len - len(de))
        # return
        yield en, de


class WmtFrEn(object):

    def __init__(self, batch_size=16, set_type=None):

        # load vocabulary
        with open('asset/data/wmt_fr_en_dic.msgpack', 'rb') as f:
            self.index2char, self.char2index = msgpack.unpackb(f.read())

        # set voca size
        self.voca_size = len(self.index2char)
        self.max_len = _max_len

        if set_type == 'train':

            # run ventilator
            _vent_corpus(set_type=set_type)

            # get tf queue
            en_q, fr_q = sg_tf_sinker([tf.sg_intx, tf.sg_intx])

            # create batch queue
            batch_queue = tf.train.shuffle_batch([en_q, fr_q], batch_size, shapes=[(_max_len,), (_max_len,)],
                                                 capacity=batch_size*128, min_after_dequeue=batch_size*64,
                                                 name='WMT_FR_EN_TRAIN')

            # split data
            self.en, self.fr = batch_queue

            # calc total batch count
            self.num_batch = _set_size // batch_size

            # print info
            tf.sg_info('WMT_FR_EN_TRAIN data loaded.(total data=%d, total batch=%d)' % (_set_size, self.num_batch))

    def to_batch(self, sentences):

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            # clean white space
            sent = ' '.join(sentences[i].lower().split())
            # remove punctuation
            sent = ''.join(x for x in sent if unicodedata.category(x) not in ['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
            sentences[i] = [self.char2index[ch] for ch in sent] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        return sentences

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = self.to_str(index, split=False)
            print(u'[%d]' % i + str_)

    def to_str(self, index, split=True):
        str_ = ''
        for ch in index:
            if ch > 1:
                str_ += self.index2char[ch]
            elif ch == 1:  # <EOS>
                break
        if split:
            return str_.split(' ')
        else:
            return str_
