import codecs
import unicodedata
import msgpack


__author__ = 'namju.kim@kakaobrain.com'


def load_corpus(file_path):

    with codecs.open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def clean_sentence(sentence):

    # make to lowercase and clean white space
    sentence = ' '.join(sentence.lower().split())

    # remove punctuation
    return ''.join(x for x in sentence if unicodedata.category(x) not in ['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])


def make_dictionary(source_path, target_path):

    #
    # load all contents
    #

    print('Now loading corpus for making dictionary from [%s] and [%s] ...' % (source_path, target_path))

    raw_source = load_corpus(source_path)
    raw_target = load_corpus(target_path)

    # code set
    codes_source, codes_target = set(), set()

    for i, (source, target) in enumerate(zip(raw_source, raw_target)):

        # clean sentence
        source = clean_sentence(source)
        target = clean_sentence(target)

        # save character codes
        codes_source = codes_source.union(set([ord(ch) for ch in source]))
        codes_target = codes_target.union(set([ord(ch) for ch in target]))

        if i % 1000 == 0:
            print('(%d th / %d) line was calculated.' % (i, len(raw_source)))

    #
    # make vocabulary
    #

    index2char = ['<EMP>', '<EOS>']    # add <EMP>, <EOS> tokens
    for c in sorted(codes_source.union(codes_target)):
        index2char.append(unichr(c))

    char2index = {}
    for i, b in enumerate(index2char):
        char2index[b] = i

    return index2char, char2index


def make_indices(source_path, target_path, char2index, min_len=0, max_len=250):

    #
    # load all contents
    #

    print('Now loading corpus for making indices from [%s] and [%s] ...' % (source_path, target_path))

    raw_source = load_corpus(source_path)
    raw_target = load_corpus(target_path)

    #
    # Convert characters to indices
    #

    indices_source, indices_target = [], []

    for i, (source, target) in enumerate(zip(raw_source, raw_target)):

        # clean sentence
        source = clean_sentence(source)
        target = clean_sentence(target)

        # find character index
        source_idx, target_idx = [], []

        for ch in source:
            if ch in char2index:
                source_idx += [char2index[ch]]
        for ch in target:
            if ch in char2index:
                target_idx += [char2index[ch]]

        # add <EOS>
        source_idx += [1]
        target_idx += [1]

        # check length
        if min_len <= len(source_idx) <= max_len and \
           min_len <= len(target_idx) <= max_len:

            # save results
            indices_source.append(source_idx)
            indices_target.append(target_idx)

        if i % 1000 == 0:
            print('(%d th / %d) line was converted.' % (i, len(raw_source)))

    return indices_source, indices_target


#
# Do preprocessing and save result
#

if __name__ == '__main__':

    # make dictionary
    index2char, char2index = make_dictionary('asset/data/europarl-v7.fr-en.en', 'asset/data/europarl-v7.fr-en.fr')
    with open('asset/data/wmt_fr_en_dic.msgpack', 'wb') as f:
        f.write(msgpack.packb([index2char, char2index]))
        print('asset/data/wmt_fr_en_dic.msgpack was saved')

    # make train set
    indices_en, indices_fr = make_indices('asset/data/europarl-v7.fr-en.en', 'asset/data/europarl-v7.fr-en.fr',
                                          char2index, min_len=50)
    with open('asset/data/wmt_fr_en_train.msgpack', 'wb') as f:
        f.write(msgpack.packb([indices_en, indices_fr]))
        print('asset/data/wmt_fr_en_train.msgpack was saved')

    # make dev set
    indices_en, indices_fr = make_indices('asset/data/newstest2013.en', 'asset/data/newstest2013.fr', char2index)
    with open('asset/data/wmt_fr_en_dev.msgpack', 'wb') as f:
        f.write(msgpack.packb([indices_en, indices_fr]))
        print('asset/data/wmt_fr_en_dev.msgpack was saved')
