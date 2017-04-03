import sugartensor as tf


__author__ = 'namju.kim@kakaobrain.com'


#
# hyper parameters
#

latent_dim = 400   # hidden layer dimension
num_blocks = 3     # dilated blocks


# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):

    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False, is_first=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    with tf.sg_context(name='block_%d_%d' % (opt.block, opt.rate)):

        # reduce dimension
        input_ = (tensor
                  .sg_bypass(act='relu', ln=(not opt.is_first), name='bypass')  # do not
                  .sg_conv1d(size=1, dim=in_dim/2, act='relu', ln=True, name='conv_in'))

        # 1xk conv dilated
        out = (input_
               .sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', ln=True, name='aconv'))

        # dimension recover and residual connection
        out = out.sg_conv1d(size=1, dim=in_dim, name='conv_out') + tensor

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)


#
# encode graph ( atrous convolution )
#
def encode(x):

    with tf.sg_context(name='encoder'):
        res = x
        # loop dilated conv block
        for i in range(num_blocks):
            res = (res
                   .sg_res_block(size=5, block=i, rate=1, is_first=True)
                   .sg_res_block(size=5, block=i, rate=2)
                   .sg_res_block(size=5, block=i, rate=4)
                   .sg_res_block(size=5, block=i, rate=8)
                   .sg_res_block(size=5, block=i, rate=16))

    return res


#
# decode graph ( causal convolution )
#

def decode(x, voca_size):

    with tf.sg_context(name='decoder'):
        res = x
        # loop dilated causal conv block
        for i in range(num_blocks):
            res = (res
                   .sg_res_block(size=3, block=i, rate=1, causal=True, is_first=True)
                   .sg_res_block(size=3, block=i, rate=2, causal=True)
                   .sg_res_block(size=3, block=i, rate=4, causal=True)
                   .sg_res_block(size=3, block=i, rate=8, causal=True)
                   .sg_res_block(size=3, block=i, rate=16, causal=True))

        # final fully convolution layer for softmax
        res = res.sg_conv1d(size=1, dim=voca_size, name='conv_final')

    return res

