import jax.numpy as xp
import numpy as np
from jax.scipy.special import logsumexp


def relu(x):
    """ stax relu"""
    return xp.maximum(x, 0.)


def layernorm(inputs):
    """
    Layer normalization across features, which are assumed to be the first dim.
    Behaves as an activation function, no learned parameters.

    Arguments:
        inputs: Activations of shape [C, NT]
    """
    mean = xp.mean(inputs, axis=1, keepdims=True)
    meaned = inputs - mean
    variance = xp.mean(meaned**2, axis=1, keepdims=True)
    outputs = meaned / xp.sqrt(variance + 0.00001)
    return outputs


def softmax_cross_entropy(activation, target):
    """
    softmax_cross_entropy.

    Arguments:
        activations: network prediction. Dimensions are [C, NT]
        target: one-hot targets

    Returns:
        cross_entropy: Loss vector over NT

    TODO: Register custom gradient to avoid numerical instability
    """
    # unnormalized = xp.exp(activation-activation.max(axis=0, keepdims=True))
    # softmax = unnormalized / xp.sum(unnormalized, axis=0)
    logsoftmax = activation - logsumexp(activation, axis=0, keepdims=True)
    cross_entropy = - xp.sum(target * logsoftmax, axis=0)
    return cross_entropy


# class Transformer(object):  # maybe later. keep it simple for JAX for now.
def transformer_forward(params, seq_x, seq_s):
    """
    Wrapper layer for self-attention plus FC stack

    Arguments:
        params: Tuple or list with all parameters
        seq_x: Input sequences (1xNT)
        seq_s: Position encodings (3xNT)
    """
    # unpack parameters
    params_attention = params[0:4]
    params_fc = params[4:10]
    # package up data seq
    dim_em = 81
    seq_onehot = xp.eye(dim_em)[:, seq_x]
    inputs = xp.array(xp.vstack((seq_onehot, seq_s)))  # CN
    # run the two parts of the layer
    attention = selfattention_forward(params_attention, inputs)
    inputs = fcblock_forward(params_fc, inputs, attention)
    return inputs


def selfattention_forward(params_attention, inputs):
    """
    Simple transformer model. Single-headed self-attention.

    Arguments:
        params_attention: Tuple with SA weight matrices
        inputs: Inputs after one-hot encoding and stacking with position
                encodings

    Returns:
        attention: Activation from self-attention layer
    """
    # unpack parameters
    (weights_k, weights_v, weights_q, weights_o) = params_attention

    # first multiply the query against the keys, [KNT] = [KC] [CNT]
    dim_A, dim_K, dim_N, dim_T = 4, 128, 256, 64
    dim_act = (dim_A, dim_K // dim_A, dim_N, dim_T)  # hidden state per head.
    act_K = xp.dot(weights_k, inputs).reshape(dim_act)
    act_V = xp.dot(weights_v, inputs).reshape(dim_act)
    act_Q = xp.dot(weights_q, inputs).reshape(dim_act)
    # reduce K, outer product T, for each N, A. Result is [ATTN]
    attention = xp.einsum('ainj,aink->ajkn', act_Q, act_K)
    # softmax over sequence (T) dimension
    unnormalized = xp.exp(attention-attention.max(axis=1, keepdims=True))
    attention = unnormalized / xp.sum(unnormalized, axis=1, keepdims=True)

    # then compute the weighted values [TTN][KNT]=[KNT]
    attention = xp.einsum('aijn,akni->aknj', attention, act_V)
    attention = attention.reshape((dim_K, dim_N * dim_T))

    # add affine output layer on top
    attention = xp.dot(weights_o, attention)
    return attention


def fcblock_forward(params_fc, inputs, attention):
    """
    Transformer block part 2: FC-normalization stack.

    Arguments:
        params_fc: tuple with FC parameters
        inputs: inputs for skip connection
        attention: activations from the SA layer

    Returns:
        inputs: activations
    """
    (weights_gamma1, weights_beta1,
        weights_fc1, weights_fc2,
        weights_gamma2, weights_beta2) = params_fc
    dim_K = weights_gamma1.shape[0]
    dim_C = inputs.shape[0]  # TODO: check this accounts for time encoding

    # skip connection
    inputs = (attention + inputs) if dim_C == dim_K else attention

    inputs = layernorm(inputs)
    inputs = weights_gamma1 * inputs + weights_beta1

    # FC stack that forms the second part of the block
    activation = xp.dot(weights_fc1, inputs)
    activation = relu(activation)
    activation = xp.dot(weights_fc2, activation)

    # skip connection
    inputs = (activation + inputs) if dim_C == dim_K else activation

    inputs = layernorm(inputs)
    inputs = weights_gamma2 * inputs + weights_beta2
    return inputs


def transformer_init(dim_K, dim_C, dim_em):
    """
    Initialize the weights for a transformer layer.
    The layer consists of self-attention with affine output, and two FC layers
    with Layer Norm.

    Arguments:
        dim_K: hidden dimension for transformer and FC
        dim_C: iput and ouput dimension

    Returns:
        params: list of arrays with randomly initalized parameters
        grads: list of arrays with zero initialized gradients
    """

    # define weights and package up params list.

    weights_k = xp.array(0.01 * np.random.randn(dim_K, dim_C))
    weights_v = xp.array(0.01 * np.random.randn(dim_K, dim_C))
    weights_q = xp.array(0.01 * np.random.randn(dim_K, dim_C))
    weights_o = xp.array(0.01 * np.random.randn(dim_K, dim_K))
    params_attention = [weights_k, weights_v, weights_q, weights_o]

    # norm
    weights_gamma1 = xp.array(0.01 * np.random.randn(dim_K, 1))
    weights_beta1 = xp.array(0.01 * np.random.randn(dim_K, 1))
    # FC
    weights_fc1 = xp.array(0.01 * np.random.randn(4*dim_K, dim_K))
    weights_fc2 = xp.array(0.01 * np.random.randn(dim_em, 4*dim_K))
    # norm
    weights_gamma2 = xp.array(0.01 * np.random.randn(dim_em, 1))
    weights_beta2 = xp.array(0.01 * np.random.randn(dim_em, 1))

    params_fc = [weights_gamma1, weights_beta1,
                 weights_fc1, weights_fc2,
                 weights_gamma2, weights_beta2]

    params = params_attention + params_fc

    grads =[xp.zeros(p.shape) for p in params]

    return params, grads
