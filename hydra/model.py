from jax import grad, jit
import numpy as np
import jax.numpy as xp
from tqdm import tqdm
from text_loader import DataLoader  # , onehot_to_string


data = DataLoader()


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
    negexp = xp.exp(-activation)
    softmax = negexp / xp.sum(negexp, axis=0)
    cross_entropy = - xp.sum(target * xp.log(softmax), axis=0)
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
    seq_onehot = xp.eye(data.embed_dim)[:, seq_x]
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
    attention = xp.exp(-attention) / xp.sum(xp.exp(-attention),
                                            axis=1, keepdims=True)

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
    dim_C, dim_K = data.embed_dim, weights_gamma1.shape[0]

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


def transformer_init(dim_K, dim_C):
    """
    Initialize the weights for a transformer layer.
    The layer consists of self-attention with affine output, and two FC layers
    with Layer Norm.

    Arguments:
        dim_K: hidden dimension for transformer and FC
        dim_C: iput and ouput dimension
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
    weights_fc2 = xp.array(0.01 * np.random.randn(data.embed_dim, 4*dim_K))
    # norm
    weights_gamma2 = xp.array(0.01 * np.random.randn(data.embed_dim, 1))
    weights_beta2 = xp.array(0.01 * np.random.randn(data.embed_dim, 1))

    params_fc = [weights_gamma1, weights_beta1,
                 weights_fc1, weights_fc2,
                 weights_gamma2, weights_beta2]

    params = params_attention + params_fc

    return params


@jit
def loss(params, seq):
    """
    This is the function that's differentiated by jax.grad.

    This means it needs to return a scalar.

    Arguments:
        params: list with all parameter tensors
        seq: list with inputs, targets, time codes
    """
    dim_T = 64
    seq_x, seq_y, seq_s = seq
    forward = transformer_forward(params, seq_x, seq_s)
    target = xp.eye(data.embed_dim)[:, seq_y]
    loss = softmax_cross_entropy(forward, target)
    mean_loss = xp.sum(loss, axis=0) / dim_T  # sum over N
    return mean_loss


@jit
def update(params, seq):
    """
    jitted update function adapted from mnist_classifier_fromscratch.py
    Differentiates the function `loss` wrt. it's first argument, `params`, and
    performs a single gradient step.

    TODO: Instead of assuming a fixed list of lists, extend this to arbitarty
    trees of parameters.

    Arguments:
        params: tuple of parameters
        seq: data batch

    Returns:
        new_params: new params
    """
    grads = grad(loss, argnums=0)(params, seq)
    step_size = .005
    new_params = [(w - step_size * dw) for w, dw in zip(params, grads)]
    return new_params


def train_loop():
    """
    Main training function. Defines initial weights, loops over dataset,
    updates weights, plots progress.
    """
    dim_T = 64
    dim_N = 256
    dim_K = 128
    dim_C = data.embed_dim + data.posit_dim  # layer 1 input: tokens + sines
    params = transformer_init(dim_K, dim_C)
    losses = []
    sequences = data.seq_iterator(batch_size=dim_N, seq_len=dim_T,
                                  iters=int(1000))

    pbar = tqdm(enumerate(sequences))
    for i, seq in pbar:
        cost = loss(params, seq) / dim_N
        losses.append(cost)
        params = update(params, seq)
        pbar.set_description("Training Cost %2.6f" % cost)

    print("Completed training, ", i, " final loss", cost,
          "perplexity", xp.exp(cost), "out of", data.embed_dim)
    # print("Some example output:", onehot_to_string(seq[0][:50], data.tokens))


if __name__ == "__main__":
    train_loop()
