from jax import grad, jit, disable_jit
import numpy as np
import jax.numpy as xp
import matplotlib.pyplot as plt
from tqdm import tqdm
from text_loader import DataLoader
import ipdb

# The transformer model
#         #         #         #         #         #         #         #       80

data = DataLoader()

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
    Simple transformer model. Single-headed self-attention.

    TODO: Pass in batch size and stuff

    Arguments:
        params: Tuple with weigh matrices
        seq_x: Input sequences (1xNT)
        seq_s: Position encodings (3xNT)
    """
    # unpack parameters
    weights_k, weights_v, weights_q, weights_fc = params
    dim_K, dim_C, = weights_k.shape
    dim_NT = seq_x.shape[0]
    dim_N = 256
    dim_T = int(dim_NT / dim_N)

    # package up data seq
    seq_onehot = xp.eye(data.embed_dim)[:,seq_x]
    inputs = xp.array(xp.vstack((seq_onehot, seq_s)))  # CN

    # first multiply the query against the keys
    act_K = xp.dot(weights_k, inputs).reshape((dim_K, dim_N, dim_T))  # [KNT] = [KC] [CNT]
    act_V = xp.dot(weights_v, inputs).reshape((dim_K, dim_N, dim_T))
    act_Q = xp.dot(weights_q, inputs).reshape((dim_K, dim_N, dim_T))
    # attention = xp.dot(act_Q, act_K.transpose())
    # reduce K, outer T, loop N. Result is [TTN]
    attention = xp.einsum('inj,ink->jkn', act_Q, act_K)
    attention = xp.exp(-attention) / xp.sum(xp.exp(-attention), axis=1)

    # then compute the weighted values
    # inputs = xp.dot(attention, act_V)
    inputs = xp.einsum('ijn,kni->knj', attention, act_V)  # [TTN][KNT]=[KNT]
    inputs = inputs.reshape((dim_K, dim_N * dim_T))  # pack NT back for FC

    # add a fully connected FC layer on top
    inputs = xp.dot(weights_fc, inputs)

    return inputs

def loss(params, seq):
    """
    run forward pass, compute loss

    Arguments:
        params: list with all parameter tensors
        seq: list with inputs, targets, time codes
    """
    seq_x, seq_y, seq_s = seq
    forward = transformer_forward(params, seq_x, seq_s)
    target = xp.eye(data.embed_dim)[:,seq_y]
    loss = softmax_cross_entropy(forward, target)
    mean_loss = xp.mean(loss, axis=0)
    return mean_loss

@jit
def update(params, seq):
    """
    jitted update function from mnist_classifier_fromscratch.py
    """
    grads = grad(loss)(params, seq)
    step_size = .01
    return [(w - step_size * dw) for w, dw in zip(params, grads)]

def train_loop():
    """
    Main training function. Defines initial weights, loops over dataset,
    updates weights, plots progress.
    """
    # define weights and package up params list.
    C = data.embed_dim + 3 # tokens + sinusoids
    K = 128  # hidden dimension for transformer and FC
    weights_k = xp.array(0.01 * np.random.randn(K, C))
    weights_v = xp.array(0.01 * np.random.randn(K, C))
    weights_q = xp.array(0.01 * np.random.randn(K, C))
    weights_fc = xp.array(0.01 * np.random.randn(data.embed_dim, K))
    params = [weights_k, weights_v, weights_q, weights_fc]

    losses = []
    sequences = data.seq_iterator(batch_size=256, seq_len=64, iters=int(100))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()

    for i, seq in tqdm(enumerate(sequences)):
        # print("BATCH ", i, ":", onehot_to_string(seq[0], data.tokens))
        cost = loss(params, seq)
        losses.append(cost)
        params = update(params, seq)

        if not (i % 100):
            ax.clear()
            ax.plot(losses)
            fig.canvas.draw()


if __name__ == "__main__":
    train_loop()
