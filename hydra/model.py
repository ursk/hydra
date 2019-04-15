from jax import grad, jit
import jax.numpy as xp
import numpy as np
from tqdm import tqdm
from text_loader import DataLoader
from utils import inspect_output, plot_loss
from layers import (relu, layernorm, softmax_cross_entropy,
                    transformer_forward, selfattention_forward,
                    fcblock_forward, transformer_init)

data = DataLoader()



@jit
def loss(params, seq):
    """
    This is the function that's differentiated by jax.grad.

    This means it needs to return a scalar.

    Arguments:
        params: list with all parameter tensors
        seq: list with inputs, targets, time codes. Each sequence is of
             length NT
    """
    dim_N = 256
    seq_x, seq_y, seq_s = seq
    forward = transformer_forward(params, seq_x, seq_s)
    target = xp.eye(data.embed_dim)[:, seq_y]
    loss = softmax_cross_entropy(forward, target)  # sum over classes
    mean_loss = xp.sum(loss, axis=0) / dim_N  # sum over T, mean over N
    return mean_loss


@jit
def update(params, updates, seq, step_size = .02):
    """
    jitted update function adapted from mnist_classifier_fromscratch.py
    Differentiates the function `loss` wrt. it's first argument, `params`, and
    performs a single gradient step.

    TODO: Instead of assuming a fixed list of lists, extend this to arbitarty
    trees of parameters.

    Arguments:
        params: list of parameters
        updates: list of momentum variables
        seq: data batch

    Returns:
        new_params: new params
    """
    momentum = 0.9
    grads = grad(loss, argnums=0)(params, seq)
    updates = [(momentum * u + (1 - momentum) * g)
        for u, g in zip(updates, grads)]
    new_params = [(w - step_size * dw) for w, dw in zip(params, updates)]
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
    params, grads = transformer_init(dim_K, dim_C, data.embed_dim)
    t_losses, e_losses = [], []
    sequences = data.seq_iterator(batch_size=dim_N, seq_len=dim_T,
                                  iters=int(100000))
    val_seq = data.validation_set(batch_size=dim_N, seq_len=dim_T)
    pbar = tqdm(enumerate(sequences))
    for i, seq in pbar:
        if not i % 100:
            train_cost = loss(params, seq) / dim_T  # loss per token
            eval_cost = loss(params, val_seq) / dim_T
            t_losses.append(train_cost)
            e_losses.append(eval_cost)
        if not i % 1000:
            inspect_output(params, seq, data.tokens)
            plot_loss(t_losses, e_losses)
        step_size = .1 if i < 500000 else 0.01
        params = update(params, grads, seq, step_size)
        pbar.set_description("Training %2.3f eval %2.3f"
                             % (train_cost, eval_cost))

    train_cost = loss(params, seq) / dim_T
    print("Completed training. Final training loss %2.2f perplexity %2.2f / %d"
          % (train_cost, xp.exp(train_cost), data.embed_dim))
    eval_cost = loss(params, val_seq) / dim_T
    print("Completed training. Final validation loss %2.2f perplexity %2.2f / %d"
          % (eval_cost, xp.exp(eval_cost), data.embed_dim))

    # print("Some example output:", onehot_to_string(seq[0][:50], data.tokens))


if __name__ == "__main__":
    train_loop()
