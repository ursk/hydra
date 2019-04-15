import jax.numpy as xp
import numpy as np
from text_loader import DataLoader, token_to_string
from matplotlib import pyplot as plt
from layers import transformer_forward


def plot_loss(t_losses, e_losses):
    """
    Create a pdf plot of training curves.

    Arguments:
    t_losses: List of training losses
    e_losses: List of eval losses
    """
    plt.clf()
    plt.plot(t_losses, '.')
    plt.plot(e_losses, 'x')
    plt.title("validation vs. training loss")
    plt.xlabel("iterations / 100")
    plt.ylabel("loss")
    plt.legend(["train", "validation"])
    plt.show()
    plt.savefig("loss.png", bbox_inches='tight')
    plt.ylim([0.8, 1.1])
    plt.savefig("loss.pdf", bbox_inches='tight')

def inspect_output(params, seq, tokens):
    """
    Compute softmax outputs from a given sequence, and decode tokens.
    Print the first 80 characters of each sequence.

    Arguments:
        params: tuple of parameters
        seq: data batch
    """
    seq_x, seq_y, seq_s = seq
    activation = transformer_forward(params, seq_x, seq_s)
    negexp = xp.exp(activation)
    softmax = negexp / xp.sum(negexp, axis=0)
    inp = token_to_string(seq_x[0:80], tokens)
    out = token_to_string(np.argmax(softmax, axis=0)[0:80], tokens)
    print()
    print('"'+inp+'"')
    print('"'+out+'"')
    print()
