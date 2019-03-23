# hydra
Simple Transformer model implemented in JAX.

### Model description

Single layer transformer with K=128 hidden units, A=1 attention head, S=64
sequence length. It's trained with a batch size of N=256.

For position encoding, a set of 3 sinusoids is used.

### Dataset
Moby Dick is used for training data, encoded in a character-wise fashion. This
is handled by `text_loader.py` which provides a `DataLoader` class with a
`seq_iterator` function, that will yield a tuple of source sequences, target
sequences and position encodings.
