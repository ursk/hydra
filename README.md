# hydra
Simple Transformer model implemented in JAX.

<img width="1045" alt="Screen Shot 2019-03-24 at 15 15 23" src="https://user-images.githubusercontent.com/1203292/54886672-deda3e00-4e47-11e9-956e-b88d03634e04.png">

### Model description

Single layer transformer with K=128 hidden units, A=1 attention head, S=64
sequence length. It's trained with a batch size of N=256.

For position encoding, a set of 3 sinusoids is used.

### Dataset
Moby Dick is used for training data, encoded in a character-wise fashion. This
is handled by `text_loader.py` which provides a `DataLoader` class with a
`seq_iterator` function, that will yield a tuple of source sequences, target
sequences and position encodings.
