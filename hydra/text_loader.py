import numpy as np


# Text processing helpers
def read_text_file(fname, lines):
    """
    Read text file from disk, return as one long string.

    Arguments:
        fname: path / name of the text file to read from
        lines: number of lines after which to stop reading.
               Set this to -1 to read the entire file.
    """
    with open(fname, 'r') as f:
        raw_lines = [line for line in f]
    return "".join(raw_lines[:lines]).replace('\n', ' ')


def get_tokens(text):
    """
    Identify the unique tokens in text, return them in a
    list. For Moby Dick there are 80 characters + the special '@' character.

    Arguments:
        text: string
    """
    return ['@'] + list(set(text))


def string_to_token(all_text, tokens):
    """
    represent a string using tokens
    """
    to_dict = dict([[j, i] for i, j in enumerate(tokens)])
    text_as_array = np.array([to_dict[i] for i in all_text])
    return text_as_array


def token_to_string(numbers, tokens):
    """
    convert numerical represenatation back to text
    """
    rev_dict = dict([[i, j] for i, j in enumerate(tokens)])
    string_array = [rev_dict[i] for i in list(numbers)]
    return "".join(string_array)


def sinusoids(stride, seq_len):
    """
    Positional encoding features: Sines waves at different frequencies.
    Generate enough points to fill one sequence. 64 is pi/2 as the base.
    """
    time_line = np.arange(stride) * (np.pi / 2) / seq_len
    frequencies = [1, 2, 5, 10, 15]
    sines = np.vstack([np.sin(f*time_line) for f in frequencies])
    return sines


def test_tokenizer():
    """
    Read text, convert to tokens, convert back and make sure we get back
    the original text.
    """
    all_text = read_text_file(fname='moby10b.txt', lines=1000)
    tokens = get_tokens(all_text)
    text_as_array = string_to_token(all_text[:100], tokens)
    string_array = token_to_string(text_as_array[:100], tokens)
    # turn position in the list into an index.
    # seq = string_to_token(all_text, tokens)
    assert string_array == all_text[:100], "text conversion fail"


class DataLoader(object):
    """
    Moby Dick character data loader.
    Generate sequences at batch size 1.

    TODO: Extend to support mini-batches
    """
    def __init__(self, fname='moby10b.txt'):
        self.all_text = read_text_file(fname, lines=-1)
        self.tokens = get_tokens(self.all_text)
        self.text_len = len(self.all_text)
        self.embed_dim = len(self.tokens)
        self.posit_dim = 5

    def seq_iterator(self, batch_size=1, seq_len=64, iters=1000):
        """
        returns an iterator that produces sequences from Moby Dick. Each
        consists of a input sequence and a target sequence, which is the input
        shifted by one. Naming convention: Sequence T, batch N

        Note that the sequences are one-hot encoded after moving to device, not
        here.

        Arguments:
            batch_size: Number of sequences in one batch [N]
            seq_len: Number of characters in each seq [T]
            iters: Number of seqes to train for. If the text is not
                   long enough, loop over.

        Yields:
            seq_x: Input sequence, [1,T*N] format
            seq_y: Target sequence (source seq shifted by one)
            seq_s: Position encodings (three sinusoids)
        """
        text_as_array = string_to_token(self.all_text, self.tokens)
        stride = seq_len * batch_size
        seq_s = sinusoids(stride, seq_len)
        for i in range(iters):
            deletions = np.random.permutation(
                np.arange(stride))[0:int(stride/10)]
            index = (i*stride) % (self.text_len-stride-1)
            seq_y = text_as_array[index:index+stride]
            seq_x = seq_y.copy()
            seq_x[deletions] = 0  # randomly selected junk token
            yield seq_x, seq_y, seq_s

    def validation_set(self, batch_size=1, seq_len=64):
        """
        Construct a single batch validation set from a separate text file.
        Rather than randomly corrupting 10% of tokens, corrupt the last
        tokens in each sequence in the batch so we can easily check the output.
        """
        eval_file = 'bartleby.txt'
        eval_text = read_text_file(eval_file, lines=-1)
        eval_array = string_to_token(eval_text, self.tokens)

        stride = seq_len * batch_size
        seq_s = sinusoids(stride, seq_len)
        seq_y = eval_array[0:stride]
        seq_x = seq_y.copy()
        for i in range(6):
            seq_x[seq_len-i:stride+seq_len-i:seq_len] = 0  # 6 out of 64

        return seq_x, seq_y, seq_s
