import ipdb
import numpy as np

# Text processing helpers
def read_text_file(fname='moby10b.txt', lines=1000):
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
    Identify the unique tokens in text, return them in
    """
    return list(set(text))


def string_to_onehot(all_text, tokens):
    """
    represent a string using onehot encoding from tokens
    """
    to_dict = dict([[j,i] for i,j in enumerate(tokens)])
    text_as_array = np.array([to_dict[i] for i in all_text])
    return text_as_array

def onehot_to_string(numbers, tokens):
    """
    convert numerical represenatation back to text
    """
    rev_dict = dict([[i,j] for i,j in enumerate(tokens)])
    string_array = [rev_dict[i] for i in list(numbers)]
    return "".join(string_array)

def sinusoids(seq_size=200):
    """
    Positional encoding features: Sines waves at different frequencies.
    Generate enough points to fill one sequence
    """
    time_line = np.arange(seq_size)
    frequencies = [0.1, 0.2, 0.3]
    sines = np.vstack([np.sin(f*time_line) for f in frequencies])
    return sines

def test_tokenizer():
    """
    Read text, convert to tokens, convert back and make sure we get back
    the original text.
    """
    all_text = read_text_file()
    tokens = get_tokens(all_text)
    text_as_array = string_to_onehot(all_text[:100], tokens)
    string_array = onehot_to_string(text_as_array[:100], tokens)
    # turn position in the list into an index.
    # seq = string_to_onehot(all_text, tokens)
    assert string_array == all_text[:100], "text conversion fail"

class DataLoader(object):
    """
    Moby Dick character data loader.
    Generate sequences at batch size 1.

    TODO: Extend to support mini-batches
    """
    def __init__(self):
        self.all_text = read_text_file()
        self.tokens = get_tokens(self.all_text)
        self.text_len = len(self.all_text)
        self.embed_dim = len(self.tokens)


    def seq_iterator(self, seq_size=64, iters=1000):
        """
        returns an iterator that produces sequences from Moby Dick.
        Each consists of a input sequence and a target sequence,
        which is the input shifted by one.

        Arguments:
            seq_size: Number of characters in each seq
            iters: Number of seqes to train for. If the text is not
                   long enough, loop over.
        """
        text_as_array = string_to_onehot(self.all_text, self.tokens)

        for i in range(iters):
            index = (i*seq_size) % (self.text_len-seq_size-1)
            seq_x = text_as_array[index:index+seq_size]
            seq_y = text_as_array[index+1:index+seq_size+1]
            seq_s = sinusoids(seq_size)
            yield seq_x, seq_y, seq_s


