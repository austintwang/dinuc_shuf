### Note: Run with RAYON_NUM_THREADS=1 for a fair comparison

from time import perf_counter

import numpy as np
import torch

from dinuc_shuf import shuffle
from tangermeme.ersatz import dinucleotide_shuffle
import seqpro as sp

SEQ_ALPHABET = np.array(["A","C","G","T"], dtype="S1")

class catchtime:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'{self.name}: {self.time:.4f} seconds'
        print(self.readout)


def time_shuffle(num_samples=10000, seq_len=2114, alphabet_size=4):
    rng = np.random.default_rng(42)

    seq = rng.choice(alphabet_size, size=(num_samples, seq_len, 1), axis=2)
    seq_ohe = (seq == np.arange(4)[None,None,:]).astype(np.uint8)

    with catchtime("dinuc_shuf") as timer:
        shuffle(seq_ohe, rng=rng, verify=False)


def time_shuffle_seqpro(num_samples=10000, seq_len=2114, alphabet_size=4):
    rng = np.random.default_rng(42)

    seq = rng.choice(alphabet_size, size=(num_samples, seq_len, 1), axis=2)
    seq_ohe = (seq == np.arange(4)[None,None,:]).astype(np.uint8)
    seq_chars = SEQ_ALPHABET[seq_ohe.argmax(axis=-1)]

    with catchtime("seqpro") as timer:
        sp.k_shuffle(seq_chars, k=2, length_axis=1, seed=1234)


def time_shuffle_tangermeme(num_samples=10000, seq_len=2114, alphabet_size=4):
    rng = np.random.default_rng(42)

    seq = rng.choice(alphabet_size, size=(num_samples, seq_len, 1), axis=2)
    seq_ohe = (seq == np.arange(4)[None,None,:]).astype(np.uint8)
    seq_ohe = torch.from_numpy(seq_ohe).swapaxes(1, 2)

    with catchtime("tangermeme") as timer:
        dinucleotide_shuffle(seq_ohe, n=1)


if __name__ == "__main__":
    time_shuffle()
    time_shuffle_seqpro()
    time_shuffle_tangermeme()
    
    
