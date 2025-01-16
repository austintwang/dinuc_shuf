from time import perf_counter

import numpy as np

from dinuc_shuf import shuffle

class catchtime:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)


def time_shuffle(num_samples=10000, seq_len=2114, alphabet_size=4):
    rng = np.random.default_rng(42)

    seq = rng.choice(alphabet_size, size=(num_samples, seq_len, 1), axis=2)
    seq_ohe = (seq == np.arange(4)[None,None,:]).astype(np.uint8)

    with catchtime() as timer:
        shuffle(seq_ohe, rng=rng, verify=False)


if __name__ == "__main__":
    time_shuffle()
    
    
