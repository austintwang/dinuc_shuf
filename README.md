# dinuc_shuf

This Python package provides a minimal and efficient implementation for performing dinucleotide shuffles on one-hot-encoded sequences.

Dinucleotide shuffling preserves the dinucleotide (nucleotide pair) frequencies of the input sequence while randomizing the order of the pairs. This is particularly useful for generating random sequences that match the compositional properties of the original input.

To ensure a uniform random sample from all possible shuffles, the algorithm leverages the rank-one-update Kirchhoff matrix method described by [Colburn et al.](https://doi.org/10.1006/jagm.1996.0014) for sampling random arborescences, combined with a random Eulerian walk on the dinucleotide transition graph. The core algorithm is implemented in Rust for performance, with Python bindings for easy integration.

This package is lightweight, requiring only a single dependency on Numpy.

## Installation

To install the package from PyPI, run:

```bash
pip install dinuc-shuf
```

## Usage

```python
import numpy as np
from dinuc_shuf import shuffle

SEQ_ALPHABET = np.array(["A","C","G","T"], dtype="S1")

def one_hot_encode(sequence, dtype=np.uint8):
    sequence = sequence.upper()
    seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
    one_hot = (seq_chararray[:,None] == SEQ_ALPHABET[None,:]).astype(dtype)

    return one_hot

def one_hot_decode(one_hot):
    return SEQ_ALPHABET[one_hot.argmax(axis=1)].tobytes().decode('UTF-8')

sequence = "ACCCACGATGATG"
one_hot_sequence = one_hot_encode(sequence)
shuffled_one_hot = shuffle(one_hot_sequence[None,:,:])
shuffled = one_hot_decode(shuffled_one_hot[0,:,:])

print(shuffled) # Output: "ACATGATGACCCG"
```

## API Reference

A full API reference is available [here](https://austintwang.github.io/dinuc_shuf/).
