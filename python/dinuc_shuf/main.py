from typing import Optional

import numpy as np

from .dinuc_shuf import _shuffle

def shuffle(seqs: np.ndarray, rng: Optional[np.random.Generator] = None, verify: bool = True) -> np.ndarray:
    """
    Shuffle the input sequences while preserving dinucleotide composition. 

    Parameters
    ----------
    seqs : np.ndarray
        A three-dimensional array of one-hot-encoded sequences with shape (num_seqs, seq_len, alphabet_size). Will be cast to np.uint8 if not already so.
    rng : Optional[np.random.Generator], optional
        A NumPy random number generator instance. If None, a new default generator instance will be used.
    verify : bool, optional
        If True, the input sequences will be verified to ensure they are one-hot encoded and have the correct shape.

    Returns
    -------
    np.ndarray
        An array of dinucleotide-shuffled sequences with the same shape as the input.
    
    Raises
    ------
    ValueError
        If the input sequences are not three-dimensional or are not one-hot encoded.
    """
    seqs = seqs.astype(np.uint8, copy=False)

    if verify:
        if seqs.ndim != 3:
            raise ValueError("Input sequences must be 3D of shape (num_seqs, seq_len, alphabet_size)")
        
    if seqs.size == 0:
        return seqs
    
    if verify:
        seqs_sum = seqs.sum(axis=2)
        if (seqs_sum != 1).any():
            raise ValueError("Input sequences must be one-hot encoded")
        
    n, l, a = seqs.shape

    if rng is None:
        rng = np.random.default_rng()
    _seed = rng.integers(0, 2**64, dtype=np.uint64)

    shuffled = _shuffle(seqs, _seed)

    rng.bit_generator.advance(n * (l + a))

    return shuffled