import unittest

import numpy as np
from scipy.stats import norm

from dinuc_shuf import shuffle

SEQ_ALPHABET = np.array(["A","C","G","T"], dtype="S1")

def one_hot_encode(sequence, dtype=np.uint8):
    sequence = sequence.upper()

    seq_chararray = np.frombuffer(sequence.encode('UTF-8'), dtype='S1')
    one_hot = (seq_chararray[:,None] == SEQ_ALPHABET[None,:]).astype(dtype)

    return one_hot


def one_hot_decode(one_hot):
    return SEQ_ALPHABET[one_hot.argmax(axis=1)].tobytes().decode('UTF-8')


def test_factory(seq_str, num_shuffles_true=None):
    class TestDinucShuffle(unittest.TestCase):
        def setUp(self):
            seq = one_hot_encode(seq_str)
            self.seq = seq
            self.seq_adj = seq[:-1,:].T @ seq[1:,:]

            self.rng = np.random.default_rng(42)

            self.num_shuffles_true = num_shuffles_true

        def test_composition(self):
            shuffled = shuffle(self.seq[None,:,:], rng=self.rng)
            shuffled_adj = shuffled[0,:-1,:].T @ shuffled[0,1:,:]
            np.testing.assert_array_equal(self.seq_adj, shuffled_adj)

        def test_uniformity(self):
            n = 100000
            seq_expanded = np.repeat(self.seq[None,:,:], n, axis=0)
            shuffled = shuffle(seq_expanded, rng=self.rng)
            decoded_shuffled = [one_hot_decode(i) for i in shuffled]
            counts = {}
            for i in decoded_shuffled:
                counts.setdefault(i, 0) 
                counts[i] += 1    

            if len(counts) < 2:
                self.skipTest("Fewer than 2 unique shuffles")
            
            p = 1 / len(counts)
            denom = np.sqrt(n * p * (1 - p))
            expected = n * p

            z_scores = np.array([(k - expected) / denom for k in counts.values()])
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores))) 
            corrected_p = min(np.min(p_values) * n, 1)
            
            self.assertGreater(corrected_p, 1 - 1e-7)

        def test_coverage(self):
            if self.num_shuffles_true is None:
                self.skipTest("True number of unique sequences not provided")

            n = 1000
            seq_expanded = np.repeat(self.seq[None,:,:], n, axis=0)
            shuffled = shuffle(seq_expanded, rng=self.rng)
            decoded_shuffled = [one_hot_decode(i) for i in shuffled]

            unique_seqs = set(decoded_shuffled)
            unique_seqs.discard("")
            num_unique_seqs = len(unique_seqs)

            self.assertEqual(num_unique_seqs, self.num_shuffles_true)

    return TestDinucShuffle


empty = test_factory("", 0)
A = test_factory("A", 1)
TT = test_factory("TT", 1)
ACGT = test_factory("ACGT", 1)
ACGCACGG = test_factory("ACGCACGG")
ACCCACGATGATA = test_factory("ACCCACGATGATA", 72)
ACCCACGATGATG = test_factory("ACCCACGATGATG", 27)


if __name__ == "__main__":
    unittest.main()



           