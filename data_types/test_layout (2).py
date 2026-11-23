#!/usr/bin/env python3
"""Pure Python layout tests. No MPI install required for these tests."""

import unittest
from layout import block_partition


class TestBlockPartition(unittest.TestCase):
    def test_even(self):
        self.assertEqual(block_partition(8, 4), ([2, 2, 2, 2], [0, 2, 4, 6]))

    def test_uneven(self):
        self.assertEqual(block_partition(7, 4), ([2, 2, 2, 1], [0, 2, 4, 6]))

    def test_zero_local_rows(self):
        self.assertEqual(block_partition(2, 4), ([1, 1, 0, 0], [0, 1, 2, 2]))

    def test_single_rank(self):
        self.assertEqual(block_partition(5000, 1), ([5000], [0]))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            block_partition(-1, 4)
        with self.assertRaises(ValueError):
            block_partition(4, 0)


if __name__ == "__main__":
    unittest.main()
