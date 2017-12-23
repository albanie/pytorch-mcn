# -*- coding: utf-8 -*-
""".
"""

import unittest, torch
import itertools
import numpy as np
from importer import convert_padding

# class TestUM(unittest.TestCase):

    # def test_convert_padding(self):
        # num_orig_ims = 2 ; num_classes = 4

        # batch_size = num_classes * num_orig_ims
        # labels = np.zeros(batch_size)
        # steps = np.linspace(0, batch_size, num_classes + 1, dtype=int)
        # labels[steps[1:-1]] = 1
        # labels = np.cumsum(labels)  # labels = [0,..,0,1,..,1, .., num_labels]

        # rows, cols = pick_rotated_indices(labels)
        # expected_rows = np.array([2, 2, 2, 2,
                                  # 3, 3, 3, 3,
                                  # 4, 4, 4, 4,
                                  # 5, 5, 5, 5,
                                  # 6, 6, 6, 6,
                                  # 7, 7, 7, 7,
                                  # 0, 0, 0, 0,
                                  # 1, 1, 1, 1])
        # expected_cols = np.array([1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0,
                                  # 1, 2, 3, 0])
        # self.assertTrue(np.array_equal(rows, expected_rows), 'row mismatch')
        # self.assertTrue(np.array_equal(cols, expected_cols), 'col mismatch')

# if __name__ == '__main__':
    # unittest.main()
