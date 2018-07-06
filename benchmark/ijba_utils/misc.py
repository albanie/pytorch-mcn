# -*- coding: utf-8 -*-
"""IJB-A misc functiosn and utilities

This code is primarily based on the code of Aruni Roy Chowdhury. The original
code can be found here:
https://github.com/AruniRC/resnet-face-pytorch/blob/master/utils.py
"""
from __future__ import division

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import matplotlib
matplotlib.use('agg')

# --------------------------------------------------------------------------
#   IJB-A helper code
# --------------------------------------------------------------------------

def get_ijba_1_1_metadata(protocol_file):
    metadata  = {}
    template_id = []
    subject_id = []
    img_filename = []
    media_id = []
    sighting_id = []
    bboxes = []

    with open(protocol_file, 'r') as f:
        for line in f.readlines()[1:]:
            line_fields = line.strip().split(',')
            template_id.append(int(line_fields[0]))
            subject_id.append(int(line_fields[1]))
            img_filename.append(line_fields[2])
            media_id.append(int(line_fields[3]))
            sighting_id.append(int(line_fields[4]))
            bboxes.append([float(x) for x in line_fields[6:10]])

    metadata['template_id'] = np.array(template_id)
    metadata['subject_id'] = np.array(subject_id)
    metadata['img_filename'] = np.array(img_filename)
    metadata['media_id'] = np.array(media_id)
    metadata['sighting_id'] = np.array(sighting_id)
    metadata['bboxes'] = np.array(bboxes)
    return metadata


def read_ijba_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split(',')
            pairs.append(pair)
    return np.array(pairs).astype(np.int)
