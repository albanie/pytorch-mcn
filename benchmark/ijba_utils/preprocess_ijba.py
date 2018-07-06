# -*- coding: utf-8 -*-
"""Preprocessing script for IJBA data

Extract faces from IJBA raw images using the ground truth bounding boxes
included with the dataset.
"""

import argparse
import numpy as np
from tqdm import tqdm
import misc
import matplotlib
matplotlib.use('agg')
import os
import matplotlib.pyplot as plt
import PIL.Image
from os.path import join as pjoin

try:
    from zsvision.zs_iterm import zs_dispFig
except:
    pass

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # prevent PIL from choking on bad inputs

def xywh2xyxy(bbox):
    """convert bounding box format from [xmin, ymin, width, height] (used
    by IJB-A) to [xmin, ymin, xmax, ymax] (used by PIL for cropping)

    Args:
       bbox (ndarray): 1x4 array of box coords

    Returns:
       bbox (ndarray): 1x4 array of transformed box coords
    """
    assert bbox.size == 4, 'unexpected number of coordinates'
    bbox[2:] = bbox[2:] + bbox[:2]
    return bbox

def crop_images_via_gt_boxes(protocol_dir, output_dir, src_dir, fold_id, vis,
                             num_partitions, partition_id):
    """Crop the full images supplied with the IJB-A using the hand-annotated
    bounding boxes provided with the data.  The resulting cropped faces are
    saved in a new directory, where each face corresponds to a single
    `sighting_id` in the dataset notation. The bounding boxes are stored
    in metadata csv files in [xmin,ymin,width,height] format.

    Args:
        protocol_dir (str): the directory containing the IJBA protocol data
        output_dir (str): the path to th edirectory where the cropped images
          will be stored.
        fold_id (int): fold identifier used for IJBA evaluation.
        vis (bool): whether to visualise the crops (requires zsvision module)
        num_partitions (int): the number of ways to partion the data to
          split over multilpe worker.
        partition_id (int): the identifier of the partition to be processed.
    """
    pairs_path = pjoin(protocol_dir, 'split{}'.format(fold_id),
                          'verify_comparisons_{}.csv'.format(fold_id))
    pairs = misc.read_ijba_pairs(pairs_path)
    protocol_file = pjoin(protocol_dir, 'split{}'.format(fold_id),
                          'verify_metadata_{}.csv'.format(fold_id))
    metadata = misc.get_ijba_1_1_metadata(protocol_file) # dict
    assert np.all(np.unique(pairs) == np.unique(metadata['template_id']))  # sanity-check

    # only process unique sighting ids to avoid duplicating work
    _,keep = np.unique(metadata['sighting_id'], return_index=True)

    # save crops as <sighting_id.jpg>
    src_paths = np.array([pjoin(src_dir, x) for
                                 x in metadata['img_filename'][keep]])
    splits = np.array_split(np.arange(len(src_paths)), num_partitions)
    current_split = splits[partition_id]
    src_paths = src_paths[current_split]
    dest_paths = [pjoin(output_dir, '{}.jpg'.format(x))
                     for x in metadata['sighting_id'][keep][current_split]]
    bboxes = metadata['bboxes'][keep][current_split]

    for ii, src_path in tqdm(enumerate(src_paths)):
        dest_path = dest_paths[ii]

        if os.path.exists(dest_path):
            print('found cropped image at {}, skipping...'.format(dest_path))
            continue

        im = PIL.Image.open(src_path)
        if not im.mode == 'RGB':
            im = im.convert('RGB')

        if vis:
            plt.imshow(np.array(im))
            zs_dispFig()

        bbox = bboxes[ii]
        bbox = xywh2xyxy(bbox) # convert xmin,ymin,width,height to XYXY
        im_cropped = im.crop(bbox)

        if vis:
            plt.imshow(np.array(im_cropped))
            zs_dispFig()

        im_cropped.save(dest_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                default='/users/albanie/data/datasets/ijba',
                help='original IJB-A data (should contain `img` folder)')
    parser.add_argument('--protocol_dir',
                default='/users/albanie/data/datasets/ijba/protocol/IJB-A_11')
    parser.add_argument('--output_dir',
                default='/users/albanie/data/datasets/ijba/cropped')
    parser.add_argument('--vis', help='visualise crops as a sanity check',
                action='store_true', default=False)
    parser.add_argument('--fold_id', default=1)
    parser.add_argument('--num_partitions', default=1, type=int,
                        help='split up processing to spread over cores')
    parser.add_argument('--partition_id', default=0, type=int,
                        help='index of the current partition')
    args = parser.parse_args()
    crop_images_via_gt_boxes(args.protocol_dir, args.output_dir, args.src_dir,
                             args.fold_id, args.vis, args.num_partitions,
                             args.partition_id)
