#!/bin/bash
# This script is a lazy way to launch a bunch of python
# processes to do face cropping.

# setup anaconda
source activate c2

num_workers=3

for worker_id in `seq 0 $((num_workers-1))`
do
    echo "launching face cropping process on worker ${worker_id}"
    ipy preprocess_ijba.py -- \
        --num_partitions $num_workers \
        --partition_id ${worker_id}
done
