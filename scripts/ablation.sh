#!/bin/bash
python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_008 \
    --num_epochs 15 \
    --pos_state False