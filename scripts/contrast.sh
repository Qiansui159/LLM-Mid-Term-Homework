#!/bin/bash
python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_002 \
    --num_epochs 15 \
    --num_heads 4


python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_003 \
    --num_epochs 15 \
    --num_heads 16


python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_004 \
    --num_epochs 15 \
    --num_layers 4


python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_005 \
    --num_epochs 15 \
    --num_layers 8


python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_006 \
    --num_epochs 15 \
    --d_model 256


python ../src/train.py \
    --data_dir ../src/data/cmn.txt \
    --results_dir ../results/run_007 \
    --num_epochs 15 \
    --d_model 768