#!/bin/bash

#Note: there is -parallel for multiple gpu training.
# 1. hr
python classifier_training.py \
    --model R32_C10 \
    --lr 1e-1 \
    --cv_dir checkpoint_hr \
    --batch_size 128 \
    --img_size 32 \
    --mode hr \
    --max_epochs 2 \
    --test_interval 2

python classifier_training.py \
    --model R32_C10 \
    --lr 1e-1 \
    --cv_dir checkpoint_lr \
    --batch_size 128 \
    --img_size 32 \
    --mode lr \
    --max_epochs 2 \
    --test_interval 2

# 2.
Note: low gpu mem so reduced batch size by /16


# Find the latest checkpoint
latest_checkpoint_hr=$(ls checkpoint_hr/ | grep -E 'ckpt_E_[0-9]+_A_[0-9]+\.[0-9]+' | sort -V | tail -n 1)

python pretrain.py \
    --model R32_C10 \
    --lr 1e-4 \
    --cv_dir checkpoint_rl \
    --ckpt_hr_cl checkpoint_hr/"$latest_checkpoint_hr" \
    --batch_size 64 \
    --penalty -0.5 \
    --max_epochs 2 \
    --test_interval 2

# 3.

latest_checkpoint_rl=$(ls checkpoint_rl/ | grep -E 'ckpt_E_[0-9]+_A_[0-9]+\.[0-9]+' | sort -V | tail -n 1)

python finetune.py \
    --model R32_C10 \
    --lr 1e-4 \
    --cv_dir checkpoint_ft \
    --batch_size 128 \
    --load checkpoint_rl/"$latest_checkpoint_rl" \
    --ckpt_hr_cl checkpoint_hr/"$latest_checkpoint_hr" \
    --penalty -10 \
    --max_epochs 2 \
    --test_interval 2

# 4.

latest_checkpoint_lr=$(ls checkpoint_lr/ | grep -E 'ckpt_E_[0-9]+_A_[0-9]+\.[0-9]+' | sort -V | tail -n 1)

python finetune2stream.py \
    --model R32_C10 \
    --lr 1e-4 \
    --cv_dir checkpoint_ft2_stream \
    --batch_size 128 \
    --load checkpoint_rl/"$latest_checkpoint_rl" \
    --ckpt_hr_cl checkpoint_hr/"$latest_checkpoint_hr" \
    --ckpt_lr_cl checkpoint_lr/"$latest_checkpoint_lr" \
    --penalty -10 \
    --max_epochs 2 \
    --test_interval 2
    