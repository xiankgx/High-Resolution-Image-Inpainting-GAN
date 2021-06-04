#!/bin/bash

python train.py \
    --epochs 40 \
    --lr_g 0.0001 \
    --batch_size 11 \
    --lambda_perceptual 100 \
    --lambda_l1 300 \
    --baseroot ../datasets/places2/data_large \
    --mask_type free_form \
    --imgsize 512 \
    --load_name_g models/deepfillv2_WGAN_epoch1_batch5000.pth \
    --load_name_d models/discriminator_WGAN_epoch1_batch5000.pth
