#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
--dataset=bbbc021 \
--batch_size=32 \
--accum_iter=1 \
--eval_frequency=100 \
--epochs=3000 \
--class_drop_prob=0.2 \
--cfg_scale=0.2 \
--compute_fid \
--ode_method heun2 \
--ode_options '{"nfe": 50}' \
--use_ema \
--edm_schedule \
--skewed_timesteps \
--eval_only \
--resume= \
--start_epoch 99 \
--fid_samples=5120 \
--use_initial=1 \
--output_dir= \
--save_fid_samples \
--noise_level=1.0 \
--config=bbbc021_all \
--interpolate \
--use_initial \