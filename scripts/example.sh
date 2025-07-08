#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
python train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=2 \
    --eval_frequency=10 \
    --epochs=10 \
    --class_drop_prob=1.0 \
    --cfg_scale=0.0 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=5120 \
    --use_initial=2 \
    --noise_level=0.5 \
    --output_dir=outputs/example \
