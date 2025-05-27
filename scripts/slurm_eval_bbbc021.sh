python submitit_train.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --batch_size=32 \
    --accum_iter=1 \
    --eval_frequency=10 \
    --epochs=3000 \
    --class_drop_prob=0.2 \
    --cfg_scale=0.2 \
    --compute_fid \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --fid_samples=30720 \
    --job_dir=/share/pi/syyeung/yuhuiz/Cell/CellFlow/outputs/0227_eval_bbbc \
    --shared_dir=/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/shared/ \
    --use_initial=2 \
    --eval_only \
    --noise_level=1.0 \
    --save_fid_samples \
    --resume=/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250121_1550_bbbc_noise1.0_drop0.2_cfg0.2/checkpoint-99.pth \
    --start_epoch 99 \
    --ngpus=4 \


# bbbc: /share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250125_1141_bbbc_noise1.0_drop0.2_cfg0.2_prob_0.5/checkpoint-99.pth
# rxrx1: /share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250125_1144_rxrx1_noise1.0_drop0.2_cfg0.2_prob0.5/checkpoint-29.pth
# cpg0000: /share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250125_1754_cpg0000_noise1.0_drop0.2_cfg0.2_prob0.5/checkpoint-79.pth

# rxrx1_old: /share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250121_0227_rxrx1_noise1.0/checkpoint-19.pth