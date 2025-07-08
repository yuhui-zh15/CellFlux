#!/bin/bash
python eval_fid.py --model_name cellflux --dataset bbbc021 --num_to_cal 5120 \
        --image_root /path/to/your/bbbc021/generated_images

# python eval_fid.py --model_name cellflux --dataset rxrx1 \
#         --image_root /path/to/your/rxrx1/generated_images

# python eval_fid.py --model_name cellflux --dataset cpg0000 \
#         --image_root /path/to/your/cpg0000/generated_images