# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from tqdm import tqdm
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import my_train_one_epoch
from training.dataloader import CellDataLoader
logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset in ['bbbc021', 'rxrx1', 'cpg0000']:
        args.num_tasks = distributed_mode.get_world_size()
        num_tasks = args.num_tasks
        args.global_rank = distributed_mode.get_rank()
        global_rank = args.global_rank
        logger.info("Intializing DataLoader")
        datamodule = CellDataLoader(args)
        data_loader_train = datamodule.train_dataloader()
        data_loader_test = datamodule.test_dataloader()
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")


    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )
    model.to(device)

    model_without_ddp = model

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )
    if args.use_initial in [1, 2]:
        logger.info("Generating From Control Image!!!!!")
    else:
        logger.info("Generating From Random Noise Image!!!!!")
    
    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = my_train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            
            eval_stats = eval_model(
                model,
                data_loader_test,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
                datamodule=datamodule,
                use_initial=args.use_initial,
                interpolate=args.interpolate,
            )
            try:
                log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
                logger.info(log_stats)
            except:
                pass
        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def load_yaml_config(yaml_path):
    yaml_path = "configs/" + yaml_path + ".yaml"
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    yaml_config = load_yaml_config(args.config)
    args_dict = vars(args)
    args_dict.update(yaml_config)
    args = SimpleNamespace(**args_dict)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
