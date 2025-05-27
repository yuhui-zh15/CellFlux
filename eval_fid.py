import json
import os
from types import SimpleNamespace
import yaml
from training.data_utils import convert_6ch_to_3ch, convert_5ch_to_3ch
from training.dataloader import CellDataLoader_Eval
import torch
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

model_name = 'impa'
dataset = 'bbbc021'
yaml_path = f"config/eval_{dataset}.yaml"

if model_name == 'phendiff':
    #### PhenDiff ######
    if 'rxrx1' in yaml_path:
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/PhenDiff/experiments/project_name/run_name/2025-01-23/01-24-35/linear_interp_custom_guidance_inverted_start/DDIM/test'
    else:
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/PhenDiff/experiments/project_name/run_name/2025-01-20/07-53-30/linear_interp_custom_guidance_inverted_start/DDIM/test'
elif model_name == 'impa':
    ##### IMPA ######
    if dataset == 'bbbc021':
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/IMPA_reproduce/outputs/bbbc021_all_iter_ctrl/epoch-200/fid_samples'
    else:
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/IMPA/outputs/fid_samples'
elif model_name == 'cellflow':
    # ##### CellFlow ######
    if 'rxrx1' in yaml_path:
        # Use old rxrx1 model without noise prob
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_rxrx1_100_class_cfg0.0/fid_samples/epoch-20'
        # synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_rxrx1_noise1.0/fid_samples'
    elif 'cpg0000' in yaml_path:
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_cpg0000_100_class_cfg0.2/fid_samples/epoch-80'
    elif dataset == 'bbbc_ood':
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_bbbc_ood/fid_samples/epoch-100'
    else:
        # different batch
        synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_bbbc_different_batch/fid_samples/epoch-100'
        # # no condition
        # synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_bbbc_no_condition/fid_samples/epoch-100'
        # # no cfg
        # synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_bbbc_no_cfg/fid_samples/epoch-100'
        # # normal
        # synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/20250125_1141_bbbc_noise1.0_drop0.2_cfg0.2_prob_0.5/fid_samples/epoch-99'
        # # no noise in control image
        # synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/MorphFlow/examples/image/output_dir_eval_bbbc_noise1.0_drop0.2_cfg0.2_prob0.5/fid_samples/epoch-100'
elif model_name == 'cellflow_from_noise': 
    ##### CellFlow From Noise ######
    synthetic_samples_path = '/share/pi/syyeung/yuhuiz/Cell/flow_matching/examples/image/output_dir_save_generated_image_from_noise/fid_samples'
else:
    synthetic_samples_path = None

def read_img_from_path(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    return img


if __name__ == '__main__':
    
    with open(yaml_path, 'r') as file:
        cfg = yaml.safe_load(file)
    args = SimpleNamespace(**cfg)
    datamodule = CellDataLoader_Eval(args)
    data_loader_train = datamodule.train_dataloader()
    data_loader_test = datamodule.test_dataloader()
    id2mol = datamodule.id2mol

    fid_metric = FrechetInceptionDistance(normalize=True).to('cuda', non_blocking=True)
    kid_metric = KernelInceptionDistance(subset_size=100, normalize=True).to('cuda', non_blocking=True)

    generated_samples = {i: [] for i in datamodule.id2mol.values()}
    target_samples = {i: [] for i in datamodule.id2mol.values()}
    num_to_cal = 5120 if 'bbbc021' in yaml_path else 30720
    all_real_features = []
    all_fake_features = []
    permute = [0, 1, 2]
    sample_num = 0
    for batch in tqdm(data_loader_test):
        x_real, y_trg = batch['X'], batch['mols']
        idx_ctrl, idx_trt = batch['idx_ctrl'], batch['idx_trt']
        img_file_ctrl, img_file_trt = batch['file_names']
        
        x_real_ctrl, x_real_trt = x_real
        if args.dataset_name == 'rxrx1':
            x_real_trt = convert_6ch_to_3ch(x_real_trt)
        elif args.dataset_name == 'cpg0000':
            x_real_trt = convert_5ch_to_3ch(x_real_trt)

        real_samples = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0)
        real_samples = torch.floor(real_samples * 255).to(torch.float32) / 255.0
        real_samples = real_samples.to('cuda')
        target_classes = [id2mol[y.item()] for y in y_trg]
        synthetic_samples = []
        for i in range(real_samples.shape[0]):
            target_class = target_classes[i]
            synthetic_sample = read_img_from_path(os.path.join(synthetic_samples_path, target_class + f'/{img_file_trt[i]}.png'))
            # synthetic_sample = read_img_from_path(os.path.join(synthetic_samples_path, target_class + f'/{int(idx_trt[i])}.png'))
            synthetic_samples.append(synthetic_sample)
        synthetic_samples = torch.stack(synthetic_samples).to('cuda')
        synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
        synthetic_samples = synthetic_samples[:, permute, :, :]
        real_samples = real_samples[:, permute, :, :]
        # if args.dataset_name == 'rxrx1':
        #     x_real_ctrl = convert_6ch_to_3ch(x_real_ctrl)
        # elif args.dataset_name == 'cpg0000':
        #     x_real_ctrl = convert_5ch_to_3ch(x_real_ctrl)
        
        # synthetic_samples = torch.clamp(x_real_ctrl * 0.5 + 0.5, min=0.0, max=1.0)
        # synthetic_samples = torch.floor(synthetic_samples * 255).to(torch.float32) / 255.0
        # synthetic_samples = synthetic_samples.to('cuda')
        for i in range(real_samples.shape[0]):
            generated_samples[target_classes[i]].append(synthetic_samples[i])
            target_samples[target_classes[i]].append(real_samples[i])

        # Update FID and KID metrics
        fid_metric.update(real_samples, real=True)
        fid_metric.update(synthetic_samples, real=False)

        kid_metric.update(real_samples, real=True)
        kid_metric.update(synthetic_samples, real=False)

        all_real_features.append(real_samples.cpu().numpy().reshape(real_samples.shape[0], -1))
        all_fake_features.append(synthetic_samples.cpu().numpy().reshape(synthetic_samples.shape[0], -1))
        sample_num += real_samples.shape[0]
        if sample_num >= num_to_cal:
            break
    
    all_real_features = np.vstack(all_real_features)
    all_fake_features = np.vstack(all_fake_features)

    # Compute FID and KID on all classes
    fid = fid_metric.compute()
    kid_mean, kid_std = kid_metric.compute()

    # import pdb; pdb.set_trace()
    fid_per_class = {}
    kid_per_class = {}
    if args.dataset_name == 'rxrx1':
        # random sample 50 class from generated samples.keys
        # np.random.seed(0)
        random_classes = np.random.choice(list(generated_samples.keys()), 50, replace=False)
    for key in generated_samples.keys():
        torch.cuda.empty_cache()
        if len(generated_samples[key]) == 0:
            continue
        if args.dataset_name == 'rxrx1' and key not in random_classes:
            continue
        generated_samples[key] = torch.stack(generated_samples[key]).cpu().numpy()
        target_samples[key] = torch.stack(target_samples[key]).cpu().numpy()
        fid_metric.reset()
        fid_metric.update(torch.tensor(target_samples[key]).to('cuda'), real=True)
        fid_metric.update(torch.tensor(generated_samples[key]).to('cuda'), real=False)
        fid_per_class[key] = fid_metric.compute().cpu().numpy()

        dynamic_subset_size = min(len(generated_samples[key]), 100)

        # 按类初始化 KID（动态子采样大小）
        kid_metric_per_class = KernelInceptionDistance(subset_size=dynamic_subset_size, normalize=True).to('cuda', non_blocking=True)
    
        kid_metric.update(torch.tensor(target_samples[key]).to('cuda'), real=True)
        kid_metric.update(torch.tensor(generated_samples[key]).to('cuda'), real=False)
        kid_mean_class, kid_std_class = kid_metric.compute()
        kid_per_class[key] = {"mean": float(kid_mean_class.cpu().numpy()), "std": float(kid_std_class.cpu().numpy())}

        print(f'{key} ({len(generated_samples[key])}): FID={fid_per_class[key]}')
        print(f'{key} ({len(generated_samples[key])}): KID={kid_per_class[key]}')


    print("Overall FID: ", fid.item())
    print("Overall KID: ", {"mean": kid_mean.item(), "std": kid_std.item()})
    
    avg_fid = np.mean(list(fid_per_class.values()))
    avg_kid_mean = np.mean([v["mean"] for v in kid_per_class.values()])
    avg_kid_std = np.mean([v["std"] for v in kid_per_class.values()])
    print(f'Average FID: {avg_fid}')
    print(f'Average KID: mean={avg_kid_mean}, std={avg_kid_std}')
    # Save results to JSON
    fid_per_class = {k: float(v) for k, v in fid_per_class.items()}
    kid_per_class = {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in kid_per_class.items()}
    results = {
        "synthetic_samples_path": synthetic_samples_path,
        "overall_fid": float(fid.item()),
        "overall_kid": {"mean": float(kid_mean.item()), "std": float(kid_std.item())},
        "average_fid": float(avg_fid),
        "average_kid": {"mean": avg_kid_mean, "std": avg_kid_std},
        "fid_per_class": fid_per_class,
    }

    with open(f"eval_results/{model_name}_{dataset}_different_batch.json", "w") as f:
        json.dump(results, f, indent=4)
