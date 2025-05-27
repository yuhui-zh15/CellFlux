import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path

class CustomTransform:
    """Class for scaling and resizing an input image, with optional augmentation and normalization."""
    
    def __init__(self, augment=False, normalize=False, dim=0):
        """
        Initialize the CustomTransform instance.
        
        Args:
            augment (bool, optional): Whether to apply augmentation (random flips). Defaults to False.
            normalize (bool, optional): Whether to normalize the image. Defaults to False.
            dim (int, optional): Dimension along which the normalization is applied. Defaults to 0.
        """
        self.augment = augment 
        self.normalize = normalize 
        self.dim = dim
        
    def __call__(self, X):
        """
        Apply the transformations to the input image.
        
        Args:
            X (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Transformed image tensor.
        """
        # Add random noise and rescale pixels between 0 and 1
        random_noise = torch.rand_like(X)  # Generate random noise
        X = (X + random_noise) / 255.0  # Scale to 0-1 range
        
        t = []
        # Normalize the input to the range [-1, 1]
        if self.normalize:
            num_channels = X.shape[self.dim]
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            t.append(T.Normalize(mean=mean, std=std))
        
        # Perform augmentation steps
        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)

def read_files_pert(file_names, mols, mol2id, y2id, dose, y, transform, image_path, dataset_name, idx, multimodal, batch, iter_ctrl):
    """
    Read and process control and treated batch images.
    
    Args:
        file_names (dict): Dictionary containing file names for 'ctrl' and 'trt' samples.
        mols (dict): Dictionary containing molecule information for 'ctrl' and 'trt' samples.
        mol2id (dict): Mapping from molecule names to IDs.
        y2id (dict): Mapping from annotation names to IDs.
        dose (dict): Dictionary containing dose information for 'ctrl' and 'trt' samples.
        y (dict): Dictionary containing annotation information for 'ctrl' and 'trt' samples.
        transform (callable): Transformation to apply to the images.
        image_path (str): Path to the image folder.
        dataset_name (str): Name of the dataset.
        idx (int): Index of the sample to retrieve.
        multimodal (bool): Whether the dataset is multimodal.
    
    Returns:
        dict: Dictionary containing processed images, molecule information, annotation ID, dose, and file names.
    """
    if iter_ctrl:
        # Sample control and treated batches 
        img_file_ctrl = file_names["ctrl"][idx]
        idx_trt = np.random.randint(0, len(file_names["trt"]))
        img_file_trt = file_names["trt"][idx_trt]
        idx_ctrl = idx
    
    else: 
        idx_trt = idx
        # Use idx to select trt image and random select a ctrl image from the same batch
        img_file_trt = file_names["trt"][idx_trt]
        batch_trt = batch["trt"][idx_trt]

        ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
        if len(ctrl_indices_same_batch) == 0:
            raise ValueError(f"No control samples found in the same batch as the treated sample (batch: {batch_trt}).")

        idx_ctrl = np.random.choice(ctrl_indices_same_batch)
        img_file_ctrl = file_names["ctrl"][idx_ctrl]

    # Split files 
    file_split_ctrl = img_file_ctrl.split('-')
    file_split_trt = img_file_trt.split('-')
    
    if len(file_split_ctrl) > 1:
        file_split_ctrl = file_split_ctrl[1].split("_")
        file_split_trt = file_split_trt[1].split("_")
        path_ctrl = Path(image_path) / "_".join(file_split_ctrl[:2]) / file_split_ctrl[2]
        path_trt = Path(image_path) / "_".join(file_split_trt[:2]) / file_split_trt[2]
        file_ctrl = '_'.join(file_split_ctrl[3:]) + ".npy"
        file_trt = '_'.join(file_split_trt[3:]) + ".npy"
    else:
        file_split_ctrl = file_split_ctrl[0].split("_")
        file_split_trt = file_split_trt[0].split("_")
        if dataset_name == "cpg0000":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}_{file_split_ctrl[2]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}_{file_split_trt[2]}"
            file_ctrl = '_'.join(file_split_ctrl[1:]) + ".npy"
            file_trt = '_'.join(file_split_trt[1:]) + ".npy"
        elif dataset_name == "bbbc021":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}"
            file_ctrl = '_'.join(file_split_ctrl[2:]) + ".npy"
            file_trt = '_'.join(file_split_trt[2:]) + ".npy"
        
    img_ctrl, img_trt = np.load(path_ctrl / file_ctrl), np.load(path_trt / file_trt)
    img_ctrl, img_trt = torch.from_numpy(img_ctrl).float(), torch.from_numpy(img_trt).float()
    img_ctrl, img_trt = img_ctrl.permute(2, 0, 1), img_trt.permute(2, 0, 1)  # Place channel dimension in front of the others 
    img_ctrl, img_trt = transform(img_ctrl), transform(img_trt)
    
    if multimodal:
        y_mod = y["trt"][idx_trt]
        mol = mol2id[y_mod][mols["trt"][idx_trt]]
    else:
        mol = mol2id[mols["trt"][idx_trt]]
    
    return {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'dose': dose["trt"][idx_trt],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    } if dataset_name == "bbbc021" else {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    }

def read_files_batch(file_names, mols, mol2id, y2id, y, transform, image_path, dataset_name, idx):
    """
    Read and process batch images.
    
    Args:
        file_names (list): List of file names for the samples.
        mols (list): List of molecule information for the samples.
        mol2id (dict): Mapping from molecule names to IDs.
        y2id (dict): Mapping from annotation names to IDs.
        y (list): List of annotation information for the samples.
        transform (callable): Transformation to apply to the images.
        image_path (str): Path to the image folder.
        dataset_name (str): Name of the dataset.
        idx (int): Index of the sample to retrieve.
    
    Returns:
        dict: Dictionary containing processed image, molecule information, annotation ID, and file name.
    """
    img_file = file_names[idx]
    file_split = img_file.split('-')
    
    if dataset_name == "rxrx1":
        file_split = file_split[1].split("_")
        path = Path(image_path) / "_".join(file_split[:2]) / file_split[2]
        file = '_'.join(file_split[3:]) + ".npy"
    elif dataset_name in ["bbbc021", "bbbc025"]:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / file_split[1]
        file = '_'.join(file_split[2:]) + ".npy"
    else:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / f"{file_split[1]}_{file_split[2]}"
        file = '_'.join(file_split[1:]) + ".npy"
        
    img = np.load(path / file)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)  # Place channel dimension in front of the others 
    img = transform(img)

    mol = mol2id[mols[idx]]
    
    return {
        'X': img,
        'mols': mol,
        'y_id': y2id[y[idx]],
        'file_names': img_file
    }

def convert_6ch_to_3ch(images):
    """
    Convert 6-channel images to 3-channel RGB composite images.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 6, H, W), values in range [0, 1].
        
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    # Define the weights for each channel in RGB
    # Channel 1-6 mapped to specific colors
    weights = torch.tensor([
        [0, 0, 1],   # Channel 1 -> Blue
        [0, 1, 0],   # Channel 2 -> Green
        [1, 0, 0],   # Channel 3 -> Red
        [0, 0.5, 0.5],  # Channel 4 -> Cyan (lower intensity)
        [0.5, 0, 0.5],  # Channel 5 -> Magenta (lower intensity)
        [0.5, 0.5, 0],  # Channel 6 -> Yellow (lower intensity)
    ], dtype=images.dtype, device=images.device)
    
    # Perform matrix multiplication to combine channels
    # Shape transformation: (batch_size, 6, H, W) -> (batch_size, 3, H, W)
    images_rgb = torch.einsum('bchw,cn->bnhw', images, weights)
    
    # Clip the result to ensure it's within [0, 1]
    images_rgb = torch.clamp(images_rgb, -1, 1)
    
    return images_rgb

def convert_5ch_to_3ch(images):
    """
    Convert 5-channel images to 3-channel RGB composite images.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 5, H, W), values in range [0, 1] or [-1, 1].
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    images_rgb = images[:, :3, :, :]
    return images_rgb