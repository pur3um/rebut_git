import argparse
import csv
import os
import random
import math
from datetime import datetime, timezone
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from optims.rank_wsd_schedulers import RankAwareWarmupStableLinearScheduler
from optims.muon import SingleDeviceMuonWithAuxAdam
from optims.lr_inc import SingleDeviceFinalIncWithAuxAdam
from optims.lr_sign import SingleDeviceSignWithAuxAdam
from optims.lr_svd import SingleDeviceLrSVDWithAuxAdam
from rank_schedule_variants.cosine_inc import (
    SingleDeviceAutoCosIncWithAuxAdam as CosineIncOptimizer,
)
from rank_schedule_variants.exp_inc import (
    SingleDeviceAutoCosIncWithAuxAdam as ExponentialIncOptimizer,
)
from rank_schedule_variants.fixed_rank import (
    SingleDeviceAutoCosIncWithAuxAdam as FixedRankOptimizer,
)
from rank_schedule_variants.linear_inc import (
    SingleDeviceAutoCosIncWithAuxAdam as LinearIncOptimizer,
)
from rank_schedule_variants.log_inc import (
    SingleDeviceAutoCosIncWithAuxAdam as LogarithmicIncOptimizer,
)
from rank_schedule_variants.logistic_inc import (
    SingleDeviceAutoCosIncWithAuxAdam as LogisticIncOptimizer,
)
from rank_schedule_variants.cosine_dec import (
    SingleDeviceAutoCosIncWithAuxAdam as CosineDecOptimizer,
)
from models import (GaussFFN, GaussMLP, ReluFFN, ReluMLP,
                    ReluPosEncoding, SirenMLP,
                    WireMLP, FinerMLP, WireRealMLP)

def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_image(path):
    img = Image.open(path).convert('RGB')
    # just for debugging with smaller images, as in SIREN demo code
    # Resize image to 256x256, for checking quality metrics against official SIREN implementation
    # img = img.resize((256, 256))#, Image.LANCZOS)
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)

def parse_list(param_str, num_layers):
    """
    Parse the parameter from command line.
    
    Args:
        param_str: String from command line (e.g., "0.05" or "0.1,0.05,0.02,0.01")
        num_layers: Number of layers with activations (excludes final output layer)
    
    Returns:
        Single float or list of floats
    """
    if ',' in param_str:
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != (num_layers - 1):
            raise ValueError(f"Number of parameter values ({len(param_values)}) must match num_layers -1 ({num_layers-1}).")
        return param_values
    else:
        param_value = float(param_str)
        param_values = [param_value] * (num_layers - 1)
        return param_values

def get_coordinates(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(),grid_y.flatten()], dim=-1) 
    return coords

def psnr(img1, img2, max_val=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        max_val (float): The maximum possible pixel value of the images 
                         (e.g., 1.0 for normalized floats, 255.0 for 8-bit images).
    """
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2)
    
    # Handle the case where images are identical (MSE=0)
    if mse == 0:
        return float('inf')
    
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val

def compute_lpips(img1, img2, lpips_model, device):
    """Compute LPIPS between two images"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    img1 = (img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    img2 = (img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    
    with torch.no_grad():
        lpips_val = lpips_model(img1, img2)
    
    return lpips_val.item()

def create_comparison_image(gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
    """Create side-by-side comparison of ground truth and reconstructed image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(recon_img, 0, 1))
    axes[1].set_title(f'Reconstruction\nEpoch {epoch}', fontsize=12)
    axes[1].axis('off')
    
    diff = np.abs(gt_img - recon_img)
    diff_amplified = np.clip(diff, 0, 1)
    axes[2].imshow(diff_amplified, cmap='viridis')
    
    title = f'Difference \nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}'
    
    axes[2].set_title(title, fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


RANK_SCHEDULE_OPTIMIZER_CLASSES = {
    'cosine_inc': CosineIncOptimizer,
    'exp_inc': ExponentialIncOptimizer,
    'log_inc': LogarithmicIncOptimizer,
    'linear_inc': LinearIncOptimizer,
    'logistic_inc': LogisticIncOptimizer,
    'cosine_dec': CosineDecOptimizer,
    'fixed_rank': FixedRankOptimizer,
}
RANK_SCHEDULE_OPTIMIZER_NAMES = set(RANK_SCHEDULE_OPTIMIZER_CLASSES)

AUTO_COS_INC_OPTIMIZER_ALIASES = {
    'auto_cos_inc',
    'auto-cos-inc',
    'auto_cos_inc_rank',
    'auto-cos-inc-rank',
    'lr-sign10-rsclF',
}
LOWRANK_OPTIMIZER_NAMES = {'lr-inc', 'lr-sign', 'lr-svd'}
LOWRANK_LIKE_OPTIMIZERS = {
    'muon',
    *LOWRANK_OPTIMIZER_NAMES,
    *RANK_SCHEDULE_OPTIMIZER_NAMES,
}


def normalize_optimizer_name(name):
    """Map accepted aliases to the optimizer's canonical CLI name."""
    if name in AUTO_COS_INC_OPTIMIZER_ALIASES:
        return 'cosine_inc'
    return name.replace('-', '_') if name.replace('-', '_') in RANK_SCHEDULE_OPTIMIZER_NAMES else name


def resolve_rank_schedule_args(args):
    """Resolve and validate rank arguments for rank_schedule_variants."""
    if args.optimizer not in RANK_SCHEDULE_OPTIMIZER_NAMES:
        return

    if args.rank <= 0:
        raise ValueError('--rank must be > 0.')

    if args.rank_start is None:
        args.rank_start = 250 if args.optimizer == 'cosine_dec' else int(args.rank)
    if args.rank_end is None:
        if args.optimizer in {'cosine_dec', 'fixed_rank'}:
            args.rank_end = int(args.rank)
        else:
            args.rank_end = 250

    args.rank_start = int(args.rank_start)
    args.rank_end = int(args.rank_end)

    if args.rank_start <= 0:
        raise ValueError('--rank_start must be > 0 when provided.')
    if args.rank_end <= 0:
        raise ValueError('--rank_end must be > 0 when provided.')
    if args.optimizer.endswith('_inc') and args.rank_end < args.rank_start:
        raise ValueError(
            f'{args.optimizer} requires --rank_end >= --rank_start.'
        )
    if args.optimizer == 'cosine_dec' and args.rank_end > args.rank_start:
        raise ValueError('cosine_dec requires --rank_end <= --rank_start.')
    if args.optimizer == 'fixed_rank' and args.rank_end != args.rank_start:
        raise ValueError('fixed_rank requires --rank_start == --rank_end.')

    if args.rank_warmup_steps is None:
        if args.scheduler == 'rank_wsd':
            if args.rank_wsd_decay_start_step is None:
                args.rank_warmup_steps = max(1, int(round(0.8 * args.epochs)))
            else:
                args.rank_warmup_steps = max(1, int(args.rank_wsd_decay_start_step))
        else:
            args.rank_warmup_steps = max(1, int(args.epochs))
    else:
        args.rank_warmup_steps = int(args.rank_warmup_steps)

    if args.rank_warmup_steps <= 0:
        raise ValueError('--rank_warmup_steps must be > 0 when provided.')
    if args.rank_oversample < 0:
        raise ValueError('--rank_oversample must be >= 0.')
    if args.muon_ns_steps <= 0:
        raise ValueError('--muon_ns_steps must be > 0.')
    if not (0.0 < args.init_energy <= 1.0):
        raise ValueError('--init_energy must be in (0, 1].')
    if args.init_probe_steps <= 0:
        raise ValueError('--init_probe_steps must be > 0.')
    if args.init_round_multiple <= 0:
        raise ValueError('--init_round_multiple must be > 0.')
    if args.exp_rate <= 0:
        raise ValueError('--exp_rate must be > 0.')
    if args.log_rate <= 0:
        raise ValueError('--log_rate must be > 0.')
    if args.logistic_steepness <= 0:
        raise ValueError('--logistic_steepness must be > 0.')
    if args.auto_init_rank_start and args.optimizer in {'cosine_dec', 'fixed_rank'}:
        raise ValueError(
            '--auto_init_rank_start is only supported for increasing schedules.'
        )

    if min(args.rank_start, args.rank_end) < args.rank:
        print(
            'WARNING: --rank is an applied-rank floor. Scheduled values below '
            '--rank will be clipped to --rank.'
        )


def split_matrix_and_aux_params(args, model):
    """Split hidden weight matrices from parameters handled by auxiliary Adam."""
    matrix_params = []
    other_params = []

    first_layer_matrix_models = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}
    is_special_model = args.model in first_layer_matrix_models

    if not hasattr(model, 'mlp') or not isinstance(model.mlp, torch.nn.Sequential):
        print(
            "WARNING: Model does not have a standard 'mlp' attribute. "
            "Cannot identify matrix-optimizer parameters."
        )
        return matrix_params, list(model.parameters())

    num_mlp_layers = len(model.mlp)
    for name, param in model.named_parameters():
        is_matrix_target = False

        if 'mlp' in name and 'weight' in name and param.ndim >= 2:
            try:
                layer_idx = int(name.split('.')[1])
                is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                is_selected_first_layer = (
                    is_special_model
                    and args.optimize_first_layer_with_muon
                    and layer_idx == 0
                )
                is_matrix_target = is_hidden_layer or is_selected_first_layer
            except (ValueError, IndexError):
                pass

        if is_matrix_target:
            matrix_params.append(param)
        else:
            other_params.append(param)

    if is_special_model and args.optimize_first_layer_with_muon:
        print(
            "INFO: --optimize_first_layer_with_muon=True. "
            "The first MLP layer will also use the selected matrix optimizer."
        )

    return matrix_params, other_params


def build_optimizer(args, model):
    """Build the selected optimizer with the same choices as fit_sisr_sched.py."""
    if args.optimizer in LOWRANK_LIKE_OPTIMIZERS:
        matrix_params, other_params = split_matrix_and_aux_params(args, model)
        if not matrix_params:
            raise ValueError(
                f"{args.optimizer} was selected (model: {args.model}), but no "
                "suitable weight matrices were identified."
            )

        param_groups = [
            dict(
                params=matrix_params,
                use_muon=True,
                lr=args.muon_lr,
                weight_decay=args.muon_weight_decay,
            ),
            dict(
                params=other_params,
                use_muon=False,
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.muon_aux_weight_decay,
            ),
        ]

        if args.optimizer in RANK_SCHEDULE_OPTIMIZER_NAMES:
            param_groups[0].update(
                momentum=args.muon_momentum,
                ns_steps=args.muon_ns_steps,
                nesterov=not args.muon_no_nesterov,
                rank=args.rank,
                rank_start=args.rank_start,
                rank_end=args.rank_end,
                warmup_steps=args.rank_warmup_steps,
                oversample=args.rank_oversample,
                lowrank_rescale=args.lowrank_rescale,
                eps=args.lowrank_eps,
                small_ns_bfloat16=args.small_ns_bfloat16,
                auto_init_rank_start=args.auto_init_rank_start,
                init_probe_steps=args.init_probe_steps,
                init_energy=args.init_energy,
                init_round_multiple=args.init_round_multiple,
                exp_rate=args.exp_rate,
                log_rate=args.log_rate,
                logistic_steepness=args.logistic_steepness,
            )

        optimizer_classes = {
            'muon': SingleDeviceMuonWithAuxAdam,
            'lr-inc': SingleDeviceFinalIncWithAuxAdam,
            'lr-sign': SingleDeviceSignWithAuxAdam,
            'lr-svd': SingleDeviceLrSVDWithAuxAdam,
            **RANK_SCHEDULE_OPTIMIZER_CLASSES,
        }
        optimizer = optimizer_classes[args.optimizer](param_groups)
        print(
            f"INFO: {args.optimizer} optimizer configured. "
            f"Matrix params: {len(matrix_params)}, auxiliary params: {len(other_params)}."
        )
        return optimizer

    if args.optimizer == 'lbfgs':
        print("INFO: Using L-BFGS optimizer.")
        return optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
        )

    if args.optimizer == 'adam':
        print("INFO: Using Adam optimizer.")
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.adam_weight_decay,
        )

    raise ValueError(f"Unsupported optimizer type: {args.optimizer}")


def is_rank_wsd_scheduler(scheduler):
    return isinstance(scheduler, RankAwareWarmupStableLinearScheduler)


def infer_rank_wsd_base_lrs(args, optimizer):
    base_lr_adam = None
    base_lr_muon = None

    for group in optimizer.param_groups:
        group_lr = group.get('lr')
        if group_lr is None:
            continue
        if group.get('use_muon', False) and base_lr_muon is None:
            base_lr_muon = float(group_lr)
        elif not group.get('use_muon', False) and base_lr_adam is None:
            base_lr_adam = float(group_lr)

    if base_lr_adam is None:
        base_lr_adam = float(args.lr)
    if base_lr_muon is None:
        base_lr_muon = float(args.muon_lr)

    return base_lr_adam, base_lr_muon


def infer_optimizer_n_rand(optimizer):
    for attr_name in ('N_rand', 'n_rand'):
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            try:
                value = int(value)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

    for group in optimizer.param_groups:
        for key in ('N_rand', 'n_rand'):
            if key in group:
                try:
                    value = int(group[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    defaults = getattr(optimizer, 'defaults', {})
    if isinstance(defaults, dict):
        for key in ('N_rand', 'n_rand'):
            if key in defaults:
                try:
                    value = int(defaults[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    return None


def resolve_rank_wsd_base_n_rand(args, optimizer):
    if args.rank_wsd_base_N_rand is not None:
        if args.rank_wsd_base_N_rand <= 0:
            raise ValueError('--rank_wsd_base_N_rand must be > 0 when provided.')
        return int(args.rank_wsd_base_N_rand)

    inferred_n_rand = infer_optimizer_n_rand(optimizer)
    return inferred_n_rand if inferred_n_rand is not None else 1


def apply_n_rand_to_optimizer(optimizer, n_rand):
    n_rand = int(n_rand)

    for setter_name in ('set_N_rand', 'set_n_rand'):
        setter = getattr(optimizer, setter_name, None)
        if callable(setter):
            try:
                setter(n_rand)
            except TypeError:
                pass

    for attr_name in ('N_rand', 'n_rand'):
        if hasattr(optimizer, attr_name):
            try:
                setattr(optimizer, attr_name, n_rand)
            except (AttributeError, TypeError):
                pass

    for group in optimizer.param_groups:
        for key in ('N_rand', 'n_rand'):
            if key in group:
                group[key] = n_rand

    defaults = getattr(optimizer, 'defaults', None)
    if isinstance(defaults, dict):
        for key in ('N_rand', 'n_rand'):
            if key in defaults:
                defaults[key] = n_rand


def apply_rank_wsd_scheduler_step(scheduler, optimizer, global_step):
    state = scheduler.step(global_step)

    for group in optimizer.param_groups:
        if group.get('use_muon', False):
            group['lr'] = state['lr_muon']
        else:
            group['lr'] = state['lr_adam']

    apply_n_rand_to_optimizer(optimizer, state['N_rand'])
    return state


def build_scheduler(args, optimizer):
    """Build the selected scheduler with the same choices as fit_sisr_sched.py."""
    if args.optimizer == 'lbfgs':
        print("INFO: Schedulers are disabled for L-BFGS optimizer.")
        return None

    if args.scheduler == 'none':
        print("INFO: Scheduler is disabled.")
        return None

    if args.scheduler == 'cosine':
        print(f"Using CosineAnnealingLR scheduler with T_max={args.T_max}")
        return CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-6)

    if args.scheduler == 'step':
        print(
            f"Using StepLR scheduler with step_size={args.step_size} "
            f"and gamma={args.gamma}"
        )
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.scheduler == 'rank_wsd':
        base_lr_adam, base_lr_muon = infer_rank_wsd_base_lrs(args, optimizer)
        base_n_rand = resolve_rank_wsd_base_n_rand(args, optimizer)
        scheduler = RankAwareWarmupStableLinearScheduler(
            base_lr_adam=base_lr_adam,
            base_lr_muon=base_lr_muon,
            total_iters=args.epochs,
            base_N_rand=base_n_rand,
            warmup_steps=args.rank_wsd_warmup_steps,
            decay_start_step=args.rank_wsd_decay_start_step,
            min_lr_ratio=args.rank_wsd_min_lr_ratio,
        )
        print(scheduler.describe())

        if args.optimizer not in LOWRANK_LIKE_OPTIMIZERS:
            print(
                "WARNING: rank_wsd is intended for Muon/low-rank optimizers. "
                "For this optimizer it schedules LR only unless N_rand/n_rand is exposed."
            )
        return scheduler

    raise ValueError(f"Unsupported scheduler type: {args.scheduler}")


def get_rank_schedule_state(optimizer):
    for group in optimizer.param_groups:
        if group.get('use_muon', False):
            return {
                'current_rank': group.get('current_rank'),
                'current_target_rank': group.get('current_target_rank'),
                'rank_floor': group.get('rank'),
                'rank_start': group.get('rank_start'),
                'rank_end': group.get('rank_end'),
                'rank_warmup_steps': group.get('warmup_steps'),
                'auto_rank_start_final': group.get('auto_rank_start_final'),
                'current_method': group.get('current_method'),
                'rank_schedule': group.get('rank_schedule'),
            }
    return {}


def train_model(args):
    """Train the neural field model on the given image with specified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    img = load_image(args.image)
    h, w, c = img.shape
 
    coords = get_coordinates(h, w)
    target = img.view(-1, c)
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    if is_inpainting_task:
        print(f"Performing inpainting task. Training on {args.inpainting_ratio * 100:.2f}% of pixels.")
        num_pixels = h * w
        num_train_pixels = int(num_pixels * args.inpainting_ratio)
        
        indices = torch.randperm(num_pixels)
        train_indices = indices[:num_train_pixels]
        test_indices = indices[num_train_pixels:]
        
        train_coords = coords[train_indices].to(device)
        train_target = target[train_indices].to(device)
        
        test_coords = coords[test_indices].to(device)
        test_target = target[test_indices].to(device)
        
        mask = torch.ones(num_pixels, 1) * 0.2 # just for visualization
        mask[train_indices] = 1.0
        mask_img = mask.view(h, w, 1).cpu().numpy()
        wandb.log({"images/training_mask": wandb.Image(mask_img, caption="Training Pixel Mask (White = Known)")})
    else:
        print("Performing overfitting task on all pixels.")
        train_coords = coords.to(device)
        train_target = target.to(device)

    if args.model == 'relu_ffn':
        model = ReluFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                       output_dim=c, num_layers=args.num_layers, sigma=args.fourier_sigma)
    elif args.model == 'relu_mlp':
        model = ReluMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)
    elif args.model == 'relu_pos_enc':
        model = ReluPosEncoding(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers)        
    elif args.model == 'gauss_ffn':
        model = GaussFFN(input_dim=2, mapping_size=args.mapping_size, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, 
                        sigma=args.fourier_sigma, a=args.gauss_scale)
    elif args.model == 'gauss_mlp':
        model = GaussMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, a=args.gauss_scale)
    elif args.model == 'siren_mlp':
        model = SirenMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.siren_omega)
    elif args.model == 'wire_mlp':
        model = WireMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    elif args.model == 'real_wire':
        model = WireRealMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.wire_omega, sigma=args.wire_sigma)
    elif args.model == 'finer_mlp':
        model = FinerMLP(input_dim=2, hidden_dim=args.hidden_dim,
                        output_dim=c, num_layers=args.num_layers, omega=args.finer_omega, 
                        init_bias=args.finer_init_bias, bias_scale=args.finer_bias_scale)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    print(f"Using model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters.")
    print(model)

    model = model.to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    
    metrics = {
        'epochs': [], 'full_psnr': [], 'ssim': [], 'lpips': [],
        'train_psnr': [], 'test_psnr': [], 'current_rank': [],
        'target_rank': [], 'rank_schedule': [], 'lr_muon': [], 'lr_aux': [],
    }
    layer_metrics = {}
    
    gt_img_np = img.numpy()
    # omit caption for ground truth
    wandb.log({"images/ground_truth": wandb.Image(gt_img_np)})

    # stable rank etc
    layer_metrics = {}
    
    for epoch in range(args.epochs):
        model.train()

        rank_wsd_state = None
        if is_rank_wsd_scheduler(scheduler):
            rank_wsd_state = apply_rank_wsd_scheduler_step(
                scheduler,
                optimizer,
                epoch,
            )

        if args.optimizer == 'lbfgs':
            # L-BFGS requires a closure function
            def closure():
                optimizer.zero_grad()
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            # Re-calculate loss for logging after the step
            with torch.no_grad():
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)

        else: # Adam and Muon
            optimizer.zero_grad()
            pred = model(train_coords)
            loss = F.mse_loss(pred, train_target)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None and not is_rank_wsd_scheduler(scheduler):
            scheduler.step()
        
        if epoch % args.log_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                # calculate metrics on the full image for SSIM, LPIPS, etc.
                full_pred = model(coords.to(device))
                pred_img = full_pred.view(h, w, c).cpu().numpy()
                target_img = img.numpy()
                
                full_psnr_val = psnr(torch.tensor(pred_img), torch.tensor(target_img)).item()
                ssim_val = ssim(target_img, pred_img, channel_axis=-1, data_range=1.0)
                lpips_val = compute_lpips(target_img, pred_img, lpips_model, device)
                
                # B. Calculate PSNR on train/test subsets
                train_psnr_val = psnr(model(train_coords), train_target).item()
                
                test_psnr_val = 0.0
                if is_inpainting_task:
                    test_psnr_val = psnr(model(test_coords), test_target).item()

                metrics['epochs'].append(epoch)
                metrics['full_psnr'].append(full_psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['lpips'].append(lpips_val)
                metrics['train_psnr'].append(train_psnr_val)
                metrics['test_psnr'].append(test_psnr_val)

                rank_state = (
                    get_rank_schedule_state(optimizer)
                    if args.optimizer in RANK_SCHEDULE_OPTIMIZER_NAMES
                    else {}
                )
                metrics['current_rank'].append(rank_state.get('current_rank'))
                metrics['target_rank'].append(rank_state.get('current_target_rank'))
                metrics['rank_schedule'].append(rank_state.get('rank_schedule'))
                if args.optimizer in LOWRANK_LIKE_OPTIMIZERS:
                    metrics['lr_muon'].append(optimizer.param_groups[0]['lr'])
                    metrics['lr_aux'].append(optimizer.param_groups[1]['lr'])
                else:
                    metrics['lr_muon'].append(None)
                    metrics['lr_aux'].append(optimizer.param_groups[0]['lr'])
                
                if is_inpainting_task:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, TrainPSNR={train_psnr_val:.2f}, TestPSNR={test_psnr_val:.2f}, FullPSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                else:
                    print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                
                log_dict = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'full_psnr': full_psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val,
                    'train_psnr': train_psnr_val,
                }

                if args.optimizer in LOWRANK_LIKE_OPTIMIZERS:
                    log_dict['learning_rate_muon'] = optimizer.param_groups[0]['lr']
                    log_dict['learning_rate_aux'] = optimizer.param_groups[1]['lr']
                else:
                    log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

                if args.optimizer in RANK_SCHEDULE_OPTIMIZER_NAMES:
                    for key, value in rank_state.items():
                        log_dict[f'rank_schedule/{key}'] = value

                if rank_wsd_state is not None:
                    log_dict['rank_wsd_phase'] = rank_wsd_state['phase']
                    log_dict['rank_wsd_phase_name'] = rank_wsd_state['phase_name']
                    log_dict['rank_wsd_lr_ratio'] = rank_wsd_state['lr_ratio']
                    log_dict['rank_wsd_N_rand'] = rank_wsd_state['N_rand']

                if is_inpainting_task:
                    log_dict['test_psnr'] = test_psnr_val
                
                if args.log_image_evolution:
                    comparison_fig = create_comparison_image(target_img, pred_img, epoch, full_psnr_val, ssim_val, lpips_val)
                    log_dict.update({
                        'images/comparison': wandb.Image(comparison_fig, caption=f'Epoch {epoch}: PSNR={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}'),
                        'images/reconstruction': wandb.Image(np.clip(pred_img, 0, 1), caption=f'Reconstruction at epoch {epoch}')
                    })
                    plt.close(comparison_fig)
                
                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    for i, layer_info in enumerate(info['layer_infos']):
                        # Initialize lists in dictionary if they don't exist
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'effective_rank_layer_{i}'] = []  # <-- ADD THIS
                            layer_metrics[f'spectral_norm_layer_{i}'] = []
                            layer_metrics[f'condition_number_layer_{i}'] = []

                        # Get scalar values
                        stable_rank_val = layer_info.get('stable_rank', 0)
                        effective_rank_val = layer_info.get('effective_rank', 0)  # <-- ADD THIS
                        spectral_norm_val = layer_info.get('linear_spectral_norm', 0)
                        condition_number_val = layer_info.get('spectral_condition_no', 0)

                        # Append to our history dictionary
                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'effective_rank_layer_{i}'].append(effective_rank_val)  # <-- ADD THIS
                        layer_metrics[f'spectral_norm_layer_{i}'].append(spectral_norm_val)
                        layer_metrics[f'condition_number_layer_{i}'].append(condition_number_val)

                        # Add to the current wandb log dictionary
                        log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                        log_dict[f'effective_rank/layer_{i}'] = effective_rank_val  # <-- ADD THIS
                        log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                        log_dict[f'condition_number/layer_{i}'] = condition_number_val
                    
                    if 'end_to_end_spectral_bound' in info:
                         if 'end_to_end_bound' not in layer_metrics:
                            layer_metrics['end_to_end_bound'] = []
                         end_to_end_val = info['end_to_end_spectral_bound']
                         layer_metrics['end_to_end_bound'].append(end_to_end_val)
                         log_dict['end_to_end_bound'] = end_to_end_val

                wandb.log(log_dict)

    return model, optimizer, metrics, layer_metrics, img, lpips_model

def plot_metrics_seaborn_separate(metrics, layer_metrics, args):
    """Plot training metrics using seaborn and log to wandb."""
    sns.set_style("darkgrid")
    
    is_inpainting_task = args.inpainting_ratio < 1.0
    
    main_df = pd.DataFrame({
        'Epoch': metrics['epochs'],
        'Full PSNR': metrics['full_psnr'],
        'SSIM': metrics['ssim'],
        'LPIPS': metrics['lpips'],
        'Train PSNR': metrics['train_psnr']
    })
    if is_inpainting_task:
        main_df['Test PSNR'] = metrics['test_psnr']

    plt.figure(figsize=(10, 6))
    if is_inpainting_task:
        psnr_df = main_df.melt(id_vars=['Epoch'], value_vars=['Train PSNR', 'Test PSNR', 'Full PSNR'],
                               var_name='Metric', value_name='PSNR (dB)')
        ax = sns.lineplot(data=psnr_df, x='Epoch', y='PSNR (dB)', hue='Metric')
        ax.set_title('PSNR Over Training (Inpainting)', fontsize=16, fontweight='bold', pad=20)
    else:
        ax = sns.lineplot(data=main_df, x='Epoch', y='Full PSNR')
        ax.set_title('PSNR Over Training (Overfitting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    plt.tight_layout()
    wandb.log({"plots/psnr_comparison": wandb.Image(plt)})
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='SSIM')
    ax.set_title('Full Image SSIM Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    wandb.log({"plots/ssim": wandb.Image(plt)})
    plt.close()
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='LPIPS')
    ax.set_title('Full Image LPIPS Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    wandb.log({"plots/lpips": wandb.Image(plt)})
    plt.close()
    
    if not layer_metrics:
        print("No layer metrics found to plot.")
        sns.reset_defaults()
        return

    layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
    for key, values in layer_metrics.items():
        layer_df[key] = values

    # 3. Stable Ranks Plot
    stable_rank_keys = [k for k in layer_df.columns if 'stable_rank_layer_' in k]
    if stable_rank_keys:
        plt.figure(figsize=(12, 7))
        stable_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=stable_rank_keys, var_name='Layer', value_name='Stable Rank')
        stable_rank_df_melted['Layer'] = stable_rank_df_melted['Layer'].str.replace('stable_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=stable_rank_df_melted, x='Epoch', y='Stable Rank', hue='Layer', linewidth=2)
        ax.set_title('Stable Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/stable_ranks": wandb.Image(plt)})
        plt.close()

    # 4. Effective Ranks Plot  
    effective_rank_keys = [k for k in layer_df.columns if 'effective_rank_layer_' in k]
    if effective_rank_keys:
        plt.figure(figsize=(12, 7))
        effective_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=effective_rank_keys, var_name='Layer', value_name='Effective Rank')
        effective_rank_df_melted['Layer'] = effective_rank_df_melted['Layer'].str.replace('effective_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=effective_rank_df_melted, x='Epoch', y='Effective Rank', hue='Layer', linewidth=2)
        ax.set_title('Effective Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/effective_ranks": wandb.Image(plt)})
        plt.close()

    # 5. Spectral Norms Plot 
    spectral_norm_keys = [k for k in layer_df.columns if 'spectral_norm_layer_' in k]
    if spectral_norm_keys:
        plt.figure(figsize=(12, 7))
        spectral_norm_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=spectral_norm_keys, var_name='Layer', value_name='Spectral Norm')
        spectral_norm_df_melted['Layer'] = spectral_norm_df_melted['Layer'].str.replace('spectral_norm_layer_', 'Layer ')
        ax = sns.lineplot(data=spectral_norm_df_melted, x='Epoch', y='Spectral Norm', hue='Layer', linewidth=2)
        ax.set_title('Spectral Norm Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/spectral_norms": wandb.Image(plt)})
        plt.close()

    # 6. Condition Numbers Plot 
    condition_keys = [k for k in layer_df.columns if 'condition_number_layer_' in k]
    if condition_keys:
        plt.figure(figsize=(12, 7))
        condition_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=condition_keys, var_name='Layer', value_name='Condition Number')
        condition_df_melted['Layer'] = condition_df_melted['Layer'].str.replace('condition_number_layer_', 'Layer ')
        ax = sns.lineplot(data=condition_df_melted, x='Epoch', y='Condition Number', hue='Layer', linewidth=2)
        ax.set_title('Condition Number Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        wandb.log({"plots/condition_numbers": wandb.Image(plt)})
        plt.close()

    sns.reset_defaults()
    print("All plots saved and logged to wandb under 'plots/' namespace")


def make_default_experiment_name(args):
    image_stem = Path(args.image).stem
    return (
        f"{args.optimizer}__r{args.rank_start}-to-{args.rank_end}"
        f"__ep{args.epochs}__{image_stem}__seed{args.seed}"
    )


def upsert_csv_row(csv_path, row, key='run_id'):
    """Atomically insert or replace one result row, keyed by run_id."""
    import fcntl

    csv_path = Path(csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{csv_path}.lock")

    with lock_path.open('a+', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing_rows = []
        fieldnames = list(row.keys())

        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open('r', newline='', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                for existing in reader:
                    if existing.get(key) != str(row[key]):
                        existing_rows.append(existing)
                for field in reader.fieldnames or []:
                    if field not in fieldnames:
                        fieldnames.append(field)

        temp_path = Path(f"{csv_path}.tmp.{os.getpid()}")
        with temp_path.open('w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow(existing)
            writer.writerow({field: row.get(field, '') for field in fieldnames})
        os.replace(temp_path, csv_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_local_results(args, metrics, final_pred, result_row):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(
        output_dir / 'training_metrics.csv',
        index=False,
    )
    pd.DataFrame([result_row]).to_csv(
        output_dir / 'final_result.csv',
        index=False,
    )

    reconstruction = (
        np.clip(final_pred.numpy(), 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(reconstruction).save(output_dir / 'final_reconstruction.png')

    if args.results_csv:
        upsert_csv_row(args.results_csv, result_row)


def main():
    parser = argparse.ArgumentParser(description="Train a Neural Field for Image Overfitting or Inpainting.")

    # General arguments
    parser.add_argument('--image', default='data/kodim01.png', type=str, help='Path to the input image.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--log_n_epochs', default=500, type=int, help='Frequency of logging metrics and images.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--log_image_evolution', action='store_true', help='Log intermediate image reconstructions to wandb.')
    parser.add_argument('--project_name', type=str, default='cvpr-images', help='Wandb project name.')
    parser.add_argument('--experiment_name', type=str, default=None, help='Stable run name used in CSV output and W&B.')
    parser.add_argument('--output_dir', type=str, default=None, help='Per-run directory for CSVs, reconstruction, and W&B files.')
    parser.add_argument('--results_csv', type=str, default=None, help='Aggregate CSV updated after a successful run.')
    parser.add_argument('--list_optimizers', action='store_true', help='Print available optimizer names and exit.')

    # Optimizer arguments
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=sorted(
            {'adam', 'muon', 'lbfgs', *LOWRANK_OPTIMIZER_NAMES}
            | AUTO_COS_INC_OPTIMIZER_ALIASES
            | RANK_SCHEDULE_OPTIMIZER_NAMES
        ),
        help='Optimizer to use.',
    )
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for Adam, Muon (aux), and L-BFGS (initial step size).')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0, help='Weight decay for pure Adam optimizer.')

    # Muon specific
    parser.add_argument('--muon_weight_decay', type=float, default=0.0, help='Weight decay for Muon optimizer (hidden weights).')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_lr', type=float, default=1e-1, help='Learning rate for the Muon part (hidden weights).')
    parser.add_argument('--muon_momentum', type=float, default=0.95, help='Momentum beta for matrix-optimizer updates.')
    parser.add_argument('--muon_ns_steps', type=int, default=10, help='Newton-Schulz steps for rank-schedule variants.')
    parser.add_argument('--muon_no_nesterov', action='store_true', help='Disable Nesterov momentum for rank-schedule variants.')
    parser.add_argument('--optimize_first_layer_with_muon', action='store_true', help='For models with embeddings (FFN, PosEnc), also optimize the first linear layer of the MLP with Muon.')

    # rank_schedule_variants specific arguments
    parser.add_argument('--rank', type=int, default=200, help='Applied-rank floor used by rank_schedule_variants.')
    parser.add_argument('--rank_start', type=int, default=None, help='Initial scheduled rank. Default: --rank.')
    parser.add_argument('--rank_end', type=int, default=None, help='Final scheduled rank. Defaults depend on the selected variant.')
    parser.add_argument('--rank_warmup_steps', type=int, default=None, help='Steps over which rank grows. Default: rank_wsd decay start if scheduler=rank_wsd, otherwise epochs.')
    parser.add_argument('--rank_oversample', type=int, default=4, help='Randomized low-rank oversampling dimension.')
    parser.add_argument('--lowrank_rescale', action='store_true', help='Enable low-rank update rescaling in auto_cos_inc_rank.py.')
    parser.add_argument('--lowrank_eps', type=float, default=1e-6, help='Numerical epsilon for low-rank Muon update.')
    parser.add_argument('--small_ns_bfloat16', action='store_true', help='Use bfloat16 for small Newton-Schulz path on CUDA.')
    parser.add_argument('--auto_init_rank_start', action='store_true', help='Estimate rank_start from early sketched Frobenius-energy probes.')
    parser.add_argument('--init_probe_steps', type=int, default=8, help='Number of early steps for auto rank_start estimation.')
    parser.add_argument('--init_energy', type=float, default=0.90, help='Energy threshold for auto rank_start estimation.')
    parser.add_argument('--init_round_multiple', type=int, default=8, help='Round auto-estimated rank_start up to this multiple.')
    parser.add_argument('--exp_rate', type=float, default=5.0, help='Curve rate for exp_inc.')
    parser.add_argument('--log_rate', type=float, default=9.0, help='Curve rate for log_inc.')
    parser.add_argument('--logistic_steepness', type=float, default=12.0, help='Curve steepness for logistic_inc.')
    parser.add_argument('--muon_delta', type=float, default=None, help='Optional Muon update operator-norm clipping radius.')
    parser.add_argument('--muon_op_clip_mode', type=str, default='matrix', choices=['matrix', 'conv'], help='Operator-norm estimation mode.')
    parser.add_argument('--muon_op_clip_iters', type=int, default=2, help='Power-iteration steps for operator-norm estimation.')

    # L-BFGS specific
    parser.add_argument('--lbfgs_max_iter', type=int, default=20, help='Max iterations per epoch for L-BFGS.')
    parser.add_argument('--lbfgs_history_size', type=int, default=100, help='History size for L-BFGS.')

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step', 'rank_wsd', 'rank-wsd'], help='Learning rate scheduler type (not used with L-BFGS).')
    parser.add_argument('--T_max', type=int, default=5000, help='T_max for CosineAnnealingLR scheduler.')
    parser.add_argument('--step_size', type=int, default=500, help='Step size for StepLR scheduler.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for StepLR scheduler.')

    # Rank-WSD scheduler arguments
    parser.add_argument('--rank_wsd_warmup_steps', type=int, default=0, help='Warmup steps for rank_wsd scheduler.')
    parser.add_argument('--rank_wsd_decay_start_step', type=int, default=None, help='Step where rank_wsd starts late linear decay. Default: 0.8 * epochs.')
    parser.add_argument('--rank_wsd_min_lr_ratio', type=float, default=0.1, help='Final LR ratio for rank_wsd scheduler.')
    parser.add_argument('--rank_wsd_base_N_rand', type=int, default=None, help='Base N_rand for rank_wsd. If omitted, infer it from the optimizer when possible.')

    # Model and architecture arguments
    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'relu_hash', 'gauss_ffn', 'gauss_mlp', 'siren_mlp', 'wire_mlp', 'real_wire', 'relu_pos_enc', 'replicate_mlp', 'finer_mlp', 'fourier_net'], default='relu_mlp')
    parser.add_argument('--num_layers', default=5, type=int, help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', default=300, type=int, help='Dimension of hidden layers.')
    parser.add_argument('--mapping_size', default=128, type=int, help='Mapping size for Fourier Feature mappings.')
    
    # Model-specific activation arguments
    parser.add_argument('--fourier_sigma', default=10.0, type=float, help='Sigma for Fourier Feature mapping.')
    parser.add_argument('--siren_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for SIREN layers.")
    parser.add_argument('--finer_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for FINER layers.")
    parser.add_argument('--finer_init_bias', action="store_true",  help="Initial bias for FINER layers.")
    parser.add_argument('--finer_bias_scale', default=float(1/math.sqrt(2)), type=float, help="Bias scale for FINER layers.")
    parser.add_argument('--gauss_scale', default='0.0236', type=str, help='Scale parameter for Gaussian activation.')
    parser.add_argument('--wire_sigma', default='10.0', type=str, help="Sigma for WIRE layers.")
    parser.add_argument('--wire_omega', default='20.0', type=str, help="Omega for WIRE layers.")
    parser.add_argument('--mfn_fourier_scale', default=256, type=float, help='Scale for MFN_FourierNet model.')
    
    # HashGrid specific arguments
    parser.add_argument('--hash_n_levels', default=16, type=int, help='Number of levels in HashGrid.')
    parser.add_argument('--hash_n_features_per_level', default=2, type=int, help='Number of features per level in HashGrid.')
    parser.add_argument('--hash_log2_hashmap_size', default=15, type=int, help='Log2 of hashmap size for HashGrid.')
    parser.add_argument('--hash_base_resolution', default=16, type=int, help='Base resolution for HashGrid.')
    parser.add_argument('--hash_finest_resolution', default=512, type=int, help='Finest resolution for HashGrid.')
    parser.add_argument('--inpainting_ratio', default=1.0, type=float, help='Ratio of pixels for training (1.0 for overfitting, <1.0 for inpainting).')

    args = parser.parse_args()
    if args.list_optimizers:
        print('\n'.join(sorted(RANK_SCHEDULE_OPTIMIZER_NAMES)))
        return

    args.optimizer = normalize_optimizer_name(args.optimizer)
    if args.scheduler == 'rank-wsd':
        args.scheduler = 'rank_wsd'
    if args.scheduler == 'rank_wsd' and args.rank_wsd_decay_start_step is None:
        args.rank_wsd_decay_start_step = max(1, int(round(0.8 * args.epochs)))
    resolve_rank_schedule_args(args)

    if args.experiment_name is None:
        args.experiment_name = make_default_experiment_name(args)
    if args.output_dir is None:
        args.output_dir = str(
            Path('results') / 'rank_schedule_variants' / args.experiment_name
        )
    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Parse model-specific parameters that can be lists
    args.gauss_scale = parse_list(args.gauss_scale, args.num_layers)
    args.siren_omega = parse_list(args.siren_omega, args.num_layers)
    args.finer_omega = parse_list(args.finer_omega, args.num_layers)
    args.wire_sigma = parse_list(args.wire_sigma, args.num_layers)
    args.wire_omega = parse_list(args.wire_omega, args.num_layers)

    if not 0.0 < args.inpainting_ratio <= 1.0:
        raise ValueError("inpainting_ratio must be between 0.0 and 1.0.")

    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        dir=args.output_dir,
        config=vars(args)
    )

    model, optimizer, metrics, layer_metrics, original_img, lpips_model = train_model(args)
    
    # Final evaluation and logging
    model.eval()
    with torch.no_grad():
        h, w, c = original_img.shape
        coords = get_coordinates(h, w).to(next(model.parameters()).device)
        final_pred = model(coords).view(h, w, c).cpu()
        
        final_psnr = psnr(final_pred, original_img).item()
        final_ssim = ssim(original_img.numpy(), final_pred.numpy(), channel_axis=-1, data_range=1.0)
        final_lpips = compute_lpips(original_img.numpy(), final_pred.numpy(), 
                                  lpips_model, next(model.parameters()).device)
        
        print("\n--- Final Results ---")
        print(f"Final Full Image PSNR: {final_psnr:.2f}")
        print(f"Final Full Image SSIM: {final_ssim:.4f}")
        print(f"Final Full Image LPIPS: {final_lpips:.4f}")
        
        final_comparison_fig = create_comparison_image(
            original_img.numpy(), final_pred.numpy(), args.epochs, final_psnr, final_ssim, final_lpips)
        
        log_dict = {
            'final_full_psnr': final_psnr,
            'final_ssim': final_ssim,
            'final_lpips': final_lpips,
            'images/final_comparison': wandb.Image(final_comparison_fig, 
                                                 caption=f'Final Result: PSNR={final_psnr:.2f}, SSIM={final_ssim:.4f}, LPIPS={final_lpips:.4f}'),                                
            'images/final_reconstruction': wandb.Image(np.clip(final_pred.numpy(), 0, 1))
        }
            
        wandb.log(log_dict)
        plt.close(final_comparison_fig)

    plot_metrics_seaborn_separate(metrics, layer_metrics, args)

    rank_state = (
        get_rank_schedule_state(optimizer)
        if args.optimizer in RANK_SCHEDULE_OPTIMIZER_NAMES
        else {}
    )
    result_row = {
        'run_id': args.experiment_name,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'optimizer': args.optimizer,
        'rank_schedule': rank_state.get('rank_schedule', ''),
        'scheduler': args.scheduler,
        'image': str(Path(args.image).expanduser().resolve()),
        'image_name': Path(args.image).name,
        'task': 'inpainting' if args.inpainting_ratio < 1.0 else 'overfitting',
        'inpainting_ratio': args.inpainting_ratio,
        'epochs': args.epochs,
        'seed': args.seed,
        'model': args.model,
        'num_layers': args.num_layers,
        'hidden_dim': args.hidden_dim,
        'lr_aux': args.lr,
        'lr_muon': args.muon_lr,
        'muon_momentum': args.muon_momentum,
        'muon_ns_steps': args.muon_ns_steps,
        'rank_floor': args.rank,
        'rank_start': args.rank_start,
        'rank_end': args.rank_end,
        'rank_warmup_steps': args.rank_warmup_steps,
        'final_applied_rank': rank_state.get('current_rank', ''),
        'rank_wsd_warmup_steps': args.rank_wsd_warmup_steps,
        'rank_wsd_decay_start_step': args.rank_wsd_decay_start_step,
        'rank_wsd_min_lr_ratio': args.rank_wsd_min_lr_ratio,
        'exp_rate': args.exp_rate,
        'log_rate': args.log_rate,
        'logistic_steepness': args.logistic_steepness,
        'final_psnr': final_psnr,
        'final_ssim': final_ssim,
        'final_lpips': final_lpips,
        'output_dir': str(Path(args.output_dir).expanduser().resolve()),
    }
    save_local_results(args, metrics, final_pred, result_row)

    wandb.finish()
    print(f"Training finished. Local results: {args.output_dir}")
    if args.results_csv:
        print(f"Aggregate CSV: {Path(args.results_csv).expanduser().resolve()}")

if __name__ == "__main__":
    main()

"""
# Increasing example (replace auto_cos_inc with auto_exp_inc, auto_log_inc,
# auto_linear_inc, or auto_logistic_inc to compare growth shapes).
python fit_image_rebut.py \
    --image data/kodim01.png \
    --optimizer auto_cos_inc \
    --scheduler rank_wsd \
    --rank 64 \
    --rank_start 64 \
    --rank_end 256 \
    --rank_warmup_steps 2400 \
    --rank_wsd_decay_start_step 2400 \
    --rank_wsd_min_lr_ratio 0.1 \
    --epochs 3000 \
    --lr 0.03 \
    --muon_lr 3e-3 \

# Decreasing example using the same 64/256 boundaries.
python fit_image_rebut.py \
    --image data/kodim01.png \
    --optimizer auto_cos_dec \
    --scheduler rank_wsd \
    --rank 64 \
    --rank_start 256 \
    --rank_end 64 \
    --rank_warmup_steps 2400 \
    --rank_wsd_decay_start_step 2400 \
    --rank_wsd_min_lr_ratio 0.1 \
    --epochs 3000 \
    --lr 0.03 \
    --muon_lr 3e-3

# Fixed-rank example.
python fit_image_rebut.py \
    --image data/kodim01.png \
    --optimizer auto_fixed \
    --scheduler rank_wsd \
    --rank 200 \
    --rank_start 200 \
    --rank_end 200 \
    --rank_warmup_steps 2400 \
    --rank_wsd_decay_start_step 2400 \
    --rank_wsd_min_lr_ratio 0.1 \
    --epochs 3000 \
    --lr 0.03 \
    --muon_lr 3e-3
"""