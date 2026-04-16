import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dataset import build_holdout_dataset, DATASET_STATS
except ImportError:
    pass

try:
    from sinetv2 import build_sinetv2
except ImportError:
    pass


def load_model(checkpoint_path, exp_name):
    """
    Loads a model from a local checkpoint, supporting both SINetV2 (Exp 1) 
    and SegFormer (Exp 2/3).
    Prioritizes local configuration to avoid hanging on Hugging Face downloads.
    
    Args:
        checkpoint_path (str): Path to the saved model weights.
        exp_name (str): Identifier of the experiment ('exp1', 'exp2', or 'exp3').
        
    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if exp_name == 'exp1':
        model = build_sinetv2()
    else:
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                'nvidia/segformer-b2-finetuned-ade-512-512',
                num_labels=1,
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
        except Exception:
            config = SegformerConfig.from_pretrained(
                'nvidia/segformer-b2-finetuned-ade-512-512',
                num_labels=1
            )
            model = SegformerForSemanticSegmentation(config)

    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    epoch = ckpt.get('epoch', '?')
    miou = ckpt.get('val_mIoU', None)
    print(f"Successfully Loaded: {checkpoint_path}")
    print(f"  Epoch: {epoch} | val mIoU: {miou:.4f}" if miou else f"  Epoch: {epoch}")
    return model


def compute_metrics_per_image(pred_prob, mask):
    """
    Computes mIoU, F1 score (Dice), and Mean Absolute Error (MAE) for a single prediction.
    
    Args:
        pred_prob (torch.Tensor): Output probability map after sigmoid, shape [H, W].
        mask (torch.Tensor): Binary ground truth mask, shape [H, W].
        
    Returns:
        dict: A dictionary containing the scalar metrics 'mIoU', 'F1', 'MAE'.
    """
    pred_bin = (pred_prob > 0.5).float()

    inter    = (pred_bin * mask).sum().item()
    union    = pred_bin.sum().item() + mask.sum().item() - inter
    iou_fg   = (inter + 1e-6) / (union + 1e-6)

    inter_bg = ((1 - pred_bin) * (1 - mask)).sum().item()
    union_bg = (1 - pred_bin).sum().item() + (1 - mask).sum().item() - inter_bg
    iou_bg   = (inter_bg + 1e-6) / (union_bg + 1e-6)

    miou = (iou_fg + iou_bg) / 2
    f1   = (2 * inter + 1e-6) / (pred_bin.sum().item() + mask.sum().item() + 1e-6)
    mae  = (pred_prob - mask).abs().mean().item()

    return {'mIoU': miou, 'F1': f1, 'MAE': mae}


def model_forward(model, image, device):
    """
    Executes a model-agnostic forward pass, returning probability maps scaled to 512x512.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        image (torch.Tensor): Input batch of images.
        device (torch.device): Compute device correctly assigned.
        
    Returns:
        torch.Tensor: Probability prediction between [0, 1] of shape (B, 1, 512, 512).
    """
    if isinstance(model, SegformerForSemanticSegmentation):
        logits = model(pixel_values=image.to(device)).logits
        upsampled = F.interpolate(logits, size=(512, 512),
                                  mode='bilinear', align_corners=False)
        return torch.sigmoid(upsampled)
    else:
        out = model(image.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[-2:] != (512, 512):
            out = F.interpolate(out, size=(512, 512),
                                mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


def evaluate_subset(model, dataset, device):
    """
    Runs model inference on an entire dataset, returning per-image metrics and prediction tensors.
    
    Args:
        model (torch.nn.Module): Traind segmentation model.
        dataset (torch.utils.data.Dataset): The evaluation dataset yielding input tensors.
        device (torch.device): Device to perform inference on.
        
    Returns:
        list: A list of dicts with each item holding metrics ('mIoU', 'F1', 'MAE') alongside 
              original arrays ('pred_prob', 'mask', 'image', 'filename', 'dataset').
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=torch.cuda.is_available())
    results = []

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            mask  = batch['mask'].to(device)

            prob = model_forward(model, image, device)

            m = compute_metrics_per_image(prob[0].cpu(), mask[0].cpu())
            m['filename']  = batch['filename'][0]
            m['dataset']   = batch['dataset'][0]
            m['pred_prob'] = prob[0].cpu()
            m['mask']      = mask[0].cpu()
            m['image']     = image[0].cpu()
            results.append(m)

    return results


def summarise(results, label):
    """
    Extracts summary statistics (mean & standard deviation) for a list of evaluation results.
    
    Args:
        results (list): List of metric dicts returned by evaluate_subset.
        label (str): Text label describing the test slice.
        
    Returns:
        dict: Aggregated statistical metadata for mIoU, F1, and MAE parameters.
    """
    mious = [r['mIoU'] for r in results]
    f1s   = [r['F1']   for r in results]
    maes  = [r['MAE']  for r in results]
    return {
        'label': label, 'n': len(results),
        'mIoU_mean': np.mean(mious) if results else 0,
        'mIoU_std':  np.std(mious) if results else 0,
        'F1_mean':   np.mean(f1s) if results else 0,
        'F1_std':    np.std(f1s) if results else 0,
        'MAE_mean':  np.mean(maes) if results else 0,
        'MAE_std':   np.std(maes) if results else 0,
    }


def denormalize(img_tensor, dataset_name='ACD1K'):
    """
    Reverses standard image normalization to produce a viewable RGB image tensor.
    
    Args:
        img_tensor (torch.Tensor): Original normalised batch array input payload.
        dataset_name (str): Identifier mapping back to DATASET_STATS configuration.
        
    Returns:
        np.ndarray: Post-transform NumPy array structured for matplotlib plotting.
    """
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS['ACD1K'])
    mean = torch.tensor(stats['mean']).view(3, 1, 1)
    std  = torch.tensor(stats['std']).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_prediction(result, ax_row, dataset_name='ACD1K'):
    """
    Draws comparison axes mapping Input, Ground Truth Layout, Predicted Layout and an Accuracy Overlay.
    
    Args:
        result (dict): Outcome data carrying model parameters, predictions, image properties and mask.
        ax_row (list): Target axes array for direct render injection.
        dataset_name (str): Associated internal namespace dataset to pull accurate denormalisation configurations.
    """
    img  = denormalize(result['image'], dataset_name)
    mask = result['mask'][0].numpy()
    pred = (result['pred_prob'][0] > 0.5).float().numpy()

    # Input
    ax_row[0].imshow(img)
    ax_row[0].set_title('Input', fontsize=9)
    ax_row[0].axis('off')

    # Ground truth
    ax_row[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax_row[1].set_title('Ground Truth', fontsize=9)
    ax_row[1].axis('off')

    # Prediction
    ax_row[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
    ax_row[2].set_title(f"Pred (mIoU={result['mIoU']:.3f})", fontsize=9)
    ax_row[2].axis('off')

    # Overlay: TP=green, FP=red, FN=blue
    overlay = img.copy()
    tp = (pred == 1) & (mask == 1)
    fp = (pred == 1) & (mask == 0)
    fn = (pred == 0) & (mask == 1)
    alpha = 0.4
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha  # green
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha  # red
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha  # blue
    ax_row[3].imshow(overlay.clip(0, 1))
    ax_row[3].set_title('Overlay (G=TP R=FP B=FN)', fontsize=9)
    ax_row[3].axis('off')


def get_final_val_miou(exp):
    """
    Retrieves the training curve's terminal validation mIoU statistics corresponding to final stage records.
    
    Args:
        exp (dict): Single experiment metadata specifying standard internal attributes (name and label).
        
    Returns:
        tuple (list, list): Associated validation epoch arrays representing matching tracking events alongside its mIoUs.
    """
    exp_name = exp['name']
    stage_path = f"outputs/{exp_name}/final/history.json"
    stage1_path = f"outputs/{exp_name}/stage1/history.json"

    if not os.path.exists(stage_path):
        hist_dir = os.path.dirname(exp.get('checkpoint', ''))
        if os.path.basename(hist_dir) == 'final':
            stage_path = os.path.join(hist_dir, 'history.json')
            stage1_path = os.path.join(os.path.dirname(hist_dir), 'stage1', 'history.json')
        else:
            stage_path = os.path.join(hist_dir, 'history.json')
            stage1_path = os.path.join(hist_dir, 'stage1', 'history.json')

    found_hists = []
    if os.path.exists(stage1_path):
        found_hists.append(('Stage 1', stage1_path))
    if os.path.exists(stage_path):
        found_hists.append(('Final/Stage 2', stage_path))
    if not found_hists:
        alt_stage = f"outputs/{exp_name}/history.json"
        if os.path.exists(alt_stage):
            found_hists.append(('Final/Stage 2', alt_stage))

    if not found_hists:
        print(f"Skipping {exp_name}: no history.json found")
        return None, None

    stage_name, path = found_hists[-1]
    with open(path) as f:
        h = json.load(f)

    epochs = [r['epoch'] for r in h]
    val_miou = [r['val_mIoU'] for r in h]
    return epochs, val_miou


def load_history(exp):
    """
    Extrapolates comprehensive validation statistics including losses across logged outputs tracking specific runs.
    
    Args:
        exp (dict): Experiment runtime contextual data struct properties.
        
    Returns:
        tuple (list, list, list): Epochs index ranges, resulting mIoU trajectories, and relative cumulative val losses matching records.
    """
    exp_name = exp['name']
    stage_path = f"outputs/{exp_name}/final/history.json"
    stage1_path = f"outputs/{exp_name}/stage1/history.json"

    if not os.path.exists(stage_path):
        hist_dir = os.path.dirname(exp.get('checkpoint', ''))
        if os.path.basename(hist_dir) == 'final':
            stage_path = os.path.join(hist_dir, 'history.json')
            stage1_path = os.path.join(os.path.dirname(hist_dir), 'stage1', 'history.json')
        else:
            stage_path = os.path.join(hist_dir, 'history.json')
            stage1_path = os.path.join(hist_dir, 'stage1', 'history.json')

    found_hists = []
    if os.path.exists(stage1_path):
        found_hists.append(stage1_path)
    if os.path.exists(stage_path):
        found_hists.append(stage_path)
    if not found_hists:
        print(f"Skipping {exp_name}: no history found")
        return None

    path = found_hists[-1]
    with open(path) as f:
        h = json.load(f)

    epochs = [r['epoch'] for r in h]
    val_miou = [r['val_mIoU'] for r in h]
    val_loss = [r['val_loss'] for r in h]

    return epochs, val_miou, val_loss


def side_by_side_comparison(experiments, subsets, dataset_name='acd1k', stats_key='ACD1K', num_images=6, seed=42):
    """
    Calculates mapping comparisons alongside matching multi-experiment arrays formatting subplots simultaneously.
    
    Args:
        experiments (list): Tracking environment dicts wrapping evaluated pipeline parameters and models arrays.
        subsets (dict): Stored testing splits reference container objects.
        dataset_name (str): The corresponding name flag for active selection mappings.
        stats_key (str): Sub-set dataset statistics context mapping variables identifying visual configurations.
        num_images (int): Capping factor limiting rendered rows rendered simultaneously defining bounding metrics counts.
        seed (int): Reproducibility factor standardising generated outputs formats avoiding shifts on redraw cycles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    ds = subsets[dataset_name]
    n_exps = len(experiments)
    
    np.random.seed(seed)
    total_imgs = len(ds)
    
    if total_imgs == 0:
        print(f"No images found for {dataset_name}.")
        return

    indices = np.random.choice(total_imgs, min(num_images, total_imgs), replace=False)
    
    fig, axes = plt.subplots(len(indices), n_exps + 2, figsize=(4*(n_exps + 2), 4 * len(indices)))
    
    if len(indices) == 1:
        axes = [axes]
        
    for i, idx in enumerate(indices):
        item = ds[idx]
        img_tensor = item['image']
        gt_mask = item['mask'].squeeze().numpy()
        
        img_disp = denormalize(img_tensor, stats_key)
        
        axes[i][0].imshow(img_disp)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis('off')
        axes[i][0].set_ylabel(item['filename'][:20], fontsize=10, rotation=0, labelpad=50)

        axes[i][1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title("Ground Truth Mask")
        axes[i][1].axis('off')
        
        for e_idx, exp in enumerate(experiments):
            ax = axes[i][e_idx + 2]
            
            exp_results = exp.get('results', {}).get(dataset_name, [])
            target = next((r for r in exp_results if r['filename'] == item['filename']), None)
            
            if target:
                pred = (target['pred_prob'][0] > 0.5).float().numpy()
                ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"{exp['name']}\nmIoU={target['mIoU']:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Not evaluated', ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def generate_summary_tables(experiments):
    import numpy as np
    
    all_summaries = []
    for exp in experiments:
        exp_results = exp.get('results', {})
        exp_summary = []
        for name in ['acd1k', 'cod10k', 'noise']:
            if name in exp_results:
                exp_summary.append(summarise(exp_results[name], name.upper()))

        camo_results = exp_results.get('acd1k', []) + exp_results.get('cod10k', [])
        if camo_results:
            exp_summary.append(summarise(camo_results, 'ALL CAMOUFLAGE'))

        all_flat = sum(exp_results.values(), [])
        if all_flat:
            exp_summary.append(summarise(all_flat, 'FULL HOLD-OUT'))

        all_summaries.append({'name': exp['name'], 'data': exp_summary})

    print("=" * 80)
    print(f"{'Experiment':<15} {'Subset':<20} {'N':>4} {'mIoU':>8} {'F1':>8} {'MAE':>8}")
    print("-" * 80)
    for entry in all_summaries:
        for r in entry['data']:
            print(f"{entry['name']:<15} {r['label']:<20} {r['n']:>4} "
                  f"{r['mIoU_mean']:>8.4f} {r['F1_mean']:>8.4f} {r['MAE_mean']:>8.4f}")
        print("-" * 80)

    print("\n[Success Criteria — ACD1K Hold-out Comparison]")
    for entry in all_summaries:
        acd1k = next((r for r in entry['data'] if r['label'] == 'ACD1K'), None)
        if acd1k:
            status = "✅ PASS" if acd1k['mIoU_mean'] >= 0.65 and acd1k['F1_mean'] >= 0.75 else "❌ FAIL"
            print(f"  {entry['name']:<10}: mIoU={acd1k['mIoU_mean']:.4f}, F1={acd1k['F1_mean']:.4f} | {status}")


def analyze_false_positives(experiments, fp_threshold=0.01):
    for exp in experiments:
        if 'noise' in exp.get('results', {}):
            noise_results = exp['results']['noise']
            fp_count = 0
            fp_images = []
            for r in noise_results:
                pred_bin = (r['pred_prob'] > 0.5).float()
                fg_ratio = pred_bin.mean().item()
                if fg_ratio > fp_threshold:
                    fp_count += 1
                    fp_images.append((r['filename'], fg_ratio))
            fpr = fp_count / max(1, len(noise_results))
            print(f"\n--- {exp['label'].replace('\n',' ')} FPR ---")
            print(f"Noise images: {len(noise_results)}")
            print(f"False positives (>{fp_threshold*100:.0f}% FG): {fp_count}")
            print(f"FPR: {fpr:.4f} ({fpr*100:.1f}%)")
            if fp_images:
                print(f"False positive filenames:")
                for fname, ratio in fp_images[:10]:
                    print(f"  {fname}  ({ratio*100:.1f}% foreground)")
        else:
            print(f"\n--- {exp['label'].replace('\n',' ')} FPR ---")
            print("Noise subset not available - skipping FPR analysis.")


def plot_best_and_worst_detections(experiments, dataset_name='acd1k', top_k=2):
    import matplotlib.pyplot as plt
    for exp in experiments:
        if dataset_name in exp.get('results', {}):
            sorted_res = sorted(exp['results'][dataset_name], key=lambda x: x['mIoU'], reverse=True)
            
            fig, axes = plt.subplots(top_k, 4, figsize=(16, 4*top_k))
            fig.suptitle(f"Top {top_k} Best {dataset_name.upper()} Detections - {exp['name']}", fontsize=14, y=0.98)
            ax_tgt = axes if top_k > 1 else [axes]
            for i, r in enumerate(sorted_res[:top_k]):
                visualize_prediction(r, ax_tgt[i], dataset_name.upper())
                ax_tgt[i][0].set_ylabel(r['filename'][:20], fontsize=8, rotation=0, labelpad=80)
            plt.tight_layout()
            plt.show()
            
            fig, axes = plt.subplots(top_k, 4, figsize=(16, 4*top_k))
            fig.suptitle(f"Top {top_k} Worst {dataset_name.upper()} Detections - {exp['name']}", fontsize=14, y=0.98)
            ax_tgt = axes if top_k > 1 else [axes]
            for i, r in enumerate(sorted_res[-top_k:]):
                visualize_prediction(r, ax_tgt[i], dataset_name.upper())
                ax_tgt[i][0].set_ylabel(r['filename'][:20], fontsize=8, rotation=0, labelpad=80)
            plt.tight_layout()
            plt.show()


def plot_dataset_detections(experiments, dataset_name='cod10k', top_k=2):
    import matplotlib.pyplot as plt
    for exp in experiments:
        if dataset_name in exp.get('results', {}):
            sorted_res = sorted(exp['results'][dataset_name], key=lambda x: x['mIoU'], reverse=True)
            n_show = min(top_k, len(sorted_res))
            if n_show == 0: continue
            fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
            fig.suptitle(f"{dataset_name.upper()} Detections - {exp['name']}", fontsize=14, y=0.98)
            ax_tgt = axes if n_show > 1 else [axes]
            for i, r in enumerate(sorted_res[:n_show]):
                visualize_prediction(r, ax_tgt[i], dataset_name.upper())
                ax_tgt[i][0].set_ylabel(r['filename'][:20], fontsize=8, rotation=0, labelpad=80)
            plt.tight_layout()
            plt.show()

def plot_noise_false_positives(experiments, top_k=2):
    import matplotlib.pyplot as plt
    for exp in experiments:
        if 'noise' in exp.get('results', {}):
            noise_sorted = sorted(exp['results']['noise'],
                                  key=lambda x: (x['pred_prob'] > 0.5).float().mean().item(),
                                  reverse=True)
            n_show = min(top_k, len(noise_sorted))
            if n_show == 0: continue
            fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
            fig.suptitle(f"Noise FP Analysis - {exp['name']}", fontsize=14, y=0.98)
            ax_tgt = axes if n_show > 1 else [axes]
            for i, r in enumerate(noise_sorted[:n_show]):
                visualize_prediction(r, ax_tgt[i], 'COD10K') 
                fg_pct = (r['pred_prob'] > 0.5).float().mean().item() * 100
                ax_tgt[i][0].set_ylabel(f"{r['filename'][:15]}\n{fg_pct:.1f}% FG",
                                      fontsize=8, rotation=0, labelpad=80)
            plt.tight_layout()
            plt.show()

def plot_all_training_curves(experiments):
    import matplotlib.pyplot as plt
    import json
    import os
    for exp in experiments:
        exp_name = exp['name']
        stage_path = f"outputs/{exp_name}/final/history.json"
        stage1_path = f"outputs/{exp_name}/stage1/history.json"

        if not os.path.exists(stage_path):
             hist_dir = os.path.dirname(exp.get('checkpoint',''))
             if os.path.basename(hist_dir) == 'final':
                 stage_path = os.path.join(hist_dir, 'history.json')
                 stage1_path = os.path.join(os.path.dirname(hist_dir), 'stage1', 'history.json')
             else:
                 stage_path = os.path.join(hist_dir, 'history.json')
                 stage1_path = os.path.join(hist_dir, 'stage1', 'history.json')

        found_hists = []
        if os.path.exists(stage1_path): found_hists.append(('Stage 1', stage1_path))
        if os.path.exists(stage_path): found_hists.append(('Final/Stage 2', stage_path))
        if not found_hists:
            alt = f"outputs/{exp_name}/history.json"
            if os.path.exists(alt): found_hists.append(('Final/Stage 2', alt))
            
        if not found_hists:
            print(f"Skipping training curves for {exp['name']} (no history.json found)")
            continue

        fig, axes = plt.subplots(2, max(1, len(found_hists)), figsize=(7*len(found_hists), 10))
        fig.suptitle(f"Training Curves - {exp['name']}", fontsize=14)
        if len(found_hists) == 1:
            axes = [[axes[0]], [axes[1]]]

        for col, (stage_name, path) in enumerate(found_hists):
            with open(path) as f:
                h = json.load(f)
            epochs     = [r['epoch'] for r in h]
            train_loss = [r['train_loss'] for r in h]
            val_loss   = [r['val_loss'] for r in h]
            train_miou = [r['train_mIoU'] for r in h]
            val_miou   = [r['val_mIoU'] for r in h]

            axes[0][col].plot(epochs, train_loss, label='Train', linewidth=1.5)
            axes[0][col].plot(epochs, val_loss, label='Val', linewidth=1.5)
            axes[0][col].set_title(f'{stage_name} - Loss')
            axes[0][col].set_xlabel('Epoch')
            axes[0][col].set_ylabel('Loss')
            axes[0][col].legend()
            axes[0][col].grid(True, alpha=0.3)

            axes[1][col].plot(epochs, train_miou, label='Train', linewidth=1.5)
            axes[1][col].plot(epochs, val_miou, label='Val', linewidth=1.5)
            best = max(h, key=lambda x: x['val_mIoU'])
            axes[1][col].axhline(y=best['val_mIoU'], color='r', linestyle='--', alpha=0.5,
                                 label=f"Best={best['val_mIoU']:.4f} @ep{best['epoch']}")
            axes[1][col].set_title(f'{stage_name} - mIoU')
            axes[1][col].set_xlabel('Epoch')
            axes[1][col].set_ylabel('mIoU')
            axes[1][col].legend()
            axes[1][col].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_training_dynamics_comparison(experiments):
    import matplotlib.pyplot as plt
    import os
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(7, 5))
    for exp in experiments:
        epochs, val_miou = get_final_val_miou(exp)
        if epochs is None: continue
        plt.plot(epochs, val_miou, linewidth=2, label=exp['label'].replace('\n', ' '))

    plt.xlabel('Epoch')
    plt.ylabel('Validation mIoU')
    plt.scatter(43, 0.617, color='blue')
    plt.text(43, 0.62, 'Exp1 best', fontsize=9)
    plt.scatter(19, 0.870, color='orange')
    plt.text(19, 0.875, 'Exp2 best', fontsize=9)
    plt.scatter(7, 0.872, color='green')
    plt.text(7, 0.877, 'Exp3 best', fontsize=9)
    plt.title('Training Dynamics Comparison')
    plt.legend()
    plt.axhline(y=0.65, linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig_13_training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Saved -> figures/fig_13_training_curves_comparison.png')

def plot_training_dynamics_full(experiments):
    import matplotlib.pyplot as plt
    import os
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle('Training Dynamics Comparison Across Experiments', fontsize=14)
    for exp in experiments:
        data = load_history(exp)
        if data is None: continue
        epochs, val_miou, val_loss = data
        label = exp['label'].replace('\n', ' ')
        axes[0].plot(epochs, val_miou, linewidth=2, label=label)
        axes[1].plot(epochs, val_loss, linewidth=2, linestyle='--', label=label)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation mIoU')
    axes[0].set_title('Validation mIoU')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.65, linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('figures/fig_13_training_curves_full.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Saved -> figures/fig_13_training_curves_full.png')

def plot_cross_experiment_comparison(experiments):
    import matplotlib.pyplot as plt
    dims = ['acd1k', 'cod10k', 'noise', 'all']
    metrics = ['mIoU', 'F1', 'MAE']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for dim in dims:
        exp_results = {}
        for exp in experiments:
            res = exp['results'].get(dim, [])
            if dim == 'all':
                res = exp['results'].get('acd1k', []) + exp['results'].get('cod10k', []) + exp['results'].get('noise', [])
            if res:
                try:
                    s = summarise(res, dim.upper())
                    exp_results[exp['label'].replace('\n', ' ')] = {
                        'mIoU': s['mIoU_mean'],
                        'F1':   s['F1_mean'],
                        'MAE':  s['MAE_mean']
                    }
                except Exception:
                    pass

        if len(exp_results) >= 1:
            labels = list(exp_results.keys())
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.suptitle(f"Cross-Experiment Comparison - {dim.upper()}", fontsize=14)
            for i, metric in enumerate(metrics):
                values = [exp_results[l][metric] for l in labels]
                bars = axes[i].bar(labels, values, color=colors[:len(labels)], alpha=0.8, width=0.5)
                axes[i].set_title(metric, fontsize=12)
                axes[i].set_ylim(0, 1.0 if metric != 'MAE' else max(values) * 1.5 + 0.01)
                for bar, val in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f"{val:.4f}", ha='center', va='bottom', fontsize=10)
                if metric == 'mIoU' and dim != 'noise':
                    axes[i].axhline(y=0.65, color='red', linestyle='--', alpha=0.6, label='Target >= 0.65')
                    axes[i].legend(fontsize=9)
                elif metric == 'F1' and dim != 'noise':
                    axes[i].axhline(y=0.75, color='red', linestyle='--', alpha=0.6, label='Target >= 0.75')
                    axes[i].legend(fontsize=9)
                axes[i].grid(True, alpha=0.2, axis='y')
            plt.tight_layout()
            plt.show()

def plot_miou_distributions(experiments):
    import matplotlib.pyplot as plt
    import numpy as np
    for exp in experiments:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Per-Image mIoU Distribution - {exp['name']}", fontsize=14)

        if 'acd1k' in exp.get('results', {}):
            mious = [r['mIoU'] for r in exp['results']['acd1k']]
            axes[0].hist(mious, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
            axes[0].axvline(np.mean(mious), color='red', linestyle='--', label=f'Mean = {np.mean(mious):.4f}')
            axes[0].set_xlabel('mIoU')
            axes[0].set_ylabel('Count')
            axes[0].set_title('ACD1K Hold-Out')
            axes[0].legend()

        all_res = exp.get('results', {}).get('acd1k', []) + exp.get('results', {}).get('cod10k', [])
        if all_res:
            mious_all = [r['mIoU'] for r in all_res]
            axes[1].hist(mious_all, bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
            axes[1].axvline(np.mean(mious_all), color='red', linestyle='--', label=f'Mean = {np.mean(mious_all):.4f}')
            axes[1].set_xlabel('mIoU')
            axes[1].set_ylabel('Count')
            axes[1].set_title('All Hold-Out (ACD1K + COD10K)')
            axes[1].legend()

        plt.tight_layout()
        plt.show()

def save_all_results_to_json(experiments):
    import os
    import json
    for exp in experiments:
        save_results = {}
        for subset_name, results in exp.get('results', {}).items():
            save_results[subset_name] = [
                {k: v for k, v in r.items() if k not in ('pred_prob', 'mask', 'image')}
                for r in results
            ]

        exp_summary = []
        for name in ['acd1k', 'cod10k', 'noise']:
            if name in exp.get('results', {}):
                exp_summary.append(summarise(exp['results'][name], name.upper()))

        out = {
            'experiment': exp['name'],
            'checkpoint': exp['checkpoint'],
            'summary':    exp_summary,
            'per_image':  save_results,
        }

        out_dir = f"outputs/{exp['name']}/eval"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/eval_results.json"
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=lambda x: round(float(x), 6))
        print(f"Saved -> {out_path}")
