#!/usr/bin/env python3
"""
Interpolate between two trained checkpoints and evaluate accuracy along the linear path.

Quick description
- This script builds models with parameters (1-t)*W_a + t*W_b for t in [0,1] and evaluates them.
- It is device-aware (auto-detects MPS/CUDA/CPU) and recomputes BatchNorm running stats if requested.

Usage examples
1) Auto device detection, recompute BN stats, 21 points:
   python convex_combo.py \
     --ckpt_a path/to/checkpoint_A.pt \
     --ckpt_b path/to/checkpoint_B.pt \
     --model resnet20 \
     --steps 21 \
     --recompute_bn
     --data_path ./data

2) Force MPS device (macOS):
   python convex_combo.py --ckpt_a A.pt --ckpt_b B.pt --model resnet20 --device mps --recompute_bn

3) Force CPU and save results to CSV:
   python convex_combo.py --ckpt_a A.pt --ckpt_b B.pt --model resnet20 --device cpu --steps 11 --save_csv results.csv

Notes
- Checkpoints saved by train.py typically contain a 'model_state' key. The script extracts that automatically.
- If your checkpoints come from CurveNet (trained with --curve), their state dict layout is different (interleaved bend parameters). In that case:
    - Export base parameters using CurveNet.export_base_parameters for each checkpoint into plain base-model checkpoints, then run this script on those base checkpoints; OR
    - Ask me and I can modify this script to interpolate curve-format checkpoints directly.
- Use --recompute_bn to refresh BatchNorm running_mean/var using data from the train loader. This usually improves evaluation accuracy for interpolated models.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import data
import models
import utils


def load_model_state(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    return ckpt


def interp_state_dict(stateA, stateB, reference_state, t):
    new_state = {}
    for k, ref_v in reference_state.items():
        a = stateA.get(k, None)
        b = stateB.get(k, None)
        if a is None and b is None:
            new_state[k] = ref_v.clone()
            print(f"warning: key {k} missing in both checkpoints -> using reference")
            continue
        if a is None:
            print(f"warning: key {k} missing in checkpoint A -> using B for this key")
            new_state[k] = b.clone()
            continue
        if b is None:
            print(f"warning: key {k} missing in checkpoint B -> using A for this key")
            new_state[k] = a.clone()
            continue
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            new_state[k] = ref_v.clone()
            print(f"warning: key {k} is not a tensor, using reference value")
            continue
        if a.shape != b.shape:
            print(f"warning: shape mismatch for key {k}: {a.shape} vs {b.shape} -> using reference")
            new_state[k] = ref_v.clone()
            continue
        if a.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            new_state[k] = a.clone() if t < 0.5 else b.clone()
        else:
            new_state[k] = (1.0 - t) * a + t * b
    return new_state


def recompute_bn_stats(model, train_loader, device, max_batches=200):
    """
    Recompute running_mean/var for BatchNorm layers by forwarding data in train mode.
    Runs up to max_batches batches from train_loader with no_grad().
    """
    if not utils.check_bn(model):
        return
    model.to(device)
    model.train()
    seen = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            model(inputs)
            seen += 1
            if seen >= max_batches:
                break
    model.eval()


def eval_on_loader(model, loader, device, criterion):
    """
    Device-aware test loop (does not rely on repo's utils.test which may call .cuda()).
    Returns dict with 'nll','loss','accuracy'.
    """
    model.to(device)
    model.eval()
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            nll = criterion(outputs, targets)
            loss = nll.clone()
            nll_sum += nll.item() * inputs.size(0)
            loss_sum += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += inputs.size(0)
    return {
        'nll': nll_sum / total,
        'loss': loss_sum / total,
        'accuracy': correct * 100.0 / total,
    }


def detect_device(preferred):
    # preferred can be 'auto' or explicit 'cpu','cuda','mps'
    if preferred and preferred != 'auto':
        return torch.device(preferred)
    # auto detect: prefer mps, then cuda, else cpu
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_a', required=True, help='checkpoint A (path)')
    parser.add_argument('--ckpt_b', required=True, help='checkpoint B (path)')
    parser.add_argument('--model', required=True, help='model name (same as used for training)')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--transform', default='VGG')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_test', action='store_true', help='use test split directly for evaluation')
    parser.add_argument('--steps', type=int, default=21, help='number of points along [0,1] (including endpoints)')
    parser.add_argument('--device', default='auto', help="device to use: 'auto'|'cpu'|'cuda'|'mps'")
    parser.add_argument('--recompute_bn', action='store_true',
                        help='recompute BatchNorm running stats on the train set for each interpolated model')
    parser.add_argument('--bn_batches', type=int, default=200,
                        help='how many train batches to use to recompute BN stats (only if --recompute_bn)')
    parser.add_argument('--save_csv', default=None, help='optional output csv to save t,loss,accuracy')
    args = parser.parse_args()

    device = detect_device(args.device)
    print("Using device:", device)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    architecture = getattr(models, args.model)
    arch_kwargs = getattr(architecture, 'kwargs', {}) if hasattr(architecture, 'kwargs') else {}

    base_model = architecture.base(num_classes=num_classes, **arch_kwargs)
    base_model.to(device)

    stateA = load_model_state(args.ckpt_a)
    stateB = load_model_state(args.ckpt_b)

    if not isinstance(stateA, dict) or not isinstance(stateB, dict):
        print("Error: loaded checkpoints do not look like state dicts.")
        sys.exit(1)

    ts = np.linspace(0.0, 1.0, args.steps)
    results = []
    for t in ts:
        print(f"\n=== t = {t:.4f} ===")
        new_state = interp_state_dict(stateA, stateB, base_model.state_dict(), float(t))
        # load and move the model to the chosen device
        base_model.load_state_dict(new_state, strict=False)
        base_model.to(device)

        if args.recompute_bn:
            print("Recomputing BatchNorm running stats on train loader...")
            recompute_bn_stats(base_model, loaders['train'], device, max_batches=args.bn_batches)

        # Evaluate using device-aware loop
        test_res = eval_on_loader(base_model, loaders['test'], device, F.cross_entropy)
        print(f"t={t:.4f}  test_nll: {test_res['nll']:.4f}  test_acc: {test_res['accuracy']:.4f}")
        results.append({'t': float(t), 'nll': float(test_res['nll']), 'accuracy': float(test_res['accuracy'])})

    if args.save_csv:
        import csv
        keys = ['t', 'nll', 'accuracy']
        os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Wrote results to {args.save_csv}")

    print("\nDone. Summary:")
    for r in results:
        print(f" t={r['t']:.4f}  acc={r['accuracy']:.4f}  nll={r['nll']:.4f}")


if __name__ == '__main__':
    main()